# =============================================================================
# REALTIME FETCHER
# Mengambil data hotspot (FIRMS) dan iklim (Open-Meteo) untuk inferensi
# operasional — hari ini dan beberapa hari ke depan.
#
# pip install requests openmeteo-requests requests-cache retry-requests
# =============================================================================

import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import openmeteo_requests
import requests_cache
from retry_requests import retry

# NASA FIRMS Map Key — daftar gratis di:
# https://firms.modaps.eosdis.nasa.gov/api/map_key/
# Ganti string di bawah dengan key Anda
FIRMS_MAP_KEY = "f0d7658e696acabca8867cffd513b41d"

OUTPUT_DIR = Path("outputs")


# =============================================================================
# BAGIAN 1: FETCH HOTSPOT REAL-TIME DARI NASA FIRMS
# =============================================================================
def fetch_firms_hotspot(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    days_back: int = 14,
) -> pd.DataFrame:
    """
    Ambil hotspot VIIRS S-NPP dari FIRMS API untuk N hari terakhir.
    FIRMS membatasi day_range maksimal 5 per request — fungsi ini
    otomatis membagi menjadi beberapa request lalu menggabungkan hasilnya.
    """
    from io import StringIO

    FIRMS_MAX_DAYS = 5
    area = f"{lon_min},{lat_min},{lon_max},{lat_max}"

    # Bagi days_back menjadi chunks maksimal 5 hari
    # Contoh days_back=14: chunks = [5, 5, 4]
    chunks = []
    remaining = days_back
    offset    = 0   # offset dari hari ini (0 = hari ini s.d. 5 hari lalu)
    while remaining > 0:
        chunk_days = min(remaining, FIRMS_MAX_DAYS)
        chunks.append((offset, chunk_days))
        offset    += chunk_days
        remaining -= chunk_days

    print(f"[FIRMS] Fetching {days_back} hari dalam {len(chunks)} request "
          f"(maks {FIRMS_MAX_DAYS} hari/request)...")

    all_frames = []
    for i, (off, n_days) in enumerate(chunks):
        # FIRMS API: jika offset > 0, gunakan parameter date range eksplisit
        # Format: .../day_range/{n_days}  untuk n hari terakhir dari hari ini
        # Untuk offset, gunakan date parameter: YYYY-MM-DD
        if off == 0:
            url = (
                f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                f"{FIRMS_MAP_KEY}/VIIRS_SNPP_NRT/{area}/{n_days}"
            )
        else:
            # Hitung tanggal mulai dan akhir untuk chunk ini
            date_end   = datetime.today().date() - timedelta(days=off)
            date_start = date_end - timedelta(days=n_days - 1)
            url = (
                f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                f"{FIRMS_MAP_KEY}/VIIRS_SNPP_NRT/{area}/{n_days}/"
                f"{date_start}"
            )

        print(f"  Request {i+1}/{len(chunks)}: {n_days} hari (offset={off})...")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            df_chunk = pd.read_csv(StringIO(resp.text))
            if not df_chunk.empty and "latitude" in df_chunk.columns:
                all_frames.append(df_chunk)
                print(f"    → {len(df_chunk)} baris")
            else:
                print(f"    → kosong atau format tidak dikenal")
        except requests.exceptions.RequestException as e:
            print(f"    → WARNING: request gagal ({e}), skip chunk ini")
            continue

    if not all_frames:
        print("  WARNING: Semua FIRMS request gagal atau kosong.")
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)

    # Normalisasi
    df["acq_date"]   = pd.to_datetime(df["acq_date"]).dt.normalize()
    df["confidence"] = df["confidence"].astype(str).str.strip().str.lower()
    df = df[df["confidence"].isin(["n", "h", "nominal", "high"])].copy()
    if "type" in df.columns:
        df = df[df["type"] == 0].copy()

    # Deduplikasi antar chunk
    df["lat_r"] = df["latitude"].round(2)
    df["lon_r"] = df["longitude"].round(2)
    df = df.drop_duplicates(subset=["acq_date", "lat_r", "lon_r"])

    print(f"  ✓ FIRMS total: {len(df)} hotspot unik dari {days_back} hari")
    return df.reset_index(drop=True)


# =============================================================================
# BAGIAN 2: FETCH IKLIM REAL-TIME + FORECAST DARI OPEN-METEO
# =============================================================================
def fetch_openmeteo_climate(
    lat_centers: np.ndarray,
    lon_centers: np.ndarray,
    days_back: int = 14,
    days_forward: int = 7,
) -> pd.DataFrame:
    """
    Ambil data iklim historis recent + forecast untuk semua grid center.
    Open-Meteo: ERA5-seamless (past) + ECMWF (forecast).
    Variabel disesuaikan dengan yang digunakan saat training ERA5-Land.

    Mengembalikan DataFrame: (acq_date, grid_idx, t2m_celsius, tp_mm, rh, wind_speed)
    """
    # Setup cache dan retry
    cache_session = requests_cache.CachedSession(".cache_openmeteo", expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
    om = openmeteo_requests.Client(session=retry_session)

    date_start = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    date_end   = (datetime.today() + timedelta(days=days_forward)).strftime("%Y-%m-%d")

    print(f"[Open-Meteo] Fetching climate {date_start} → {date_end} "
          f"untuk {len(lat_centers)} grid...")

    all_frames = []

    for idx, (lat, lon) in enumerate(zip(lat_centers, lon_centers)):
        params = {
            "latitude"            : lat,
            "longitude"           : lon,
            "daily"               : [
                "temperature_2m_mean",       # → t2m_celsius
                "precipitation_sum",         # → tp_mm
                "relative_humidity_2m_mean", # → rh
                "wind_speed_10m_mean",        # → wind_speed
                "et0_fao_evapotranspiration", # bonus: kekeringan proxy
            ],
            "start_date"          : date_start,
            "end_date"            : date_end,
            "timezone"            : "Asia/Jakarta",
        }

        try:
            responses = om.weather_api(
                "https://api.open-meteo.com/v1/forecast", params=params
            )
            r    = responses[0]
            daily = r.Daily()

            dates = pd.date_range(
                start=pd.Timestamp(daily.Time(), unit="s", tz="Asia/Jakarta"),
                end=pd.Timestamp(daily.TimeEnd(), unit="s", tz="Asia/Jakarta"),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ).normalize().tz_localize(None)

            df_grid = pd.DataFrame({
                "acq_date"   : dates,
                "grid_idx"   : idx,
                "t2m_celsius": daily.Variables(0).ValuesAsNumpy(),
                "tp_mm"      : daily.Variables(1).ValuesAsNumpy(),
                "rh"         : daily.Variables(2).ValuesAsNumpy(),
                "wind_speed" : daily.Variables(3).ValuesAsNumpy(),
                "et0"        : daily.Variables(4).ValuesAsNumpy(),
            })
            all_frames.append(df_grid)

        except Exception as e:
            print(f"  WARNING: Grid idx={idx} (lat={lat:.2f}, lon={lon:.2f}) gagal: {e}")
            continue

    if not all_frames:
        print("  ERROR: Semua grid gagal fetch.")
        return pd.DataFrame()

    climate = pd.concat(all_frames, ignore_index=True)
    print(f"  ✓ Open-Meteo: {len(climate)} baris ({len(all_frames)} grid berhasil)")
    return climate


# =============================================================================
# BAGIAN 3: BUILD WINDOW FEATURES UNTUK INFERENSI
# =============================================================================
def build_realtime_features(
    grid_df: pd.DataFrame,
    climate_rt: pd.DataFrame,
    firms_df: pd.DataFrame,
    merged_hist: pd.DataFrame,
    config: dict,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Bangun feature vector untuk setiap grid pada target_date.
    Menggabungkan:
      - 14 hari iklim real-time dari Open-Meteo
      - Hotspot count dari FIRMS (jika ada)
      - Data historis dari merged_dataset untuk fitur rolling yang lebih jauh

    grid_df  : zone_labels.parquet (grid_id, lat_center, lon_center, risk_score, risk_category, risk_label)
    climate_rt: output fetch_openmeteo_climate
    firms_df : output fetch_firms_hotspot
    merged_hist: merged_with_zones.parquet
    target_date: tanggal yang ingin diprediksi
    """
    window    = config["window_size"]        # 14
    feat_cols = config["feature_cols"]

    # Grid center arrays
    lat_centers = grid_df["lat_center"].values
    lon_centers = grid_df["lon_center"].values
    grid_ids    = grid_df["grid_id"].values

    # Resolusi grid
    grid_res = 0.25

    results = []

    for i, gid in enumerate(grid_ids):
        lat_c = lat_centers[i]
        lon_c = lon_centers[i]

        # ── Ambil data historis (hingga 30 hari sebelum target) ──
        hist = merged_hist[
            (merged_hist["grid_id"] == gid) &
            (merged_hist["acq_date"] < target_date)
        ].sort_values("acq_date").tail(30)

        # ── Ambil data iklim real-time untuk grid ini ──
        clim_rt = climate_rt[climate_rt["grid_idx"] == i].copy()
        clim_rt = clim_rt.sort_values("acq_date")

        # ── Ambil hotspot real-time untuk area grid ini ──
        if not firms_df.empty:
            hs_rt = firms_df[
                (firms_df["latitude"]  >= lat_c - grid_res/2) &
                (firms_df["latitude"]  <  lat_c + grid_res/2) &
                (firms_df["longitude"] >= lon_c - grid_res/2) &
                (firms_df["longitude"] <  lon_c + grid_res/2)
            ].copy()
            hs_agg = hs_rt.groupby("acq_date").agg(
                hotspot_count=("frp", "count"),
                frp_mean=("frp", "mean"),
                frp_max=("frp", "max"),
            ).reset_index()
        else:
            hs_agg = pd.DataFrame(columns=["acq_date","hotspot_count",
                                            "frp_mean","frp_max"])

        # ── Bangun date range window (target - 14 hari s.d. target - 1 hari) ──
        dates_window = pd.date_range(
            end=target_date - timedelta(days=1),
            periods=window,
            freq="D"
        )

        # ── Susun feature per hari dalam window ──
        rows_window = []
        for d in dates_window:
            row = {"acq_date": d}

            # Prioritas iklim: real-time > historis
            clim_day = clim_rt[clim_rt["acq_date"] == d]
            if not clim_day.empty:
                row["t2m_celsius"] = float(clim_day["t2m_celsius"].iloc[0])
                row["tp_mm"]       = float(clim_day["tp_mm"].iloc[0])
                row["rh"]          = float(clim_day["rh"].iloc[0])
                row["wind_speed"]  = float(clim_day["wind_speed"].iloc[0])
            else:
                hist_day = hist[hist["acq_date"] == d]
                if not hist_day.empty:
                    row["t2m_celsius"] = float(hist_day["t2m_celsius"].iloc[0])
                    row["tp_mm"]       = float(hist_day["tp_mm"].iloc[0])
                    row["rh"]          = float(hist_day["rh"].iloc[0])
                    row["wind_speed"]  = float(hist_day["wind_speed"].iloc[0])
                else:
                    # Fallback: rata-rata bulan yang sama dari historis
                    hist_month = hist[hist["acq_date"].dt.month == d.month]
                    for col in ["t2m_celsius","tp_mm","rh","wind_speed"]:
                        row[col] = float(hist_month[col].mean()) if not hist_month.empty else 0.0

            # Hotspot: real-time > historis
            hs_day = hs_agg[hs_agg["acq_date"] == d]
            if not hs_day.empty:
                row["hotspot_count"] = float(hs_day["hotspot_count"].iloc[0])
                row["frp_mean"]      = float(hs_day["frp_mean"].iloc[0])
                row["frp_max"]       = float(hs_day["frp_max"].iloc[0])
            else:
                hist_day = hist[hist["acq_date"] == d]
                row["hotspot_count"] = float(hist_day["hotspot_count"].iloc[0]) \
                                       if not hist_day.empty else 0.0
                row["frp_mean"]      = float(hist_day["frp_mean"].iloc[0]) \
                                       if not hist_day.empty else 0.0
                row["frp_max"]       = float(hist_day["frp_max"].iloc[0]) \
                                       if not hist_day.empty else 0.0

            row["fire_occurred"] = 1 if row["hotspot_count"] > 0 else 0
            rows_window.append(row)

        win_df = pd.DataFrame(rows_window).sort_values("acq_date")

        # ── Hitung fitur turunan ──
        win_df["month"]        = win_df["acq_date"].dt.month
        win_df["day_of_year"]  = win_df["acq_date"].dt.dayofyear
        win_df["dry_season"]   = win_df["month"].isin([6,7,8,9,10]).astype(int)
        elnino = {2015:1, 2019:1, 2023:1}
        win_df["elnino"]       = win_df["acq_date"].dt.year.map(elnino).fillna(0).astype(int)

        for w in [7, 14, 30]:
            win_df[f"hotspot_roll{w}"] = (
                win_df["hotspot_count"].rolling(w, min_periods=1).mean()
            )
            win_df[f"precip_roll{w}"] = (
                win_df["tp_mm"].rolling(w, min_periods=1).sum()
            )

        win_df["dry_day"] = (win_df["tp_mm"] < 1.0).astype(int)
        win_df["consec_dry_days"] = (
            win_df["dry_day"]
            .groupby((win_df["dry_day"] != win_df["dry_day"].shift()).cumsum())
            .cumcount() + 1
        ) * win_df["dry_day"]

        win_df["fire_lag1"] = win_df["fire_occurred"].shift(1).fillna(0)
        win_df["fire_lag3"] = win_df["fire_occurred"].shift(3).fillna(0)
        win_df["fire_lag7"] = win_df["fire_occurred"].shift(7).fillna(0)

        # Zona info dari grid_df
        zone_row = grid_df[grid_df["grid_id"] == gid].iloc[0]
        win_df["risk_score"]    = float(zone_row.get("risk_score", 0))
        win_df["risk_category"] = int(zone_row.get("risk_category", 1))

        # Pastikan semua feat_cols tersedia
        avail_feats = [c for c in feat_cols if c in win_df.columns]
        if len(avail_feats) < len(feat_cols):
            missing = set(feat_cols) - set(avail_feats)
            for m in missing:
                win_df[m] = 0.0

        if len(win_df) < window:
            continue

        # Simpan feature matrix + metadata
        results.append({
            "grid_id"    : int(gid),
            "lat_center" : float(lat_c),
            "lon_center" : float(lon_c),
            "risk_label" : str(zone_row.get("risk_label", "N/A")),
            "risk_score" : float(zone_row.get("risk_score", 0)),
            "_win_df"    : win_df,   # disimpan sementara untuk inference
            "_feat_cols" : feat_cols,
        })

    return results   # list of dicts, bukan DataFrame biasa


# =============================================================================
# BAGIAN 4: INFERENCE BATCH DARI REALTIME FEATURES
# =============================================================================
def realtime_inference(
    feature_list: list,
    model,
    scaler,
    config: dict,
    threshold: float = None,
) -> pd.DataFrame:
    """
    Jalankan inference LSTM untuk semua grid dari output build_realtime_features.
    """
    if threshold is None:
        threshold = config["optimal_threshold"]

    window    = config["window_size"]
    feat_cols = config["feature_cols"]

    X_list   = []
    meta_list = []

    for item in feature_list:
        win_df    = item["_win_df"]
        avail     = [c for c in feat_cols if c in win_df.columns]
        feats     = win_df[avail].tail(window).values.astype(np.float32)

        if feats.shape[0] < window:
            continue

        X_list.append(feats)
        meta_list.append({k: v for k, v in item.items()
                          if not k.startswith("_")})

    if not X_list:
        return pd.DataFrame()

    import tensorflow as tf
    X = np.array(X_list, dtype=np.float32)          # (N, window, features)
    orig_shape = X.shape
    X_2d = X.reshape(-1, X.shape[-1])
    X_2d = scaler.transform(X_2d)
    X    = X_2d.reshape(orig_shape)

    probs = model.signatures["serving_default"](
        tf.constant(X)
    )["output_0"].numpy().flatten()

    result = pd.DataFrame(meta_list)
    result["fire_prob"] = probs
    result["alert"]     = (result["fire_prob"] >= threshold).map(
        {True: "⚠️ PERINGATAN", False: "✅ Aman"}
    )

    return result.sort_values("fire_prob", ascending=False).reset_index(drop=True)