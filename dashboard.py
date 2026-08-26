# =============================================================================
# FASE 4: SISTEM PERINGATAN DINI KARHUTLA — STREAMLIT DASHBOARD v2
# Mendukung: data historis (2015–2025) + prediksi real-time & forecast
#
# Jalankan: streamlit run phase4_dashboard.py
# pip install streamlit folium streamlit-folium plotly
#             openmeteo-requests requests-cache retry-requests
# =============================================================================

import json, pickle, warnings
import numpy as np
import pandas as pd
import folium
import plotly.express as px
import streamlit as st
import tensorflow as tf
from pathlib import Path
from datetime import datetime, timedelta
from streamlit_folium import st_folium
from realtime_fetcher import (
    fetch_firms_hotspot,
    fetch_openmeteo_climate,
    build_realtime_features,
    realtime_inference,
)

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

MODEL_DIR  = Path("models")
OUTPUT_DIR = Path("outputs")

st.set_page_config(
    page_title="Sistem Peringatan Dini Karhutla",
    page_icon="🔥", layout="wide",
)

# =============================================================================
# LOAD ARTIFACTS
# =============================================================================
@st.cache_resource
def load_model():
    return tf.saved_model.load(str(MODEL_DIR / "lstm_model"))

@st.cache_resource
def load_scaler():
    with open(MODEL_DIR / "scaler_features.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_config():
    with open(MODEL_DIR / "model_config.json") as f:
        return json.load(f)

@st.cache_data
def load_data():
    merged = pd.read_parquet(OUTPUT_DIR / "merged_with_zones.parquet")
    merged["acq_date"] = pd.to_datetime(merged["acq_date"])
    zones  = pd.read_parquet(OUTPUT_DIR / "zone_labels.parquet")
    return merged, zones

# =============================================================================
# PREDICT — HISTORIS (dari merged_dataset)
# =============================================================================
def predict_historical(model, scaler, config, merged, target_date):
    window    = config["window_size"]
    feat_cols = [c for c in config["feature_cols"] if c in merged.columns]
    threshold = config["optimal_threshold"]
    results   = []

    for gid in merged["grid_id"].unique():
        gdf  = merged[merged["grid_id"] == gid].sort_values("acq_date")
        hist = gdf[gdf["acq_date"] < target_date].tail(window)
        if len(hist) < window:
            continue

        feats        = hist[feat_cols].values.astype(np.float32)
        feats_scaled = scaler.transform(feats)
        X            = feats_scaled[np.newaxis, :, :]
        prob         = float(
            model.signatures["serving_default"](tf.constant(X))["output_0"].numpy()[0][0]
        )
        last = hist.iloc[-1]
        results.append({
            "grid_id"        : int(gid),
            "lat_center"     : float(last["lat_center"]),
            "lon_center"     : float(last["lon_center"]),
            "fire_prob"      : prob,
            "risk_label"     : str(last.get("risk_label", "N/A")),
            "risk_score"     : float(last.get("risk_score", 0)),
            "t2m_celsius"    : float(last.get("t2m_celsius", 0)),
            "tp_mm"          : float(last.get("tp_mm", 0)),
            "rh"             : float(last.get("rh", 0)),
            "wind_speed"     : float(last.get("wind_speed", 0)),
            "consec_dry_days": float(last.get("consec_dry_days", 0)),
            "hotspot_roll7"  : float(last.get("hotspot_roll7", 0)),
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df["alert"] = df["fire_prob"].apply(
            lambda p: "⚠️ PERINGATAN" if p >= threshold else "✅ Aman"
        )
    return df

# =============================================================================
# PREDICT — REAL-TIME (Open-Meteo + FIRMS)
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)   # cache 1 jam
def fetch_realtime_data(lat_min, lat_max, lon_min, lon_max):
    firms_df  = fetch_firms_hotspot(lat_min, lat_max, lon_min, lon_max, days_back=10)
    return firms_df

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_climate_data(_lat_arr, _lon_arr, days_back, days_forward):
    # underscore prefix agar Streamlit tidak coba hash numpy array
    return fetch_openmeteo_climate(_lat_arr, _lon_arr, days_back, days_forward)

# =============================================================================
# FOLIUM MAP
# =============================================================================
def build_map(pred_df: pd.DataFrame) -> folium.Map:
    lat_c = pred_df["lat_center"].mean()
    lon_c = pred_df["lon_center"].mean()
    m = folium.Map(location=[lat_c, lon_c], zoom_start=8,
                   tiles="CartoDB positron")

    fg = folium.FeatureGroup(name="Prediksi Risiko", show=True)
    for _, row in pred_df.iterrows():
        p = row["fire_prob"]
        if   p >= 0.7: clr = "#8B0000"
        elif p >= 0.5: clr = "#e74c3c"
        elif p >= 0.3: clr = "#f39c12"
        elif p >= 0.1: clr = "#f1c40f"
        else:          clr = "#2ecc71"

        popup = (
            f"<b>Grid {row['grid_id']}</b><br>"
            f"<b>P(api): {p:.3f}</b><br>"
            f"<b>{row['alert']}</b><br><hr>"
            f"Suhu: {row.get('t2m_celsius',0):.1f}°C | "
            f"Hujan: {row.get('tp_mm',0):.2f}mm<br>"
            f"RH: {row.get('rh',0):.1f}% | "
            f"Angin: {row.get('wind_speed',0):.1f}m/s<br>"
            f"Zona historis: {row['risk_label']}"
        )
        folium.CircleMarker(
            location=[row["lat_center"], row["lon_center"]],
            radius=7, color=clr, fill=True,
            fill_color=clr, fill_opacity=0.85, weight=1,
            popup=folium.Popup(popup, max_width=200),
            tooltip=f"P={p:.3f} | {row['alert']}",
        ).add_to(fg)
    fg.add_to(m)

    legend = """
    <div style='position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px;border-radius:8px;
                box-shadow:2px 2px 6px rgba(0,0,0,.3);font-size:11px'>
        <b>🔥 P(Kebakaran)</b><br>
        <span style='color:#8B0000'>■</span> ≥0.7 Sangat Tinggi<br>
        <span style='color:#e74c3c'>■</span> ≥0.5 Tinggi<br>
        <span style='color:#f39c12'>■</span> ≥0.3 Sedang<br>
        <span style='color:#f1c40f'>■</span> ≥0.1 Rendah<br>
        <span style='color:#2ecc71'>■</span> &lt;0.1 Sangat Rendah
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl().add_to(m)
    return m

# =============================================================================
# MAIN
# =============================================================================
def main():
    st.title("🔥 Sistem Peringatan Dini Kebakaran Hutan dan Lahan")
    st.markdown("**Berbasis Machine Learning | VIIRS S-NPP + ERA5-Land | Riau**")
    st.divider()

    with st.spinner("Memuat model..."):
        model  = load_model()
        scaler = load_scaler()
        config = load_config()
        merged, zones = load_data()

    # Bounding box dari data
    lat_min = float(merged["lat_center"].min()) - 0.15
    lat_max = float(merged["lat_center"].max()) + 0.15
    lon_min = float(merged["lon_center"].min()) - 0.15
    lon_max = float(merged["lon_center"].max()) + 0.15

    # ── SIDEBAR ──
    st.sidebar.header("⚙️ Mode Prediksi")

    mode = st.sidebar.radio(
        "Pilih mode:",
        ["📅 Historis (2015–2025)", "🔴 Real-time & Forecast"],
        index=0,
    )

    threshold = st.sidebar.slider(
        "Threshold Peringatan",
        min_value=0.05, max_value=0.99,
        value=float(config.get("optimal_threshold", 0.5)),
        step=0.01,
        help="Turunkan untuk lebih sensitif, naikkan untuk mengurangi false alarm"
    )

    st.sidebar.divider()
    st.sidebar.markdown(f"**AUC-ROC:** 0.8917 | **Window:** 14 hari")

    pred_df    = pd.DataFrame()
    target_str = ""

    # =========================================================================
    # MODE 1: HISTORIS
    # =========================================================================
    if mode == "📅 Historis (2015–2025)":
        hist_min = (merged["acq_date"].min() + timedelta(days=14)).date()
        hist_max = merged["acq_date"].max().date()

        selected_date = st.sidebar.date_input(
            "Tanggal Prediksi",
            value=hist_max, min_value=hist_min, max_value=hist_max,
        )
        target_dt  = pd.Timestamp(selected_date)
        target_str = str(selected_date)

        with st.spinner(f"Menghitung prediksi historis {selected_date}..."):
            pred_df = predict_historical(
                model, scaler, config, merged, target_dt
            )
        pred_df["alert"] = pred_df["fire_prob"].apply(
            lambda p: "⚠️ PERINGATAN" if p >= threshold else "✅ Aman"
        )

        # Kondisi iklim hari itu
        clim_day = merged[merged["acq_date"] == target_dt]
        if not clim_day.empty:
            avg = clim_day[["t2m_celsius","tp_mm","rh",
                             "wind_speed","consec_dry_days"]].mean()
            pred_df["t2m_celsius"]     = avg["t2m_celsius"]
            pred_df["tp_mm"]           = avg["tp_mm"]
            pred_df["rh"]              = avg["rh"]
            pred_df["wind_speed"]      = avg["wind_speed"]
            pred_df["consec_dry_days"] = avg["consec_dry_days"]

    # =========================================================================
    # MODE 2: REAL-TIME & FORECAST
    # =========================================================================
    else:
        today     = datetime.today().date()
        max_fcast = today + timedelta(days=7)

        selected_date = st.sidebar.date_input(
            "Tanggal Prediksi / Forecast",
            value=today,
            min_value=today - timedelta(days=3),   # 3 hari ke belakang
            max_value=max_fcast,
        )
        target_dt  = pd.Timestamp(selected_date)
        target_str = str(selected_date)

        days_forward = max(0, (selected_date - today).days)
        is_forecast  = selected_date > today

        if is_forecast:
            st.info(
                f"📡 Mode **Forecast** — prediksi {days_forward} hari ke depan. "
                f"Data iklim dari ECMWF forecast via Open-Meteo."
            )
        else:
            st.info("📡 Mode **Real-time** — menggunakan data aktual terkini.")

        # Ambil grid aktif saja (bukan Tidak Aktif)
        active_zones = zones[zones["risk_label"] != "Tidak Aktif"].copy()
        lat_arr = active_zones["lat_center"].values
        lon_arr = active_zones["lon_center"].values

        col_fetch1, col_fetch2 = st.columns(2)

        with col_fetch1:
            with st.spinner("🛰️ Mengambil hotspot FIRMS..."):
                firms_df = fetch_realtime_data(lat_min, lat_max, lon_min, lon_max)
            if firms_df.empty:
                st.warning("⚠️ FIRMS tidak mengembalikan data. "
                           "Pastikan FIRMS_MAP_KEY di realtime_fetcher.py sudah diisi. "
                           "Prediksi lanjut tanpa data hotspot terkini.")
            else:
                st.success(f"✅ FIRMS: {len(firms_df)} hotspot aktif")

        with col_fetch2:
            with st.spinner("🌤️ Mengambil data iklim Open-Meteo..."):
                climate_rt = fetch_climate_data(
                    tuple(lat_arr), tuple(lon_arr),
                    days_back=14, days_forward=days_forward + 1
                )
            if climate_rt.empty:
                st.error("❌ Open-Meteo gagal. Periksa koneksi internet.")
                st.stop()
            else:
                st.success(f"✅ Open-Meteo: {climate_rt['acq_date'].nunique()} hari data")

        # Build features & inference
        with st.spinner("🧠 Menjalankan model LSTM..."):
            feat_list = build_realtime_features(
                grid_df     = active_zones,
                climate_rt  = climate_rt,
                firms_df    = firms_df if not firms_df.empty else pd.DataFrame(),
                merged_hist = merged,
                config      = config,
                target_date = target_dt,
            )
            pred_df = realtime_inference(feat_list, model, scaler, config, threshold)

        if pred_df.empty:
            st.error("Prediksi gagal — data tidak cukup.")
            st.stop()

        # Isi kolom iklim dari climate_rt untuk display
        clim_today_rt = climate_rt[
            climate_rt["acq_date"] == target_dt - timedelta(days=1)
        ]
        if not clim_today_rt.empty:
            avg_rt = clim_today_rt[["t2m_celsius","tp_mm",
                                    "rh","wind_speed"]].mean()
            for col in ["t2m_celsius","tp_mm","rh","wind_speed"]:
                pred_df[col] = float(avg_rt[col])

        pred_df["consec_dry_days"] = 0.0   # akan dihitung per grid di fetcher

    # =========================================================================
    # DISPLAY HASIL (sama untuk kedua mode)
    # =========================================================================
    if pred_df.empty:
        st.warning("Tidak ada hasil prediksi.")
        return

    n_alert  = (pred_df["alert"].str.contains("PERINGATAN")).sum()
    max_prob = pred_df["fire_prob"].max()
    avg_prob = pred_df["fire_prob"].mean()

    # KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("⚠️ Grid Peringatan",      f"{n_alert} / {len(pred_df)}")
    k2.metric("🔴 Probabilitas Maks",    f"{max_prob:.3f}")
    k3.metric("📈 Probabilitas Rata-rata",f"{avg_prob:.4f}")
    k4.metric("📅 Tanggal Prediksi",     target_str)

    st.divider()

    # Peta + Tabel
    col_map, col_tbl = st.columns([3, 2])

    with col_map:
        st.subheader(f"🗺️ Peta Prediksi Risiko — {target_str}")
        fmap = build_map(pred_df)
        st_folium(fmap, width=700, height=500, returned_objects=[])

    with col_tbl:
        st.subheader("🔴 Top 15 Grid Risiko Tertinggi")
        top = (pred_df.sort_values("fire_prob", ascending=False)
                      .head(15)[["grid_id","lat_center","lon_center",
                                 "fire_prob","risk_label","alert"]]
                      .rename(columns={"grid_id":"Grid","lat_center":"Lat",
                                       "lon_center":"Lon","fire_prob":"P(api)",
                                       "risk_label":"Zona Historis",
                                       "alert":"Status"}))
        top["P(api)"] = top["P(api)"].round(4)
        st.dataframe(
            top, use_container_width=True, hide_index=True,
            column_config={
                "P(api)": st.column_config.ProgressColumn(
                    "P(api)", min_value=0, max_value=1, format="%.4f"
                )
            }
        )

        # Download CSV
        csv = pred_df.drop(columns=["_win_df","_feat_cols"], errors="ignore")
        st.download_button(
            "📥 Download Hasil Prediksi (CSV)",
            data=csv.to_csv(index=False).encode(),
            file_name=f"prediksi_karhutla_{target_str}.csv",
            mime="text/csv",
        )

    st.divider()

    # Kondisi iklim
    st.subheader("🌤️ Kondisi Iklim Referensi")
    if "t2m_celsius" in pred_df.columns:
        avg = pred_df[["t2m_celsius","tp_mm","rh","wind_speed"]].mean()
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("🌡️ Suhu",        f"{avg['t2m_celsius']:.1f}°C")
        cc2.metric("🌧️ Curah Hujan", f"{avg['tp_mm']:.2f} mm")
        cc3.metric("💧 Kelembaban",   f"{avg['rh']:.1f}%")
        cc4.metric("💨 Angin",        f"{avg['wind_speed']:.1f} m/s")

    st.divider()

    # Tren historis (selalu tampil)
    st.subheader("📈 Tren Historis Hotspot (2015–2025)")
    daily = (merged.groupby("acq_date")["hotspot_count"]
                   .sum().reset_index())
    fig = px.area(daily, x="acq_date", y="hotspot_count",
                  color_discrete_sequence=["#e74c3c"],
                  labels={"acq_date":"Tanggal","hotspot_count":"Hotspot"})
    fig.update_layout(height=260, margin=dict(t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div style='text-align:center;color:#888;font-size:11px'>"
        "Sistem Peringatan Dini Karhutla | BiLSTM+LSTM | AUC-ROC 0.8917"
        "</div>", unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()