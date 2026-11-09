
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ==========================================
# KONFIGURASI WATTWISE
# ==========================================

st.set_page_config(
    page_title="WattWise | Prediksi Energi",
    page_icon="⚡",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "model" / "wattwise_model.pth"

# ==========================================
# ARSITEKTUR MODEL PYTORCH
# Harus sama dengan arsitektur saat training
# ==========================================

class WattWiseModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# MEMUAT MODEL
# ==========================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model = WattWiseModel(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


try:
    model, checkpoint = load_model()
except Exception as e:
    st.error(f"Gagal memuat model PyTorch: {e}")
    st.stop()


# ==========================================
# STYLING
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg, #eefbf4 0%, #f7fafc 55%, #eff6ff 100%
    );
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(115deg, #064e3b, #059669);
    padding: 30px 34px;
    border-radius: 22px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 12px 30px #064e3b20;
}

.hero h1 {
    color: white;
    font-size: 40px;
    margin: 3px 0 5px;
    font-weight: 800;
}

.hero p {
    color: #d1fae5;
    font-size: 16px;
    margin: 0;
}

.eyebrow {
    letter-spacing: 2px;
    font-size: 11px;
    font-weight: 700;
    color: #a7f3d0;
}

.panel {
    background: white;
    border: 1px solid #dceee5;
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 8px 24px #064e3b0b;
}

.result {
    background: linear-gradient(120deg, #dcfce7, #ecfdf5);
    border: 1px solid #a7e5c3;
    border-radius: 18px;
    padding: 24px;
    margin-top: 16px;
}

.result-label {
    font-size: 12px;
    letter-spacing: 1px;
    color: #166534;
    font-weight: 700;
}

.result-value {
    font-size: 42px;
    font-weight: 800;
    color: #065f46;
    line-height: 1.2;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background: #059669;
    color: white;
    border: 0;
    border-radius: 11px;
    font-weight: 700;
    min-height: 48px;
    width: 100%;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: #047857;
    color: white;
}

h2, h3 {
    color: #123d30;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">SMART HOME • MACHINE LEARNING</div>
    <h1>⚡ WattWise</h1>
    <p>
        Kenali dan perkirakan konsumsi energi rumah tangga
        berdasarkan kondisi rumah.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# FORM INPUT
# ==========================================

left, right = st.columns([1.25, 0.75], gap="large")

with left:
    st.markdown("### 🏠 Data Kondisi Rumah")
    st.caption(
        "Masukkan informasi rumah, kemudian jalankan prediksi."
    )

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            suhu = st.number_input(
                "Suhu Udara (°C)", -10.0, 60.0, 25.0, 0.5
            )

            kelembapan = st.number_input(
                "Kelembapan (%)", 0.0, 100.0, 50.0, 1.0
            )

            luas = st.number_input(
                "Luas Bangunan (sq ft)",
                1.0, 20000.0, 1500.0, 50.0
            )

            penghuni = st.number_input(
                "Jumlah Penghuni", 0, 50, 3
            )

            energi = st.number_input(
                "Penggunaan Energi Terbarukan",
                0.0, 100.0, 10.0, 1.0
            )

        with c2:
            hvac = st.selectbox(
                "Penggunaan HVAC", ["On", "Off"]
            )

            lampu = st.selectbox(
                "Penggunaan Lampu", ["On", "Off"]
            )

            hari_opsi = {
                "Senin": "Monday",
                "Selasa": "Tuesday",
                "Rabu": "Wednesday",
                "Kamis": "Thursday",
                "Jumat": "Friday",
                "Sabtu": "Saturday",
                "Minggu": "Sunday"
            }

            hari_label = st.selectbox(
                "Hari", list(hari_opsi.keys())
            )

            libur_label = st.selectbox(
                "Hari Libur?", ["Tidak", "Ya"]
            )

        submitted = st.form_submit_button(
            "⚡ Hitung Prediksi Konsumsi Energi"
        )


# ==========================================
# INFORMASI MODEL
# ==========================================

with right:
    st.markdown("### 📊 Informasi Model")

    st.markdown("""
    <div class="panel">
        <div style="font-size:12px;color:#64748b">
            JENIS MODEL
        </div>
        <div style="font-size:22px;color:#065f46;font-weight:800">
            Neural Network
        </div>
        <hr style="border:0;border-top:1px solid #e2e8f0"/>

        <b>Framework</b><br/>
        <span style="color:#065f46">PyTorch</span>

        <hr style="border:0;border-top:1px solid #e2e8f0"/>

        <b>Format Model</b><br/>
        <span style="color:#065f46">wattwise_model.pth</span>

        <hr style="border:0;border-top:1px solid #e2e8f0"/>

        <b>Fungsi</b><br/>
        <span style="color:#64748b">
        Memprediksi konsumsi energi rumah tangga.
        </span>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PROSES PREDIKSI
# ==========================================

if submitted:

    row = pd.DataFrame([{
        "Temperature": suhu,
        "Humidity": kelembapan,
        "SquareFootage": luas,
        "Occupancy": penghuni,
        "HVACUsage": hvac,
        "LightingUsage": lampu,
        "RenewableEnergy": energi,
        "DayOfWeek": hari_opsi[hari_label],
        "Holiday": "Yes" if libur_label == "Ya" else "No"
    }])

    try:
        # Membuat fitur kategori menjadi dummy
        X = pd.get_dummies(
            row,
            columns=checkpoint["categorical_cols"],
            dtype=float
        )

        # Menyamakan urutan dan jumlah fitur dengan training
        feature_columns = checkpoint["feature_columns"]

        X = X.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Normalisasi menggunakan nilai dari data training
        mean_values = pd.Series(
            checkpoint["mean_values"]
        ).reindex(feature_columns)

        std_values = pd.Series(
            checkpoint["std_values"]
        ).reindex(feature_columns).replace(0, 1)

        X_scaled = (X - mean_values) / std_values

        # Mengubah data menjadi tensor PyTorch
        input_tensor = torch.tensor(
            X_scaled.values,
            dtype=torch.float32
        )

        # Prediksi
        with torch.no_grad():
            prediction = model(input_tensor).item()

        st.markdown(f"""
        <div class="result">
            <div class="result-label">
                HASIL PREDIKSI KONSUMSI ENERGI
            </div>
            <div class="result-value">
                {prediction:,.2f}
            </div>
            <div style="color:#166534">
                Hasil estimasi menggunakan model PyTorch WattWise.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Lihat Detail Data Prediksi"):
            st.dataframe(
                row,
                use_container_width=True,
                hide_index=True
            )

    except Exception as exc:
        st.error(f"Prediksi gagal dijalankan: {exc}")


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Catatan: Hasil prediksi merupakan estimasi berdasarkan "
    "pola data yang digunakan saat pelatihan model."
)

st.markdown("""
<div style="text-align:center;color:#64748b;font-size:12px">
    WATTWISE • HOUSEHOLD ENERGY PREDICTION • PYTORCH
</div>
""", unsafe_allow_html=True)
