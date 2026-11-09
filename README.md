# WattWise – Prediksi Konsumsi Energi Rumah Tangga

Aplikasi Streamlit untuk memperkirakan konsumsi energi rumah tangga menggunakan model **Linear Regression** dari tugas Part 2.

## Isi project
- `app.py` – antarmuka dan proses prediksi.
- `model/wattwise_linear_regression.joblib` – model hasil training yang digunakan aplikasi.
- `requirements.txt` – dependensi Python.

## Menjalankan aplikasi di Windows
1. Ekstrak folder project.
2. Buka terminal/CMD di folder `WattWise_Part3`.
3. Buat virtual environment (opsional tetapi disarankan):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```
5. Jalankan:
   ```bash
   python -m streamlit run app.py
   ```

## Model
Model yang dipakai adalah `model/wattwise_linear_regression.joblib`, sebuah pipeline Scikit-learn. Karena itu format `.joblib` sesuai; `.pth` biasanya dipakai untuk model PyTorch. File model sudah disertakan di dalam folder `model`.

## Hasil evaluasi Part 2
- MAE Test: 4.0907
- RMSE Test: 5.1661
- R² Test: 0.5925

## Upload ke GitHub
Buat repository baru bernama `WattWise`, lalu jalankan dari folder project:
```bash
git init
git add .
git commit -m "Initial commit - WattWise Streamlit app"
git branch -M main
git remote add origin https://github.com/USERNAME/WattWise.git
git push -u origin main
```
Ganti `USERNAME` dengan username GitHub kamu.
