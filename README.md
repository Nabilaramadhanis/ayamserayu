# 🍗 Sistem Prediksi Penjualan Restoran Ayam Serayu Bekasi

Aplikasi web berbasis Flask menggunakan metodologi **CRISP-DM** untuk memprediksi
penjualan harian menggunakan **Linear Regression** dan **Random Forest Regressor**.

---

## 🗂️ Struktur Proyek

```
ayam_serayu/
├── app.py                  ← Aplikasi Flask utama
├── requirements.txt        ← Daftar library Python
├── ayam_serayu.db          ← Database SQLite (auto-generated)
├── uploads/                ← Folder penyimpanan CSV upload
├── static/
│   ├── css/style.css       ← Stylesheet utama
│   └── js/main.js          ← JavaScript utama
└── templates/
    ├── base.html           ← Template dasar (sidebar + navbar)
    ├── dashboard.html      ← Halaman Dashboard
    ├── input_data.html     ← Halaman Input Data
    ├── prediksi.html       ← Halaman Proses Prediksi
    └── hasil.html          ← Halaman Hasil Prediksi
```

---

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi
```bash
python app.py
```

### 3. Buka di Browser
```
http://localhost:5000
```

---

## 📋 Fitur

| Halaman         | Fitur                                                              |
|-----------------|--------------------------------------------------------------------|
| Dashboard       | Statistik total data, metode, RMSE terbaik, grafik tren, form input |
| Input Data      | Input manual harian, upload CSV POS Pawoon, reset data, export Excel |
| Proses Prediksi | Konfigurasi test_size & n_estimators, pilih sumber data, CRISP-DM info |
| Hasil           | Grafik aktual vs prediksi, tabel perbandingan, evaluasi MAE/RMSE, export Excel |

---

## 🤖 Model Machine Learning

- **Linear Regression** – parametrik, cepat, interpretable
- **Random Forest** – ensemble 100 pohon, non-parametrik

**11 Fitur:** `hari_ke`, `hari_dalam_minggu`, `hari_dalam_bulan`, `bulan`, `minggu_ke`,
`lag_1`, `lag_2`, `lag_3`, `lag_7`, `rata_rata_3hari`, `rata_rata_7hari`

**Kriteria pemilihan model terbaik:** RMSE terkecil

---

## 📂 Format CSV yang Didukung

File CSV dari **POS Pawoon** dengan kolom:
- `Tanggal & Waktu` atau `Tanggal` → format `dd-mm-yyyy HH:MM:SS`
- `Jumlah Produk` → jumlah item per transaksi
- `Nama Produk` → nama item (opsional)
- `Kategori` → kategori menu (opsional)
- `Penjualan Kotor` → nilai penjualan (opsional)
