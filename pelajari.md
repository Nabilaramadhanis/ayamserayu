# Materi Pembelajaran Skripsi — Sistem Prediksi Penjualan Restoran Ayam Serayu

> **Studi Kasus:** Prediksi penjualan harian makanan dan minuman pada Restoran Ayam Serayu Bekasi menggunakan metode Linear Regression dan Random Forest dengan pendekatan CRISP-DM.

---

## Daftar Isi

1. [Gambaran Umum Proyek](#1-gambaran-umum-proyek)
2. [Metodologi CRISP-DM](#2-metodologi-crisp-dm)
3. [Data dan Sumber Data](#3-data-dan-sumber-data)
4. [Pra-pemrosesan Data](#4-pra-pemrosesan-data)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Machine Learning](#6-model-machine-learning)
7. [Dua Lapisan Sistem Prediksi](#7-dua-lapisan-sistem-prediksi)
8. [Evaluasi Model](#8-evaluasi-model)
9. [Prediksi Stok Multi-Horizon](#9-prediksi-stok-multi-horizon)
10. [Arsitektur Sistem](#10-arsitektur-sistem)
11. [Teknologi dan Library](#11-teknologi-dan-library)
12. [Alur Kerja Aplikasi](#12-alur-kerja-aplikasi)
13. [Pertanyaan Potensial Sidang Skripsi](#13-pertanyaan-potensial-sidang-skripsi)

---

## 1. Gambaran Umum Proyek

### Latar Belakang
Restoran Ayam Serayu Bekasi membutuhkan sistem untuk memprediksi jumlah penjualan harian agar dapat mengoptimalkan manajemen stok bahan baku dan perencanaan operasional. Data transaksi diperoleh dari sistem POS (Point of Sale) **Pawoon**.

### Tujuan Sistem
- Memprediksi jumlah item terjual per hari untuk kategori **Makanan** dan **Minuman** secara terpisah (evaluasi akurasi model).
- Memprediksi kebutuhan stok untuk **setiap item produk** (besok, 1 minggu, 1 bulan) sebagai rekomendasi operasional.
- Membandingkan performa dua model ML: **Linear Regression** vs **Random Forest Regressor**.
- Memilih model terbaik berdasarkan nilai **RMSE** terkecil.
- Menyajikan hasil dalam antarmuka web yang mudah digunakan beserta fitur ekspor Excel.

### Dua Lapisan Fungsionalitas
Sistem bekerja dalam dua lapisan yang saling melengkapi:

| Lapisan | Target Prediksi | Tujuan | Halaman |
|---|---|---|---|
| **Lapisan 1** | Total harian per kategori (Makanan/Minuman) | Evaluasi akurasi model (MAE/RMSE) | `/prediksi` → `/hasil` |
| **Lapisan 2** | Per item produk (misal: Ayam serayu Paha) | Rekomendasi stok operasional | `/stok-besok` |

### Batasan Sistem
- Prediksi bersifat **regresi** (memprediksi nilai numerik: jumlah item), bukan klasifikasi.
- Minimal dibutuhkan **10 hari data** per kategori/item untuk menggunakan model ML.
- Data yang digunakan mencakup **Desember 2025 – Januari 2026** (±58 hari).

---

## 2. Metodologi CRISP-DM

**CRISP-DM** (Cross-Industry Standard Process for Data Mining) adalah metodologi standar dalam proyek data mining/machine learning yang terdiri dari 6 fase berulang (iteratif).

```
[Business Understanding] → [Data Understanding] → [Data Preparation]
        ↑                                                    ↓
  [Deployment]          ←         [Evaluation]       ←  [Modeling]
```

### Fase 1 — Business Understanding
**Pertanyaan bisnis:** Berapa jumlah makanan dan minuman yang akan terjual esok hari, minggu ini, dan bulan ini?

- **Tujuan bisnis:** Optimasi stok bahan baku dan perencanaan kapasitas operasional restoran.
- **Tujuan data mining:** Membangun model regresi untuk memprediksi penjualan harian per kategori dan per item produk.
- **Kriteria sukses:** Model dengan MAE dan RMSE serendah mungkin; model dengan RMSE terkecil dipilih sebagai rekomendasi.

### Fase 2 — Data Understanding
Sumber data: **Laporan Transaksi POS Pawoon** (format CSV, 35 kolom).

**Kolom relevan yang diekstrak:**
| Kolom Asli (CSV) | Peran dalam Sistem |
|---|---|
| `Tanggal & Waktu` | Kunci waktu transaksi |
| `Jumlah Produk` | Target agregasi (jumlah item terjual) |
| `Nama Produk` | Identitas produk (untuk prediksi per item) |
| `Kategori` | Filter Makanan / Minuman |
| `Harga Produk` | Informasi nilai penjualan |

### Fase 3 — Data Preparation
Lihat detail di [Bagian 4](#4-pra-pemrosesan-data).

### Fase 4 — Modeling
Lihat detail di [Bagian 6](#6-model-machine-learning) dan [Bagian 7](#7-dua-lapisan-sistem-prediksi).

### Fase 5 — Evaluation
Lihat detail di [Bagian 8](#8-evaluasi-model).

### Fase 6 — Deployment
- Sistem dikemas sebagai **aplikasi web Flask** yang dapat dijalankan secara lokal.
- Hasil prediksi ditampilkan dalam grafik interaktif, tabel perbandingan, dan rekomendasi stok.
- Hasil dapat diekspor ke **file Excel (.xlsx)** multi-sheet.
- Data prediksi tersimpan di database SQLite untuk diakses kembali tanpa re-training.

---

## 3. Data dan Sumber Data

### Sumber Data Primer
**File:** `Laporan Transaksi Penjualan Desember 2025 - Januari 2026.csv`
**Sistem POS:** Pawoon | **Outlet:** AYAM SERAYU (ID: OT-374333)
**Rentang waktu:** 5 Desember 2025 — 31 Januari 2026 (±58 hari)

### Contoh Data Transaksi Nyata
```
Tanggal & Waktu  : 31-01-2026 22:20:00
Outlet           : AYAM SERAYU
Tipe Penjualan   : Dine in
Nama Produk      : Ayam goreng serayu - Paha
Kategori         : Makanan
Jumlah Produk    : 1
Harga Produk     : 29.000
Metode Pembayaran: Tunai
```

### Daftar Produk Aktual (dari data transaksi)

**Makanan (29 item):**
| Kelompok | Produk |
|---|---|
| Ayam serayu | Ayam goreng serayu, Ayam goreng serayu - Paha, Ayam goreng serayu - Dada |
| Ayam kremes | Ayam goreng kremes, Ayam goreng kremes - Paha, Ayam goreng kremes - Dada |
| Nasi | Nasi, Nasi 1/2, Nasi Uduk., Nasi uduk 1/2 |
| Paket | Nasi+ayam goreng serayu, Nasi+ayam goreng serayu - Paha, Nasi+ayam goreng serayu - Dada |
| Sayuran | Cah kangkung, Kangkung crispy, Kangkung goreng, Jukut goreng, Pete goreng, Tumis pete |
| Lauk | Kulit cabe garam, Kulit goreng, Usus cabe garam, Usus goreng, Tahu, Tempe, Telor crispy, Nila goreng, ati ampela, Sambal |

**Minuman (10 item):** Teh manis dingin/panas, Teh tawar dingin/panas, Airmineral, Airmineral Dingin/Biasa, Air putih es, Ait putih, Es batu

### Karakteristik Data
- Tipe penjualan: **Dine in** (makan di tempat) seluruhnya
- Metode pembayaran: Tunai dan Qris Gopay
- Jam operasional: hingga pukul 22.00+
- Rentang harga per item: Rp 3.000 – Rp 29.000

---

## 4. Pra-pemrosesan Data

### Langkah 1 — Deteksi Kolom Otomatis (`detect_columns`)
Sistem menggunakan **heuristik berbasis keyword** untuk mengenali kolom CSV secara otomatis:

```python
# app.py: fungsi detect_columns()
if 'tanggal' in cl or 'waktu' in cl:   → col_map['tanggal']
if 'jumlah' in cl and 'produk' in cl:  → col_map['jumlah_produk']
if 'nama' in cl and 'produk' in cl:    → col_map['nama_produk']
if 'kategori' in cl:                   → col_map['kategori']
```

Pendekatan ini memungkinkan sistem bekerja dengan berbagai format CSV tanpa konfigurasi manual.

### Langkah 2 — Parsing Tanggal
Format tanggal dari Pawoon: `dd-mm-yyyy HH:MM:SS`

```python
df['_tgl'] = pd.to_datetime(df[col_tanggal], format='%d-%m-%Y %H:%M:%S', errors='coerce')
df = df.dropna(subset=['_tgl'])  # buang baris yang gagal parse
df['tanggal'] = df['_tgl'].dt.date  # ambil tanggal saja, buang jam
```

### Langkah 3 — Pembersihan Angka (`clean_number`)
Kolom numerik di CSV Pawoon menggunakan format `"29,000"` (tanda kutip + koma ribuan):

```python
def clean_number(val):
    val = val.replace('"', '').replace(',', '').strip()
    return int(float(val))
# "29,000" → "29000" → 29000
```

### Langkah 4 — Penyimpanan ke Database
Setiap baris CSV disimpan ke tabel `penjualan` dengan kolom: `tanggal`, `produk`, `kategori`, `jumlah`, `harga`. Setiap baris adalah satu item dalam satu transaksi.

### Langkah 5 — Agregasi Harian
**Untuk prediksi kategori** — seluruh item dalam kategori dijumlahkan per hari:
```python
daily_kat = sub.groupby('tanggal').agg(
    jumlah_terjual=('jumlah_produk', 'sum')
).reset_index()
# Hasil: 1 baris = 1 hari = total semua item Makanan/Minuman terjual
```

**Untuk prediksi per item** — setiap produk diagregasi sendiri:
```python
# app.py: fungsi build_daily_per_produk_from_db(produk)
SELECT tanggal, SUM(jumlah) as total FROM penjualan
WHERE produk=? GROUP BY tanggal ORDER BY tanggal
# Hasil: 1 baris = 1 hari = total satu produk terjual
```

### Langkah 6 — Validasi Minimum Data
```python
if len(daily) >= 10:     # minimal 10 hari data untuk ML
    daily_feat = make_features(daily)
    if len(daily_feat) >= 5:   # minimal 5 baris efektif setelah dropna
        # jalankan ML
    # fallback: gunakan rata-rata historis jika data tidak cukup
```

---

## 5. Feature Engineering

**Feature Engineering** adalah proses membuat variabel baru dari data mentah untuk meningkatkan kemampuan prediksi model.

### Dua Set Fitur (Perbedaan Penting)

Proyek ini menggunakan **dua set fitur yang berbeda** untuk dua tujuan yang berbeda:

```python
# Set 1: FEATURES (11 fitur) — untuk evaluasi model per kategori
FEATURES = [
    'hari_ke',            # ← tren linear jangka panjang
    'hari_dalam_minggu', 'hari_dalam_bulan', 'bulan', 'minggu_ke',
    'lag_1', 'lag_2', 'lag_3', 'lag_7',
    'rata_rata_3hari', 'rata_rata_7hari'
]

# Set 2: FEATURES_ITEM (9 fitur) — untuk prediksi stok per item
FEATURES_ITEM = [
    # hari_ke dan bulan DIHAPUS — alasan lihat bagian 7.2
    'hari_dalam_minggu', 'hari_dalam_bulan', 'minggu_ke',
    'lag_1', 'lag_2', 'lag_3', 'lag_7',
    'rata_rata_3hari', 'rata_rata_7hari'
]
```

### Penjelasan Setiap Fitur

#### Kelompok A — Fitur Temporal

| Fitur | Nilai | Fungsi |
|---|---|---|
| `hari_ke` | 1, 2, 3, ... N | Urutan hari sejak data pertama; menangkap tren linear jangka panjang |
| `hari_dalam_minggu` | 0=Senin ... 6=Minggu | Pola hari kerja vs akhir pekan |
| `hari_dalam_bulan` | 1 – 31 | Pola awal/akhir bulan (hari gajian) |
| `bulan` | 1 – 12 | Pola musiman bulanan |
| `minggu_ke` | 1 – 53 | Minggu ke-berapa dalam setahun |

**Mengapa `hari_ke` dan `bulan` dikeluarkan dari FEATURES_ITEM?**
> Karena data hanya mencakup 2 bulan (Desember–Januari), Linear Regression akan menetapkan koefisien sangat besar pada `hari_ke` dan `bulan` akibat tren awal data yang rendah (restoran baru mulai mencatat). Saat memprediksi hari ke-N+1, LR mengekstrapolasi tren ini secara tak terbatas — menghasilkan prediksi ratusan item yang tidak realistis. Contoh nyata: koefisien `hari_ke` = 44.69 menyebabkan prediksi Nasi mencapai 1.692 porsi padahal aktual hanya 151.

#### Kelompok B — Lag Features

| Fitur | Kode | Fungsi |
|---|---|---|
| `lag_1` | `.shift(1)` | Penjualan kemarin |
| `lag_2` | `.shift(2)` | Penjualan 2 hari lalu |
| `lag_3` | `.shift(3)` | Penjualan 3 hari lalu |
| `lag_7` | `.shift(7)` | Penjualan seminggu lalu (pola mingguan) |

**Mengapa penting?** Penjualan restoran bersifat **autokorelasi serial** — hari ramai cenderung diikuti hari ramai. Lag-7 sangat penting karena restoran ramai di hari yang sama setiap minggunya.

**Konsekuensi:** 7 baris pertama akan bernilai NaN dan dibuang (`dropna`), sehingga data efektif berkurang 7 baris.

#### Kelompok C — Rolling Mean / Moving Average

| Fitur | Kode | Fungsi |
|---|---|---|
| `rata_rata_3hari` | `.shift(1).rolling(3).mean()` | Rata-rata 3 hari sebelumnya |
| `rata_rata_7hari` | `.shift(1).rolling(7).mean()` | Rata-rata 7 hari sebelumnya |

**Catatan `.shift(1)` sebelum `.rolling()`:** Wajib untuk mencegah **data leakage** — nilai hari ini tidak boleh masuk dalam rata-rata yang digunakan untuk memprediksi hari ini sendiri.

### Variabel Target (Y)
```
Lapisan 1: jumlah_terjual = total semua item per kategori per hari
Lapisan 2: jumlah_terjual = total item produk tertentu per hari
```

---

## 6. Model Machine Learning

### Model 1 — Linear Regression (Regresi Linear)

**Formula:**
```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε
```
- Y = jumlah item terjual (target)
- X₁...Xₙ = fitur input (11 untuk kategori, 9 untuk per item)
- β = koefisien yang dipelajari dari data training
- ε = error/residual

**Karakteristik:**
- Parametrik: mengasumsikan hubungan linear antara fitur dan target.
- Cepat dilatih dan mudah diinterpretasi (koefisien dapat dibaca langsung).
- Rentan terhadap outlier dan multikolinearitas.
- **Berbahaya untuk fitur tren tak terbatas** (mis. `hari_ke`) karena mengekstrapolasi secara linear.

**Implementasi:**
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
```

### Model 2 — Random Forest Regressor

**Konsep:**
Random Forest adalah metode **ensemble** yang membangun banyak pohon keputusan (decision tree) dan mengambil rata-rata hasilnya.

```
Ŷ = rata-rata(pohon₁(X), pohon₂(X), ..., pohon_n(X))
```

**Mekanisme "Random":**
1. **Bagging:** Setiap pohon dilatih pada subset acak data (bootstrap sampling dengan penggantian).
2. **Feature Randomness:** Setiap split hanya mempertimbangkan subset acak fitur.

**Karakteristik:**
- Non-parametrik: tidak mengasumsikan bentuk hubungan tertentu.
- **Tidak mengekstrapolasi** di luar range data training — prediksi selalu dalam rentang nilai yang pernah dilihat saat training.
- Lebih robust terhadap outlier dan lebih baik untuk hubungan non-linear.
- Hyperparameter utama: `n_estimators` (jumlah pohon, default 100, dapat diatur 10–200).

**Mengapa RF dipilih sebagai prediksi utama di FEATURES_ITEM?**
> Karena RF tidak bisa mengekstrapolasi tren, sehingga prediksi per item selalu wajar (dalam batas pernah terjadi). LR tetap dihitung dan ditampilkan sebagai pembanding, tetapi RF yang digunakan sebagai dasar rekomendasi stok.

**Implementasi:**
```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

**Safety Cap (Batas Aman):**
```python
cap_max = round(float(np.max(vals[-30:])) * 2)
# Prediksi tidak boleh melebihi 2× nilai maksimum 30 hari terakhir
pred_rf = max(0, min(round(float(rf.predict(feat)[0])), cap_max))
```

---

## 7. Dua Lapisan Sistem Prediksi

### 7.1 — Lapisan 1: Evaluasi Model per Kategori

**Tujuan:** Menguji seberapa akurat model dalam mengestimasi penjualan harian total per kategori.

**Fungsi utama:** `train_eval(daily, test_size, n_estimators)`

**Split data — Chronological Split (bukan random):**
```python
# BENAR untuk time series: data awal = train, data terbaru = test
n_test  = max(1, int(len(daily) * test_size))  # mis. 20% = 10 hari terakhir
n_train = len(daily) - n_test

train = daily.iloc[:n_train]   # 80% data pertama (Desember – awal Januari)
test  = daily.iloc[n_train:]   # 20% data terakhir (22–31 Januari 2026)
```

**Mengapa chronological, bukan random?**
> Dengan random split, baris dari Desember awal (6–72 porsi/hari, saat restoran baru buka) bisa masuk ke test set. Model yang dilatih pada data volume tinggi akan membuat prediksi yang jauh di atas aktual untuk hari-hari awal tersebut — menghasilkan error semu yang tidak merepresentasikan kemampuan prediksi masa depan yang sebenarnya.

**Output fungsi:**
```python
{
  'y_test':       [aktual hari 1, aktual hari 2, ...],   # test set
  'y_pred_lr':    [prediksi LR hari 1, ...],
  'y_pred_rf':    [prediksi RF hari 1, ...],
  'tanggal_test': ['2026-01-22', '2026-01-23', ...],     # tanggal test set
  'mae_lr', 'rmse_lr', 'mae_rf', 'rmse_rf',
  'best_model',
  'aktual_list':  [seluruh data aktual],
  'tanggal_list': [seluruh tanggal]
}
```

### 7.2 — Lapisan 2: Prediksi Stok Per Item

**Tujuan:** Memberikan rekomendasi stok operasional untuk setiap produk (bukan evaluasi akurasi).

**Fungsi utama:** `predict_item_horizon(daily_raw, horizon, n_estimators, buffer_pct)`

**Perbedaan fundamental dengan Lapisan 1:**

| Aspek | Lapisan 1 | Lapisan 2 |
|---|---|---|
| Target | Total kategori (makanan/minuman) | Per produk (ayam serayu paha, dll) |
| Tujuan | Evaluasi (MAE/RMSE) | Rekomendasi stok |
| Fitur | FEATURES (11) | FEATURES_ITEM (9, tanpa hari_ke & bulan) |
| Split | Chronological 80/20 | Tidak ada split — train pada semua data |
| Prediksi utama | LR vs RF (dibandingkan) | RF saja (LR ditampilkan sebagai info) |
| Output | y_test vs y_pred | stok_anjuran untuk horizon ke depan |

**Alur prediksi per item:**
```
Data historis produk X
        ↓
make_features() → FEATURES_ITEM
        ↓
Training pada SEMUA data (tidak ada split)
     LR model + RF model
        ↓
Bangun fitur untuk "hari besok":
  hari_dalam_minggu, hari_dalam_bulan, minggu_ke,
  lag_1=nilai_terakhir, lag_2=..., lag_7=...,
  rata3=rolling_3hari, rata7=rolling_7hari
        ↓
RF.predict(fitur_besok) → capped oleh safety cap
        ↓
stok_anjuran = prediksi × (1 + buffer_pct / 100)
```

**Buffer stok:**
```python
stok_anjuran = round(prediksi * (1 + buffer_pct / 100))
# Default buffer 15%: prediksi 100 → stok anjuran 115
# Buffer adalah pengaman untuk mengantisipasi lonjakan permintaan
```

---

## 8. Evaluasi Model

### Metrik 1 — MAE (Mean Absolute Error)

**Formula:**
```
MAE = (1/n) × Σ|yᵢ - ŷᵢ|
```

**Interpretasi:**
- Rata-rata selisih mutlak antara nilai aktual dan prediksi.
- Satuan sama dengan target (jumlah item).
- Contoh: MAE = 105 → rata-rata prediksi meleset 105 item per hari (untuk total kategori makanan yang ~500–800 item/hari).
- Tidak sensitif terhadap outlier.

### Metrik 2 — RMSE (Root Mean Squared Error)

**Formula:**
```
RMSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]
```

**Interpretasi:**
- Akar kuadrat dari rata-rata kuadrat error.
- **Lebih sensitif terhadap outlier** (error besar mendapat hukuman lebih besar).
- Digunakan sebagai **kriteria pemilihan model terbaik**.

### Pemilihan Model Terbaik
```python
best = 'Linear Regression' if rmse_lr <= rmse_rf else 'Random Forest'
```

### Hasil Evaluasi Aktual (Data Desember 2025 – Januari 2026)

**Makanan** (test set: 22–31 Januari 2026, chronological split):
| Tanggal | Aktual | RF | Error RF |
|---|---|---|---|
| 22 Jan | 351 | 606 | -255 (73%) |
| 23 Jan | 523 | 699 | -176 (34%) |
| 24 Jan | 816 | 823 | -7 (1%) |
| 25 Jan | 831 | 886 | -55 (7%) |
| ... | ... | ... | ... |
- **MAE RF: 105.72** (LR: 261.32)
- **RMSE RF: 128.83** (LR: 276.19)
- Model terbaik: **Random Forest**

**Minuman** (test set: 22–31 Januari 2026):
- **MAE RF: 25.61** (LR: 71.65)
- **RMSE RF: 35.61** (LR: 77.95)
- Model terbaik: **Random Forest**

**Catatan 22 Januari:** Error besar (73%) karena penjualan hari itu unusually rendah (anomali data, bukan kelemahan model). Model yang baik pun tidak bisa memprediksi anomali/kejadian luar biasa.

### Hubungan MAE dan RMSE
- Jika RMSE >> MAE → ada outlier signifikan dalam prediksi.
- Jika RMSE ≈ MAE → error terdistribusi merata.

---

## 9. Prediksi Stok Multi-Horizon

### Tiga Pilihan Horizon

| Horizon | Keterangan | Kegunaan |
|---|---|---|
| **Besok (1 hari)** | Prediksi 1 hari ke depan | Persiapan harian |
| **1 Minggu (7 hari)** | Total prediksi 7 hari ke depan | Pembelian bahan mingguan |
| **1 Bulan (30 hari)** | Total prediksi 30 hari ke depan | Perencanaan bulanan |

### Metode Prediksi Rekursif

Untuk horizon > 1, sistem menggunakan **prediksi rekursif** (recursive multi-step forecasting):

```python
for step in range(1, horizon + 1):
    pred_date = last_date + Timedelta(days=step)
    p_rf, p_lr = _predict_one(cur_vals, pred_date)
    preds_rf.append(p_rf)
    cur_vals.append(float(p_rf))  # prediksi hari ini → jadi lag untuk hari berikutnya

total_prediksi = sum(preds_rf)   # total stok untuk seluruh periode
stok_anjuran = round(total_prediksi * (1 + buffer_pct / 100))
```

**Cara kerjanya:**
```
Hari ke-1: lag_1=aktual_terakhir → prediksi_1
Hari ke-2: lag_1=prediksi_1       → prediksi_2
Hari ke-3: lag_1=prediksi_2       → prediksi_3
...
Hari ke-7: lag_1=prediksi_6       → prediksi_7
TOTAL = prediksi_1 + prediksi_2 + ... + prediksi_7
```

**Catatan untuk horizon panjang:** Prediksi rekursif berangsur konvergen ke nilai rata-rata historis karena error dari setiap langkah terakumulasi. Ini adalah perilaku normal dan justru menghasilkan estimasi konservatif yang baik untuk perencanaan.

### Contoh Output Nyata

| Produk | Besok | Rata7/hari | 7 Hari Total | 30 Hari Total |
|---|---|---|---|---|
| Nasi | 170 | 115 | 1.051 | 3.935 |
| Ayam goreng serayu | 129 | 130 | 917 | 3.871 |
| Teh manis dingin | 63 | 43 | 294 | 1.297 |

*(stok anjuran = total × 1.15, buffer 15%)*

---

## 10. Arsitektur Sistem

### Stack Teknologi
```
Frontend  : HTML5 + CSS3 + JavaScript (Vanilla) + Chart.js 4.4.0
Backend   : Python Flask (web framework)
Database  : SQLite (file-based: ayam_serayu.db)
ML Engine : scikit-learn
Template  : Jinja2 (template engine Flask)
Export    : openpyxl (Excel .xlsx)
```

### Struktur Database SQLite

**Tabel `penjualan`** — menyimpan setiap baris transaksi dari CSV:
```sql
CREATE TABLE penjualan (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    tanggal   TEXT NOT NULL,       -- format: YYYY-MM-DD
    produk    TEXT NOT NULL,       -- nama produk (maks 60 karakter)
    kategori  TEXT DEFAULT 'Makanan',   -- 'Makanan' atau 'Minuman'
    jumlah    INTEGER NOT NULL DEFAULT 0,
    harga     INTEGER DEFAULT 0
)
```

**Tabel `prediksi_hasil`** — menyimpan hasil evaluasi model:
```sql
CREATE TABLE prediksi_hasil (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT,     -- timestamp prediksi dijalankan
    result_json TEXT      -- hasil lengkap dalam format JSON
)
```

### Struktur File Proyek
```
ayam_serayu/
├── app.py                  ← Logika utama: routes, ML, preprocessing (734 baris)
├── requirements.txt        ← Dependensi Python
├── pelajari.md             ← Dokumen ini
├── ayam_serayu.db          ← Database SQLite (auto-generated)
├── uploads/
│   └── Laporan Transaksi Penjualan Desember 2025 - Januari 2026.csv
├── static/
│   ├── css/style.css       ← Styling (warna hijau, orange, biru)
│   └── js/main.js          ← Interaktivitas frontend (auto-dismiss alert)
└── templates/
    ├── base.html           ← Layout dasar (navbar + sidebar, 5 menu)
    ├── dashboard.html      ← Halaman utama, statistik, grafik tren
    ├── input_data.html     ← Input manual & upload CSV
    ├── prediksi.html       ← Konfigurasi & jalankan prediksi kategori
    ├── hasil.html          ← Hasil evaluasi + prediksi stok per item
    └── stok_besok.html     ← Prediksi stok per item (besok/minggu/bulan)
```

### Route (Endpoint) Aplikasi

| Route | Method | Fungsi |
|---|---|---|
| `/` | GET | Dashboard: statistik, grafik tren, hasil terakhir |
| `/input-data` | GET | Halaman input data |
| `/save-data` | POST | Simpan input manual |
| `/upload-csv` | POST | Proses upload CSV Pawoon |
| `/reset-data` | POST | Hapus semua data |
| `/prediksi` | GET | Halaman konfigurasi prediksi |
| `/run-prediksi` | POST | Jalankan ML, simpan hasil |
| `/hasil` | GET | Tampilkan hasil evaluasi + stok besok |
| `/export-hasil` | GET | Download Excel hasil evaluasi |
| `/export-data` | GET | Download Excel data penjualan |
| `/stok-besok` | GET | Prediksi stok per item (horizon: 1/7/30 hari) |
| `/export-stok-besok` | GET | Download Excel prediksi stok per item |

### Alur Data Lengkap dalam Sistem
```
CSV Pawoon (upload)
      ↓
detect_columns() → temukan kolom tanggal/jumlah/produk/kategori
      ↓
parse tanggal 'dd-mm-yyyy HH:MM:SS' → dt.date
clean_number() → '29,000' → 29000
      ↓
Simpan per baris ke tabel penjualan (produk + kategori + jumlah + tanggal)
      ↓
═══════════════════════════════════════
LAPISAN 1: Evaluasi Kategori
─────────────────────────────────────
build_daily_from_db(kategori)
      → GROUP BY tanggal, SUM(jumlah)
      → make_features() → 11 fitur (FEATURES)
      ↓
train_eval(daily, test_size, n_estimators)
      → Chronological split (80% train, 20% test)
      → Train: LinearRegression + RandomForestRegressor
      → Evaluate: MAE + RMSE pada test set
      → simpan ke prediksi_hasil (JSON)
      ↓
Halaman /hasil: grafik aktual vs prediksi + tabel dengan tanggal

═══════════════════════════════════════
LAPISAN 2: Prediksi Stok Per Item
─────────────────────────────────────
build_daily_per_produk_from_db(produk)
      → GROUP BY tanggal, SUM(jumlah) WHERE produk=?
      → make_features() → 9 fitur (FEATURES_ITEM)
      ↓
predict_item_horizon(daily, horizon, n_estimators, buffer_pct)
      → Train RF + LR pada SEMUA data
      → Buat fitur untuk hari berikutnya
      → Prediksi rekursif untuk horizon hari
      → stok_anjuran = total × (1 + buffer%)
      ↓
Halaman /stok-besok: tabel per item dengan toggle Besok/Minggu/Bulan
```

---

## 11. Teknologi dan Library

| Library | Versi Min | Fungsi dalam Proyek |
|---|---|---|
| `Flask` | ≥ 2.3.0 | Web framework: routing, template rendering, session |
| `pandas` | ≥ 1.5.0 | Manipulasi data: read CSV, groupby, rolling, shift |
| `numpy` | ≥ 1.23.0 | Komputasi numerik: sqrt (RMSE), mean, max |
| `scikit-learn` | ≥ 1.2.0 | Model ML: LinearRegression, RandomForestRegressor, MAE, MSE |
| `openpyxl` | ≥ 3.1.0 | Export Excel: workbook multi-sheet, styling header |
| `sqlite3` | built-in | Database lokal: simpan transaksi & hasil prediksi |
| `Jinja2` | bawaan Flask | Template engine: render HTML dinamis |
| `Chart.js` | 4.4.0 (CDN) | Grafik interaktif di browser |

### Konsep Penting per Library

**pandas:**
```python
pd.read_csv()                          # membaca file CSV
pd.to_datetime(s, format='%d-%m-%Y')  # konversi string ke datetime
df.groupby('tanggal').agg({'col': 'sum'})  # agregasi harian
df['col'].shift(n)                     # lag features (geser n baris)
df['col'].shift(1).rolling(n).mean()  # rolling average (tanpa leakage)
df.dropna(subset=['col'])             # buang baris dengan NaN
df.iloc[:n] / df.iloc[n:]            # chronological split
```

**scikit-learn:**
```python
model.fit(X_train, y_train)           # melatih model
model.predict(X_test)                  # prediksi pada data baru
mean_absolute_error(y_true, y_pred)    # hitung MAE
mean_squared_error(y_true, y_pred)     # hitung MSE (di-sqrt untuk RMSE)
```

**Flask:**
```python
@app.route('/path', methods=['GET','POST'])  # definisi endpoint
render_template('file.html', **data)          # render template
request.form.get('key')                        # baca data form HTML
request.files.get('key')                       # baca file upload
session['key'] = value                         # simpan di session
redirect(url_for('nama_fungsi'))               # redirect ke halaman lain
send_file(buffer, as_attachment=True)          # kirim file download
```

---

## 12. Alur Kerja Aplikasi

### Halaman 1 — Dashboard (`/`)
- Statistik: total data, jumlah data makanan, jumlah data minuman.
- Grafik tren penjualan harian Makanan dan Minuman (Chart.js).
- Ringkasan hasil prediksi terakhir (MAE, RMSE, model terbaik).
- Form input data cepat dengan dropdown produk yang sesuai data nyata.
- Zona upload CSV.

### Halaman 2 — Input Data (`/input-data`)
**Dua cara input:**
1. **Input Manual:** Form dengan dropdown produk (sesuai menu Ayam Serayu), kategori, tanggal, jumlah, harga.
2. **Upload CSV:** File laporan dari Pawoon, diproses otomatis.

**Operasi lain:** Reset semua data, Export seluruh data ke Excel.

**Dropdown produk** mencakup seluruh 39 item aktual dari data transaksi (29 makanan + 10 minuman).

### Halaman 3 — Proses Prediksi (`/prediksi` → `/run-prediksi`)
**Konfigurasi:**
- **Test Size:** 10%–40% (default 20%) — persentase data terbaru untuk evaluasi.
- **n_estimators:** 10–200 (default 100) — jumlah pohon Random Forest.
- **Sumber data:** Database internal atau upload CSV baru.

Menampilkan tahapan CRISP-DM secara visual dan penjelasan 11 fitur yang digunakan.

### Halaman 4 — Hasil (`/hasil`)
**Bagian atas — Evaluasi Model:**
- Kartu MAE/RMSE per model per kategori, highlight model terbaik.
- Grafik aktual vs prediksi LR vs prediksi RF (Chart.js).
- Tabel detail dengan tanggal, nilai aktual, prediksi, dan error (warna hijau = ≤20% dari aktual).
- Kotak penjelasan: apa arti Aktual, Prediksi, dan Error.

**Bagian bawah — Prediksi Stok Besok:**
- Tabel per item (Makanan dan Minuman) dengan stok anjuran untuk keesokan harinya.
- Kolom: Terjual Hari Ini, Rata-rata 7 Hari, Prediksi LR, Prediksi RF, Stok Anjuran (+15%).
- Badge stok berwarna: merah (≥100), oranye (≥50), kuning (≥20), hijau (<20).
- Link ke halaman Stok Besok untuk konfigurasi lebih lanjut.

### Halaman 5 — Stok Besok (`/stok-besok`)
**Toggle Horizon:**
- 📅 **Besok** — prediksi 1 hari, tampilkan kolom LR dan RF terpisah.
- 📆 **1 Minggu** — prediksi rekursif 7 hari, tampilkan total + avg/hari.
- 🗓️ **1 Bulan** — prediksi rekursif 30 hari, tampilkan total + avg/hari.

**Konfigurasi:** Buffer stok (0–50%, slider) dan n_estimators (10–200, slider).

**Tabel per kategori** (Makanan & Minuman), diurutkan stok tertinggi ke terendah.

**Export Excel** menghasilkan 3 sheet: Semua Item, Makanan, Minuman — dengan filename menyertakan horizon dan tanggal akhir periode.

### Export Excel
**`/export-hasil`** — Evaluasi model:
- Sheet `Prediksi Makanan/Minuman`: No, Aktual, LR, RF per hari test.
- Sheet `Evaluasi Makanan/Minuman`: MAE, RMSE, model terbaik.

**`/export-stok-besok`** — Rekomendasi stok:
- Sheet utama: semua item dengan kolom prediksi, avg/hari, stok anjuran, metode, tanggal.
- Sheet per kategori: Makanan dan Minuman terpisah.

---

## 13. Pertanyaan Potensial Sidang Skripsi

### Tentang Metodologi CRISP-DM

**Q: Mengapa memilih CRISP-DM sebagai metodologi?**
> CRISP-DM adalah standar industri untuk proyek data mining, bersifat iteratif (tidak linear), dan tidak terikat pada tools/algoritma tertentu. Iterasi terjadi saat pengguna mengubah `test_size` atau `n_estimators`, melihat evaluasi berubah, lalu menyesuaikan lagi — ini adalah siklus Modeling → Evaluation → Modeling yang mencerminkan filosofi CRISP-DM.

---

### Tentang Data

**Q: Mengapa hanya menggunakan data 2 bulan?**
> Data yang tersedia dibatasi oleh periode ekspor POS Pawoon. Untuk penelitian selanjutnya, disarankan data minimal 6–12 bulan agar pola musiman (Lebaran, akhir tahun, dll) dapat tertangkap lebih baik oleh model.

**Q: Bagaimana sistem menangani missing values?**
> Dari dua sumber: (1) baris dengan tanggal tidak terparsing dibuang dengan `dropna(subset=['_tgl'])`; (2) baris awal yang belum memiliki nilai lag (NaN akibat shift 7 baris) dibuang dengan `dropna(subset=FEATURES)`.

**Q: Mengapa tanggal 22 Januari errornya besar?**
> Penjualan hari itu unusually rendah (351 vs rata-rata 600+), kemungkinan ada faktor eksternal. Model ML tidak bisa memprediksi anomali/kejadian luar biasa yang tidak pernah terjadi sebelumnya dalam training data. Ini keterbatasan umum semua model prediksi.

---

### Tentang Feature Engineering

**Q: Mengapa ada dua set fitur (FEATURES dan FEATURES_ITEM)?**
> Untuk evaluasi model kategori, `hari_ke` dan `bulan` berguna karena model hanya perlu mengestimasi nilai pada rentang training data. Namun untuk prediksi per item ke masa depan, `hari_ke` menyebabkan LR mengekstrapolasi tren secara tak terbatas — menghasilkan prediksi ribuan porsi yang tidak masuk akal. Dengan menghapus fitur tren tak terbatas tersebut, prediksi per item menjadi realistis.

**Q: Mengapa lag-7 dipilih?**
> Lag-7 merepresentasikan hari yang sama di minggu sebelumnya, menangkap pola mingguan (mis. Sabtu malam selalu lebih ramai dari Senin pagi). Ini sangat relevan untuk restoran.

**Q: Mengapa `.shift(1)` sebelum `.rolling()` untuk moving average?**
> Untuk mencegah **data leakage** — nilai hari ini tidak boleh masuk perhitungan rata-rata yang digunakan sebagai fitur untuk memprediksi hari itu sendiri, karena saat prediksi nilai hari ini belum diketahui.

---

### Tentang Model

**Q: Mengapa memilih Linear Regression dan Random Forest?**
> Keduanya dipilih untuk perbandingan antara model sederhana interpretable (LR) vs model kompleks akurat (RF). Perbandingan ini menjawab pertanyaan penelitian: apakah kompleksitas tambahan RF memberikan peningkatan performa yang signifikan?

**Q: Mengapa Random Forest dipilih sebagai prediksi utama untuk stok per item?**
> RF secara desain tidak bisa mengekstrapolasi di luar range nilai training. Untuk prediksi stok praktis, ini sangat penting — lebih baik prediksi konservatif yang realistis daripada prediksi optimistis yang menyebabkan overstock/understock ekstrem.

**Q: Mengapa `random_state=42`?**
> Untuk **reproducibility** — memastikan setiap kali kode dijalankan menghasilkan model yang identik (karena Random Forest menggunakan sampling acak internal). Angka 42 adalah konvensi komunitas data science, tidak memiliki makna matematis khusus.

**Q: Apakah tidak sebaiknya menggunakan ARIMA?**
> ARIMA murni tidak dapat mengintegrasikan fitur eksternal (exogenous features) seperti hari dalam minggu. Pendekatan yang digunakan — mengubah time series menjadi supervised learning dengan lag features — lebih fleksibel dan memungkinkan penambahan fitur apapun di masa depan.

---

### Tentang Evaluasi

**Q: Mengapa chronological split, bukan random split?**
> Untuk time series, chronological split lebih realistis: model hanya "melihat masa lalu" saat dievaluasi, persis seperti kondisi nyata saat model akan digunakan. Random split bisa memasukkan data masa depan ke training (data leakage temporal), membuat evaluasi terlalu optimistis.

**Q: Mengapa RMSE digunakan sebagai kriteria pemilihan model, bukan MAE?**
> RMSE memberikan penalti lebih besar untuk error yang besar (error dikuadratkan sebelum dirata-rata). Dalam manajemen stok, prediksi yang jauh meleset lebih merugikan (kehabisan stok atau pemborosan besar) dibandingkan banyak prediksi yang sedikit meleset — sehingga RMSE lebih representatif sebagai kriteria utama.

**Q: Bagaimana menginterpretasikan MAE RF = 105 untuk Makanan?**
> Total makanan terjual per hari berkisar 400–800 item. MAE 105 berarti rata-rata prediksi meleset sekitar 105 item per hari, atau sekitar 15–20% dari total. Untuk perencanaan stok dengan buffer 15%, tingkat akurasi ini masih dapat digunakan sebagai panduan.

---

### Tentang Fitur Stok Per Item

**Q: Apa itu "prediksi rekursif" pada multi-horizon?**
> Prediksi rekursif adalah teknik di mana prediksi hari ke-1 digunakan sebagai input (lag_1) untuk memprediksi hari ke-2, dan seterusnya. Ini memungkinkan prediksi untuk horizon berapa pun, meskipun error terakumulasi. Untuk horizon panjang (30 hari), prediksi cenderung konvergen ke rata-rata historis, yang justru merupakan estimasi yang wajar untuk perencanaan.

**Q: Apa itu "buffer stok" dan mengapa 15%?**
> Buffer stok adalah cadangan pengaman di atas prediksi untuk mengantisipasi lonjakan permintaan tak terduga. Nilai 15% adalah default yang dapat disesuaikan (0–50%). Buffer 15% artinya jika model memprediksi 100 porsi, sistem merekomendasikan menyiapkan 115 porsi — 15 porsi sebagai cadangan. Nilai ini dapat dinaikkan untuk item populer atau hari-hari ramai yang diantisipasi.

**Q: Mengapa safety cap ditambahkan?**
> Meskipun FEATURES_ITEM sudah menghapus fitur tren, Linear Regression tetap bisa menghasilkan prediksi di luar rentang wajar akibat interaksi antar fitur. Safety cap (maks 2× nilai tertinggi 30 hari terakhir) memastikan tidak ada rekomendasi stok yang secara matematis mustahil tercapai.

---

### Tentang Sistem

**Q: Mengapa menggunakan SQLite, bukan MySQL/PostgreSQL?**
> SQLite dipilih karena file-based (tidak perlu server terpisah), cocok untuk aplikasi lokal skala kecil. Untuk deployment multi-user atau ke server produksi, disarankan migrasi ke PostgreSQL.

**Q: Bagaimana sistem bisa digunakan tanpa koneksi internet?**
> Seluruh komponen berjalan lokal: Flask sebagai server lokal (port 5000), SQLite sebagai database file, dan scikit-learn dilatih di mesin yang sama. Satu-satunya ketergantungan eksternal adalah Chart.js dari CDN — jika tidak ada internet, grafik tidak tampil tetapi semua fungsi lain tetap berjalan.

**Q: Apa yang terjadi jika suatu item memiliki kurang dari 10 hari data?**
> Sistem menggunakan **rata-rata historis** sebagai fallback (metode `Rata-rata` di kolom Metode). Rata-rata 7 hari terakhir digunakan sebagai prediksi, tanpa ML. Ini lebih aman dan jujur daripada memaksakan ML pada data yang tidak cukup.

---

*Dokumen ini mencerminkan kondisi sistem versi terkini: prediksi dua lapisan (kategori + per item), chronological split, FEATURES_ITEM tanpa tren tak terbatas, prediksi multi-horizon rekursif, dan safety cap.*
