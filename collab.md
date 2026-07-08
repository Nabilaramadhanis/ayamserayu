# Google Colab — CRISP-DM: Sistem Prediksi Penjualan Restoran Ayam Serayu

> **Studi Kasus Skripsi:** Prediksi penjualan harian makanan dan minuman pada Restoran Ayam Serayu Bekasi menggunakan Linear Regression dan Random Forest dengan metodologi CRISP-DM.
>
> **Data:** Laporan Transaksi POS Pawoon — Desember 2025 s.d. Januari 2026 (±58 hari, 26.610 transaksi)

---

## Cara Pakai di Google Colab

1. Buka [Google Colab](https://colab.research.google.com) → New Notebook
2. Upload file CSV: `Laporan Transaksi Penjualan Desember 2025 - Januari 2026.csv`
3. Copy-paste setiap blok kode ke cell baru, jalankan berurutan dari atas ke bawah
4. Atau gunakan File → Upload Notebook lalu paste seluruh isi file ini

---

## Struktur CRISP-DM

```
Phase 1: Business Understanding  → Definisi masalah & tujuan
Phase 2: Data Understanding      → Eksplorasi & statistik data
Phase 3: Data Preparation        → Pembersihan & feature engineering
Phase 4: Modeling                → Training LR & Random Forest
Phase 5: Evaluation              → MAE, RMSE, perbandingan model
Phase 6: Deployment              → Prediksi stok multi-horizon
```

---

---

# PHASE 0 — SETUP & INSTALASI

```python
# Cell 0.1 — Install library yang dibutuhkan
# (sebagian besar sudah tersedia di Colab, tapi kita pastikan versinya)
!pip install -q pandas numpy scikit-learn matplotlib seaborn openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 11

print("✅ Library berhasil dimuat")
print(f"   pandas     : {pd.__version__}")
print(f"   numpy      : {np.__version__}")
print(f"   sklearn    : {__import__('sklearn').__version__}")
```

```python
# Cell 0.2 — Upload file CSV dari komputer lokal
from google.colab import files

print("📂 Silakan upload file CSV transaksi Pawoon...")
uploaded = files.upload()

# Ambil nama file yang baru diupload
CSV_FILE = list(uploaded.keys())[0]
print(f"\n✅ File berhasil diupload: {CSV_FILE}")
```

---

---

# PHASE 1 — BUSINESS UNDERSTANDING

```python
# Cell 1.1 — Deskripsi masalah bisnis

print("=" * 65)
print("   BUSINESS UNDERSTANDING — SISTEM PREDIKSI PENJUALAN")
print("=" * 65)

print("""
📍 LATAR BELAKANG
─────────────────
Restoran Ayam Serayu Bekasi menggunakan sistem POS (Point of Sale)
Pawoon untuk mencatat setiap transaksi. Data historis transaksi
tersebut belum dimanfaatkan secara optimal untuk perencanaan
operasional, terutama manajemen stok bahan baku.

🎯 TUJUAN BISNIS
────────────────
1. Mengurangi pemborosan bahan baku akibat over-stocking
2. Mencegah kehabisan stok saat jam sibuk (under-stocking)
3. Membantu perencanaan pembelian bahan baku harian/mingguan

🔬 TUJUAN DATA MINING
──────────────────────
1. Membangun model regresi untuk memprediksi jumlah item
   terjual per hari (kategori Makanan dan Minuman)
2. Membandingkan performa Linear Regression vs Random Forest
3. Memilih model terbaik berdasarkan metrik RMSE terkecil
4. Menghasilkan rekomendasi stok per produk (besok / 7 hari / 30 hari)

📏 KRITERIA SUKSES
──────────────────
• MAE (Mean Absolute Error) sekecil mungkin → error rata-rata rendah
• RMSE (Root Mean Squared Error) terkecil  → model terpilih
• Prediksi stok per item dengan buffer safety stock (+15%)

🗂️  DUA LAPISAN SISTEM
──────────────────────
Lapisan 1: Evaluasi akurasi model pada total harian per kategori
           (Makanan vs Minuman) dengan chronological 80/20 split
Lapisan 2: Prediksi stok operasional per item produk
           (training menggunakan seluruh data tersedia)
""")
```

```python
# Cell 1.2 — Daftar produk yang dikelola sistem

PRODUK_MAKANAN = [
    # Ayam goreng serayu
    'Ayam goreng serayu',
    'Ayam goreng serayu - Paha',
    'Ayam goreng serayu - Dada',
    # Ayam goreng kremes
    'Ayam goreng kremes',
    'Ayam goreng kremes - Paha',
    'Ayam goreng kremes - Dada',
    # Nasi
    'Nasi',
    'Nasi 1/2',
    'Nasi Uduk.',
    'Nasi uduk 1/2',
    # Paket nasi + ayam
    'Nasi+ayam goreng serayu',
    'Nasi+ayam goreng serayu - Paha',
    'Nasi+ayam goreng serayu - Dada',
    # Sayuran
    'Cah kangkung',
    'Kangkung crispy',
    'Kangkung goreng',
    'Jukut goreng',
    'Pete goreng',
    'Tumis pete',
    # Lauk & lainnya
    'Kulit cabe garam',
    'Kulit goreng',
    'Usus cabe garam',
    'Usus goreng',
    'Tahu',
    'Tempe',
    'Telor crispy',
    'Nila goreng',
    'ati ampela',
    'Sambal',
]

PRODUK_MINUMAN = [
    'Teh manis dingin',
    'Teh manis panas',
    'Teh tawar dingin',
    'Teh tawar panas',
    'Airmineral',
    'Airmineral - Dingin',
    'Airmineral - Biasa',
    'Air putih es',
    'Ait putih',
    'Es batu',
]

print(f"📋 Total produk Makanan : {len(PRODUK_MAKANAN)} item")
print(f"📋 Total produk Minuman : {len(PRODUK_MINUMAN)} item")
print(f"📋 Total keseluruhan    : {len(PRODUK_MAKANAN) + len(PRODUK_MINUMAN)} item")
```

---

---

# PHASE 2 — DATA UNDERSTANDING

```python
# Cell 2.1 — Muat data mentah dari CSV Pawoon

df_raw = pd.read_csv(CSV_FILE, low_memory=False)

print(f"📊 Dimensi data mentah: {df_raw.shape[0]:,} baris × {df_raw.shape[1]} kolom")
print(f"\n📋 Nama kolom ({len(df_raw.columns)} kolom):")
for i, col in enumerate(df_raw.columns, 1):
    print(f"   {i:2d}. {col}")
```

```python
# Cell 2.2 — Tampilkan 5 baris pertama

print("📄 5 Baris pertama data mentah:")
df_raw.head()
```

```python
# Cell 2.3 — Informasi tipe data dan missing values

print("📊 Informasi tipe data & missing values:\n")
info_df = pd.DataFrame({
    'Tipe Data': df_raw.dtypes,
    'Non-Null': df_raw.count(),
    'Null Count': df_raw.isnull().sum(),
    'Null %': (df_raw.isnull().sum() / len(df_raw) * 100).round(2)
})
print(info_df.to_string())
```

```python
# Cell 2.4 — Deteksi otomatis kolom relevan (sama seperti app.py)

def detect_columns(df):
    """Deteksi otomatis mapping kolom CSV Pawoon ke field yang dibutuhkan."""
    col_map = {}
    for c in df.columns:
        cl = c.lower().replace(' ', '_').replace('&', '_').replace('/', '_')
        cl = '_'.join(filter(None, cl.split('_')))

        if 'tanggal' in cl or 'waktu' in cl or 'date' in cl or 'time' in cl:
            col_map.setdefault('tanggal', c)
        if ('jumlah' in cl and 'produk' in cl) or cl in ('qty', 'quantity', 'jumlah'):
            col_map.setdefault('jumlah_produk', c)
        if 'nama' in cl and ('produk' in cl or 'item' in cl or 'menu' in cl):
            col_map.setdefault('nama_produk', c)
        if 'kategori' in cl or 'category' in cl:
            col_map.setdefault('kategori', c)
        if 'harga' in cl and 'produk' in cl:
            col_map.setdefault('harga_produk', c)
        if 'kotor' in cl or 'gross' in cl:
            col_map.setdefault('penjualan_kotor', c)

    return col_map

col_map = detect_columns(df_raw)
print("🗺️  Mapping kolom yang terdeteksi otomatis:")
for k, v in col_map.items():
    print(f"   {k:20s} → '{v}'")
```

```python
# Cell 2.5 — Eksplorasi nilai unik kolom kunci

print("📅 Contoh nilai kolom tanggal (5 pertama):")
print(df_raw[col_map['tanggal']].head().tolist())

print(f"\n🏷️  Nilai unik Kategori:")
print(df_raw[col_map['kategori']].value_counts().to_string())

print(f"\n🍗 Contoh Nama Produk (20 pertama unik):")
print(df_raw[col_map['nama_produk']].dropna().unique()[:20].tolist())

print(f"\n📦 Contoh nilai Jumlah Produk (sebelum cleaning):")
print(df_raw[col_map['jumlah_produk']].head(10).tolist())
```

```python
# Cell 2.6 — Statistik deskriptif awal

print("📈 Statistik deskriptif kolom numerik:")
print(df_raw.describe().to_string())
```

---

---

# PHASE 3 — DATA PREPARATION

```python
# Cell 3.1 — Fungsi pembersihan data

def clean_number(val):
    """Bersihkan nilai numerik dengan format string dari CSV Pawoon.
    Contoh: '"29,000"' → 29000, '1,234' → 1234
    """
    if isinstance(val, str):
        val = val.replace('"', '').replace(',', '').strip()
    try:
        return int(float(val))
    except:
        return 0

# Test fungsi
contoh = ['"29,000"', '1,234', '100', 0, None, 'abc']
print("🧹 Uji fungsi clean_number:")
for v in contoh:
    print(f"   clean_number({repr(v):15s}) = {clean_number(v)}")
```

```python
# Cell 3.2 — Preprocessing: parse tanggal, bersihkan jumlah

df = df_raw.copy()

# Parse tanggal format Pawoon: '05-12-2025 08:30:00'
df['_tgl'] = pd.to_datetime(df[col_map['tanggal']], format='%d-%m-%Y %H:%M:%S', errors='coerce')
n_before = len(df)
df = df.dropna(subset=['_tgl'])
n_after = len(df)

print(f"📅 Parsing tanggal:")
print(f"   Sebelum  : {n_before:,} baris")
print(f"   Setelah  : {n_after:,} baris")
print(f"   Dibuang  : {n_before - n_after} baris (tanggal tidak valid)")

# Ekstrak komponen waktu
df['tanggal']  = df['_tgl'].dt.date
df['jam']      = df['_tgl'].dt.hour
df['hari']     = df['_tgl'].dt.day_name()
df['bulan_nm'] = df['_tgl'].dt.month_name()

# Bersihkan jumlah produk
df[col_map['jumlah_produk']] = df[col_map['jumlah_produk']].apply(clean_number)
df.rename(columns={
    col_map['jumlah_produk']: 'jumlah',
    col_map['nama_produk']:   'nama_produk',
    col_map['kategori']:      'kategori',
}, inplace=True)

# Normalisasi kategori: pastikan hanya 'Makanan' / 'Minuman'
df['kategori_clean'] = df['kategori'].apply(
    lambda x: 'Minuman' if 'minuman' in str(x).lower() else 'Makanan'
)

print(f"\n✅ Data setelah preprocessing: {len(df):,} baris")
print(f"\n📊 Distribusi Kategori:")
print(df['kategori_clean'].value_counts().to_string())
```

```python
# Cell 3.3 — Rentang tanggal dan statistik dasar

tgl_min = df['tanggal'].min()
tgl_max = df['tanggal'].max()
durasi  = (pd.to_datetime(tgl_max) - pd.to_datetime(tgl_min)).days + 1

print(f"📅 Rentang data: {tgl_min} → {tgl_max}")
print(f"📅 Durasi      : {durasi} hari")
print(f"📦 Total transaksi item: {len(df):,}")
print(f"🍗 Produk unik : {df['nama_produk'].nunique()}")

print(f"\n💰 Statistik Jumlah per transaksi:")
print(df['jumlah'].describe().rename({
    'count': 'Jumlah Baris', 'mean': 'Rata-rata',
    'std': 'Std Dev', 'min': 'Minimum',
    '25%': 'Kuartil 1', '50%': 'Median',
    '75%': 'Kuartil 3', 'max': 'Maksimum'
}).to_string())
```

```python
# Cell 3.4 — Visualisasi distribusi penjualan per hari

daily_all = df.groupby('tanggal').agg(total=('jumlah', 'sum')).reset_index()
daily_all['tanggal_dt'] = pd.to_datetime(daily_all['tanggal'])

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Eksplorasi Data — Distribusi Penjualan', fontsize=14, fontweight='bold')

# Plot 1: Total harian semua produk
ax1 = axes[0, 0]
ax1.plot(daily_all['tanggal_dt'], daily_all['total'], color='#2D6A2D', linewidth=2, marker='o', markersize=4)
ax1.set_title('Total Penjualan Harian (Semua Produk)')
ax1.set_xlabel('Tanggal')
ax1.set_ylabel('Jumlah Item')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Plot 2: Perbandingan kategori per hari
daily_kat = df.groupby(['tanggal', 'kategori_clean']).agg(total=('jumlah', 'sum')).reset_index()
daily_kat['tanggal_dt'] = pd.to_datetime(daily_kat['tanggal'])
for kat, color in [('Makanan', '#E74C3C'), ('Minuman', '#3498DB')]:
    sub = daily_kat[daily_kat['kategori_clean'] == kat]
    axes[0, 1].plot(sub['tanggal_dt'], sub['total'], label=kat, color=color, linewidth=2)
axes[0, 1].set_title('Penjualan Harian per Kategori')
axes[0, 1].set_xlabel('Tanggal')
axes[0, 1].set_ylabel('Jumlah Item')
axes[0, 1].legend()
axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)

# Plot 3: Penjualan per hari dalam seminggu
df['hari_num'] = pd.to_datetime(df['tanggal']).dt.dayofweek
hari_labels = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
avg_per_hari = df.groupby('hari_num')['jumlah'].sum() / durasi * 7
axes[1, 0].bar(hari_labels, avg_per_hari.values, color='#2D6A2D', edgecolor='white')
axes[1, 0].set_title('Rata-rata Penjualan per Hari dalam Seminggu')
axes[1, 0].set_ylabel('Jumlah Item (rata-rata)')

# Plot 4: Top 10 produk terlaris
top10 = df.groupby('nama_produk')['jumlah'].sum().nlargest(10)
axes[1, 1].barh(top10.index[::-1], top10.values[::-1], color='#E67E22')
axes[1, 1].set_title('Top 10 Produk Terlaris')
axes[1, 1].set_xlabel('Total Terjual')

plt.tight_layout()
plt.savefig('eksplorasi_data.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Grafik eksplorasi data disimpan sebagai 'eksplorasi_data.png'")
```

```python
# Cell 3.5 — Analisis distribusi jam penjualan

jam_dist = df.groupby('jam')['jumlah'].sum().reset_index()

plt.figure(figsize=(14, 4))
plt.bar(jam_dist['jam'], jam_dist['jumlah'], color='#8E44AD', edgecolor='white', width=0.7)
plt.title('Distribusi Penjualan per Jam', fontsize=13, fontweight='bold')
plt.xlabel('Jam')
plt.ylabel('Total Item Terjual')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig('distribusi_jam.png', dpi=150, bbox_inches='tight')
plt.show()

print("⏰ Jam tersibuk (top 5):")
print(jam_dist.nlargest(5, 'jumlah').to_string(index=False))
```

```python
# Cell 3.6 — Analisis produk per kategori

print("🍗 TOP 15 PRODUK MAKANAN:")
top_mak = (df[df['kategori_clean']=='Makanan']
           .groupby('nama_produk')['jumlah']
           .agg(['sum', 'mean', 'count'])
           .rename(columns={'sum':'Total', 'mean':'Rata-rata/hari', 'count':'Frekuensi'})
           .sort_values('Total', ascending=False)
           .head(15))
top_mak['Rata-rata/hari'] = top_mak['Rata-rata/hari'].round(1)
print(top_mak.to_string())

print("\n🥤 TOP 10 PRODUK MINUMAN:")
top_min = (df[df['kategori_clean']=='Minuman']
           .groupby('nama_produk')['jumlah']
           .agg(['sum', 'mean', 'count'])
           .rename(columns={'sum':'Total', 'mean':'Rata-rata/hari', 'count':'Frekuensi'})
           .sort_values('Total', ascending=False)
           .head(10))
top_min['Rata-rata/hari'] = top_min['Rata-rata/hari'].round(1)
print(top_min.to_string())
```

```python
# Cell 3.7 — Agregasi data harian per kategori (data siap untuk modeling)

def build_daily_kategori(df_clean, kategori):
    """Agregasi transaksi menjadi total harian per kategori."""
    sub = df_clean[df_clean['kategori_clean'] == kategori]
    daily = (sub.groupby('tanggal')
               .agg(jumlah_terjual=('jumlah', 'sum'))
               .reset_index()
               .sort_values('tanggal')
               .reset_index(drop=True))
    daily['tanggal'] = pd.to_datetime(daily['tanggal'])
    return daily

daily_makanan = build_daily_kategori(df, 'Makanan')
daily_minuman = build_daily_kategori(df, 'Minuman')

print(f"📊 Data harian Makanan : {len(daily_makanan)} hari")
print(f"   Rata-rata/hari      : {daily_makanan['jumlah_terjual'].mean():.1f} item")
print(f"   Min/Max             : {daily_makanan['jumlah_terjual'].min()} / {daily_makanan['jumlah_terjual'].max()}")

print(f"\n📊 Data harian Minuman : {len(daily_minuman)} hari")
print(f"   Rata-rata/hari      : {daily_minuman['jumlah_terjual'].mean():.1f} item")
print(f"   Min/Max             : {daily_minuman['jumlah_terjual'].min()} / {daily_minuman['jumlah_terjual'].max()}")
```

```python
# Cell 3.8 — Feature Engineering: buat 11 fitur waktu + lag + rolling average

# Kolom fitur untuk evaluasi model kategori (11 fitur)
FEATURES = [
    'hari_ke', 'hari_dalam_minggu', 'hari_dalam_bulan',
    'bulan', 'minggu_ke',
    'lag_1', 'lag_2', 'lag_3', 'lag_7',
    'rata_rata_3hari', 'rata_rata_7hari'
]

# Kolom fitur untuk prediksi per-item (9 fitur — tanpa hari_ke & bulan)
# Alasan: dengan data 2 bulan, hari_ke & bulan menyebabkan LR overfitting
# terhadap tren jangka panjang yang tidak realistis saat ekstrapolasi.
FEATURES_ITEM = [
    'hari_dalam_minggu', 'hari_dalam_bulan', 'minggu_ke',
    'lag_1', 'lag_2', 'lag_3', 'lag_7',
    'rata_rata_3hari', 'rata_rata_7hari'
]

def make_features(daily):
    """Tambahkan fitur temporal, lag, dan rolling average ke data harian."""
    daily = daily.copy().sort_values('tanggal').reset_index(drop=True)

    # Fitur temporal
    daily['hari_ke']           = range(1, len(daily) + 1)
    daily['hari_dalam_minggu'] = pd.to_datetime(daily['tanggal']).dt.dayofweek
    daily['hari_dalam_bulan']  = pd.to_datetime(daily['tanggal']).dt.day
    daily['bulan']             = pd.to_datetime(daily['tanggal']).dt.month
    daily['minggu_ke']         = pd.to_datetime(daily['tanggal']).dt.isocalendar().week.astype(int)

    # Fitur lag (autoregresif)
    for lag in [1, 2, 3, 7]:
        daily[f'lag_{lag}'] = daily['jumlah_terjual'].shift(lag)

    # Rolling average (shift 1 untuk hindari data leakage)
    daily['rata_rata_3hari'] = daily['jumlah_terjual'].shift(1).rolling(3).mean()
    daily['rata_rata_7hari'] = daily['jumlah_terjual'].shift(1).rolling(7).mean()

    return daily.dropna(subset=FEATURES + ['jumlah_terjual']).reset_index(drop=True)

# Terapkan ke data harian
feat_makanan = make_features(daily_makanan)
feat_minuman = make_features(daily_minuman)

print(f"📊 Data setelah feature engineering:")
print(f"   Makanan: {len(daily_makanan)} hari → {len(feat_makanan)} baris (setelah dropna lag)")
print(f"   Minuman: {len(daily_minuman)} hari → {len(feat_minuman)} baris (setelah dropna lag)")
print(f"\n📋 Fitur yang dibuat ({len(FEATURES)} fitur):")
for i, f in enumerate(FEATURES, 1):
    print(f"   {i:2d}. {f}")
```

```python
# Cell 3.9 — Visualisasi fitur yang dibuat

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Feature Engineering — Distribusi Fitur (Data Makanan)', fontsize=13, fontweight='bold')

fitur_plot = ['hari_dalam_minggu', 'hari_dalam_bulan', 'lag_1', 'lag_7', 'rata_rata_3hari', 'rata_rata_7hari']
judul_plot = ['Hari dalam Minggu', 'Hari dalam Bulan', 'Lag 1 Hari', 'Lag 7 Hari',
              'Rolling Avg 3 Hari', 'Rolling Avg 7 Hari']

for ax, fitur, judul in zip(axes.flat, fitur_plot, judul_plot):
    ax.hist(feat_makanan[fitur], bins=15, color='#2D6A2D', edgecolor='white', alpha=0.8)
    ax.set_title(judul)
    ax.set_xlabel('Nilai')
    ax.set_ylabel('Frekuensi')

plt.tight_layout()
plt.savefig('distribusi_fitur.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Grafik distribusi fitur disimpan sebagai 'distribusi_fitur.png'")
```

```python
# Cell 3.10 — Heatmap korelasi fitur terhadap target

plt.figure(figsize=(12, 8))
corr_data = feat_makanan[FEATURES + ['jumlah_terjual']].corr()
mask = np.triu(np.ones_like(corr_data, dtype=bool))
sns.heatmap(
    corr_data, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
    center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8},
    annot_kws={'size': 8}
)
plt.title('Korelasi Fitur — Data Makanan', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('korelasi_fitur.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Heatmap korelasi disimpan sebagai 'korelasi_fitur.png'")
```

```python
# Cell 3.11 — Train/Test Split (Chronological — bukan random)

def chronological_split(feat_df, test_size=0.2):
    """
    Bagi data secara kronologis: data lama = train, data terbaru = test.
    Ini lebih realistis untuk time series — model hanya 'melihat masa lalu'.
    Berbeda dengan random_split yang bisa 'melihat masa depan' saat training.
    """
    n_test  = max(1, int(len(feat_df) * test_size))
    n_train = len(feat_df) - n_test
    train = feat_df.iloc[:n_train].copy()
    test  = feat_df.iloc[n_train:].copy()
    return train, test

TEST_SIZE = 0.2  # 80% train, 20% test

train_mak, test_mak = chronological_split(feat_makanan, TEST_SIZE)
train_min, test_min = chronological_split(feat_minuman, TEST_SIZE)

print("📊 PEMBAGIAN DATA (Chronological Split 80/20):")
print(f"\n   Makanan:")
print(f"   Train: {len(train_mak)} hari ({train_mak['tanggal'].min().date()} → {train_mak['tanggal'].max().date()})")
print(f"   Test : {len(test_mak)} hari ({test_mak['tanggal'].min().date()} → {test_mak['tanggal'].max().date()})")
print(f"\n   Minuman:")
print(f"   Train: {len(train_min)} hari ({train_min['tanggal'].min().date()} → {train_min['tanggal'].max().date()})")
print(f"   Test : {len(test_min)} hari ({test_min['tanggal'].min().date()} → {test_min['tanggal'].max().date()})")

print(f"\n⚠️  Catatan: chronological split dipakai karena data time series.")
print(f"   Random split tidak tepat — bisa 'bocorkan' data masa depan ke training.")
```

---

---

# PHASE 4 — MODELING

```python
# Cell 4.1 — Training Linear Regression & Random Forest

N_ESTIMATORS = 100  # jumlah pohon Random Forest

def train_models(train_df, test_df, features, n_estimators=100):
    """Training Linear Regression dan Random Forest, kembalikan prediksi dan model."""
    X_train = train_df[features]
    y_train = train_df['jumlah_terjual']
    X_test  = test_df[features]
    y_test  = test_df['jumlah_terjual']

    # Model 1: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    # Model 2: Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    return lr, rf, pred_lr, pred_rf, y_test

print("⏳ Training model untuk Makanan...")
lr_mak, rf_mak, pred_lr_mak, pred_rf_mak, y_test_mak = train_models(
    train_mak, test_mak, FEATURES, N_ESTIMATORS
)
print("✅ Model Makanan selesai")

print("⏳ Training model untuk Minuman...")
lr_min, rf_min, pred_lr_min, pred_rf_min, y_test_min = train_models(
    train_min, test_min, FEATURES, N_ESTIMATORS
)
print("✅ Model Minuman selesai")
print("\n🎉 Semua model berhasil ditraining!")
```

```python
# Cell 4.2 — Hyperparameter yang digunakan

print("⚙️  KONFIGURASI MODEL")
print("=" * 50)
print(f"\n📌 Linear Regression:")
print(f"   fit_intercept = True (default)")
print(f"   normalize     = False")
print(f"   Formula       : Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε")

print(f"\n📌 Random Forest Regressor:")
print(f"   n_estimators  = {N_ESTIMATORS} (jumlah pohon keputusan)")
print(f"   random_state  = 42 (reproducibility)")
print(f"   max_features  = 'sqrt' (default untuk regresi)")
print(f"   bootstrap     = True")

print(f"\n📌 Fitur yang digunakan: {len(FEATURES)} fitur")
print(f"   {', '.join(FEATURES)}")

print(f"\n📌 Data split: Chronological 80/20")
print(f"   Train = {int((1-TEST_SIZE)*100)}% data pertama (kronologis terdahulu)")
print(f"   Test  = {int(TEST_SIZE*100)}% data terakhir (kronologis terbaru)")
```

```python
# Cell 4.3 — Feature importance Random Forest (Makanan)

importances_mak = pd.Series(rf_mak.feature_importances_, index=FEATURES).sort_values(ascending=True)
importances_min = pd.Series(rf_min.feature_importances_, index=FEATURES).sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Feature Importance — Random Forest', fontsize=13, fontweight='bold')

for ax, imp, title, color in [
    (axes[0], importances_mak, 'Makanan', '#E74C3C'),
    (axes[1], importances_min, 'Minuman', '#3498DB')
]:
    bars = ax.barh(imp.index, imp.values, color=color, alpha=0.8, edgecolor='white')
    ax.set_title(f'Kategori {title}')
    ax.set_xlabel('Importance Score')

    # Tambahkan label nilai
    for bar, val in zip(bars, imp.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', ha='left', fontsize=9)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("📊 Fitur paling penting (Makanan):")
for f, v in importances_mak.nlargest(5).items():
    print(f"   {f:20s}: {v:.4f}")
```

```python
# Cell 4.4 — Koefisien Linear Regression

coef_mak = pd.Series(lr_mak.coef_, index=FEATURES).sort_values()
coef_min = pd.Series(lr_min.coef_, index=FEATURES).sort_values()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Koefisien Linear Regression', fontsize=13, fontweight='bold')

for ax, coef, title, color in [
    (axes[0], coef_mak, 'Makanan', '#E74C3C'),
    (axes[1], coef_min, 'Minuman', '#3498DB')
]:
    colors = [color if v > 0 else '#7F8C8D' for v in coef.values]
    ax.barh(coef.index, coef.values, color=colors, edgecolor='white', alpha=0.85)
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f'Kategori {title}')
    ax.set_xlabel('Koefisien')

plt.tight_layout()
plt.savefig('koefisien_lr.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n📋 Intercept LR Makanan : {lr_mak.intercept_:.2f}")
print(f"📋 Intercept LR Minuman : {lr_min.intercept_:.2f}")
```

---

---

# PHASE 5 — EVALUATION

```python
# Cell 5.1 — Hitung metrik evaluasi MAE & RMSE

def evaluate_model(y_true, y_pred, nama_model, kategori):
    """Hitung dan tampilkan MAE & RMSE."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'model': nama_model, 'kategori': kategori,
            'MAE': round(mae, 2), 'RMSE': round(rmse, 2)}

hasil_eval = []
hasil_eval.append(evaluate_model(y_test_mak, pred_lr_mak, 'Linear Regression', 'Makanan'))
hasil_eval.append(evaluate_model(y_test_mak, pred_rf_mak, 'Random Forest',     'Makanan'))
hasil_eval.append(evaluate_model(y_test_min, pred_lr_min, 'Linear Regression', 'Minuman'))
hasil_eval.append(evaluate_model(y_test_min, pred_rf_min, 'Random Forest',     'Minuman'))

df_eval = pd.DataFrame(hasil_eval)

print("=" * 60)
print("   HASIL EVALUASI MODEL")
print("=" * 60)
print(df_eval.to_string(index=False))

print("\n🏆 MODEL TERBAIK (berdasarkan RMSE terkecil):")
for kat in ['Makanan', 'Minuman']:
    subset = df_eval[df_eval['kategori'] == kat]
    best   = subset.loc[subset['RMSE'].idxmin()]
    print(f"   {kat}: {best['model']} (RMSE={best['RMSE']}, MAE={best['MAE']})")

print("\n💡 Interpretasi:")
print("   MAE  = rata-rata selisih absolut prediksi vs aktual (satuan: item/hari)")
print("   RMSE = seperti MAE tapi menghukum kesalahan besar lebih berat")
print("   Model dengan RMSE terkecil = model terpilih untuk rekomendasi stok")
```

```python
# Cell 5.2 — Visualisasi Aktual vs Prediksi (Makanan)

fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Evaluasi Model — Aktual vs Prediksi', fontsize=14, fontweight='bold')

tanggal_test_mak = test_mak['tanggal'].values
tanggal_test_min = test_min['tanggal'].values

# Makanan
ax1 = axes[0]
ax1.plot(tanggal_test_mak, y_test_mak.values,   'ko-',  linewidth=2, markersize=5, label='Aktual', zorder=5)
ax1.plot(tanggal_test_mak, pred_lr_mak, 'b--s', linewidth=1.5, markersize=4, label='Linear Regression', alpha=0.8)
ax1.plot(tanggal_test_mak, pred_rf_mak, 'r-^',  linewidth=1.5, markersize=4, label='Random Forest', alpha=0.8)
ax1.set_title(f'Makanan — LR: MAE={df_eval.loc[0,"MAE"]}, RMSE={df_eval.loc[0,"RMSE"]} | RF: MAE={df_eval.loc[1,"MAE"]}, RMSE={df_eval.loc[1,"RMSE"]}')
ax1.set_ylabel('Jumlah Item Terjual')
ax1.legend(loc='upper left')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Minuman
ax2 = axes[1]
ax2.plot(tanggal_test_min, y_test_min.values,   'ko-',  linewidth=2, markersize=5, label='Aktual', zorder=5)
ax2.plot(tanggal_test_min, pred_lr_min, 'b--s', linewidth=1.5, markersize=4, label='Linear Regression', alpha=0.8)
ax2.plot(tanggal_test_min, pred_rf_min, 'r-^',  linewidth=1.5, markersize=4, label='Random Forest', alpha=0.8)
ax2.set_title(f'Minuman — LR: MAE={df_eval.loc[2,"MAE"]}, RMSE={df_eval.loc[2,"RMSE"]} | RF: MAE={df_eval.loc[3,"MAE"]}, RMSE={df_eval.loc[3,"RMSE"]}')
ax2.set_ylabel('Jumlah Item Terjual')
ax2.legend(loc='upper left')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('aktual_vs_prediksi.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Grafik aktual vs prediksi disimpan sebagai 'aktual_vs_prediksi.png'")
```

```python
# Cell 5.3 — Visualisasi error residual

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Analisis Residual (Error = Aktual - Prediksi)', fontsize=13, fontweight='bold')

data_pairs = [
    (y_test_mak.values, pred_lr_mak, 'Makanan — Linear Regression', '#3498DB'),
    (y_test_mak.values, pred_rf_mak, 'Makanan — Random Forest',     '#E74C3C'),
    (y_test_min.values, pred_lr_min, 'Minuman — Linear Regression', '#9B59B6'),
    (y_test_min.values, pred_rf_min, 'Minuman — Random Forest',     '#F39C12'),
]

for ax, (y_true, y_pred, title, color) in zip(axes.flat, data_pairs):
    resid = np.array(y_true) - np.array(y_pred)
    ax.scatter(y_pred, resid, color=color, alpha=0.7, edgecolor='white', s=60)
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('Nilai Prediksi')
    ax.set_ylabel('Residual')
    ax.text(0.02, 0.95, f'MAE={mean_absolute_error(y_true, y_pred):.1f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('analisis_residual.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Grafik residual disimpan sebagai 'analisis_residual.png'")
```

```python
# Cell 5.4 — Scatter Plot Aktual vs Prediksi (perfect prediction line)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Scatter Plot: Aktual vs Prediksi\n(Garis putus-putus = prediksi sempurna)',
             fontsize=13, fontweight='bold')

for ax, (y_true, y_pred, title, color) in zip(axes.flat, data_pairs):
    ax.scatter(y_true, y_pred, color=color, alpha=0.7, edgecolor='white', s=60)
    # Garis sempurna y=x
    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, label='Prediksi sempurna')
    ax.set_title(title)
    ax.set_xlabel('Aktual')
    ax.set_ylabel('Prediksi')
    r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
    ax.text(0.05, 0.92, f'R² = {r2:.3f}', transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('scatter_aktual_prediksi.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Scatter plot disimpan sebagai 'scatter_aktual_prediksi.png'")
```

```python
# Cell 5.5 — Perbandingan MAE & RMSE (bar chart)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Perbandingan Metrik Evaluasi: MAE & RMSE', fontsize=13, fontweight='bold')

for ax, metrik in [(axes[0], 'MAE'), (axes[1], 'RMSE')]:
    data_plot = {}
    for kat in ['Makanan', 'Minuman']:
        subset = df_eval[df_eval['kategori'] == kat]
        data_plot[kat] = {
            'Linear Regression': float(subset[subset['model']=='Linear Regression'][metrik]),
            'Random Forest':     float(subset[subset['model']=='Random Forest'][metrik]),
        }

    x = np.arange(2)
    width = 0.3
    bars1 = ax.bar(x - width/2, [data_plot['Makanan']['Linear Regression'],
                                   data_plot['Minuman']['Linear Regression']],
                   width, label='Linear Regression', color='#3498DB', edgecolor='white')
    bars2 = ax.bar(x + width/2, [data_plot['Makanan']['Random Forest'],
                                   data_plot['Minuman']['Random Forest']],
                   width, label='Random Forest', color='#E74C3C', edgecolor='white')

    ax.set_title(f'Perbandingan {metrik}')
    ax.set_ylabel(f'Nilai {metrik} (item/hari)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Makanan', 'Minuman'])
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('perbandingan_metrik.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Grafik perbandingan metrik disimpan sebagai 'perbandingan_metrik.png'")
```

```python
# Cell 5.6 — Tabel evaluasi lengkap dengan interpretasi

print("=" * 70)
print("   RINGKASAN EVALUASI MODEL — RESTORAN AYAM SERAYU")
print("=" * 70)

for kat in ['Makanan', 'Minuman']:
    subset  = df_eval[df_eval['kategori'] == kat]
    lr_row  = subset[subset['model'] == 'Linear Regression'].iloc[0]
    rf_row  = subset[subset['model'] == 'Random Forest'].iloc[0]
    best    = 'Random Forest' if rf_row['RMSE'] <= lr_row['RMSE'] else 'Linear Regression'
    best_row = rf_row if best == 'Random Forest' else lr_row

    daily_avg = daily_makanan['jumlah_terjual'].mean() if kat == 'Makanan' else daily_minuman['jumlah_terjual'].mean()
    mae_pct = best_row['MAE'] / daily_avg * 100

    print(f"\n📊 Kategori: {kat}")
    print(f"   Rata-rata penjualan harian : {daily_avg:.1f} item/hari")
    print(f"   {'Model':20s} {'MAE':>10s} {'RMSE':>10s} {'Status':>15s}")
    print(f"   {'-'*60}")
    star_lr = ' ⭐ Terpilih' if best == 'Linear Regression' else ''
    star_rf = ' ⭐ Terpilih' if best == 'Random Forest'     else ''
    print(f"   {'Linear Regression':20s} {lr_row['MAE']:>10.2f} {lr_row['RMSE']:>10.2f}{star_lr}")
    print(f"   {'Random Forest':20s} {rf_row['MAE']:>10.2f} {rf_row['RMSE']:>10.2f}{star_rf}")
    print(f"\n   ✅ Model terpilih : {best}")
    print(f"   ✅ MAE            : {best_row['MAE']} item/hari (error rata-rata)")
    print(f"   ✅ RMSE           : {best_row['RMSE']} item/hari")
    print(f"   ✅ Error relatif  : {mae_pct:.1f}% dari rata-rata penjualan")
```

---

---

# PHASE 6 — DEPLOYMENT: PREDIKSI STOK

```python
# Cell 6.1 — Fungsi prediksi per-item dengan multi-horizon

def predict_item_horizon(daily_raw, horizon=1, n_estimators=100, buffer_pct=15):
    """
    Prediksi stok untuk satu produk dengan horizon hari ke depan.

    Alur:
    1. Jika data >= 10 hari → gunakan ML (LR + RF)
    2. Jika data < 10 hari  → gunakan rata-rata 7 hari terakhir
    3. Prediksi multi-hari dilakukan secara rekursif (output hari ini = input hari berikutnya)
    4. Hasil RF di-cap di 2× nilai maksimum 30 hari terakhir (safety cap)
    5. Stok anjuran = prediksi × (1 + buffer_pct/100)

    Args:
        daily_raw    : DataFrame dengan kolom 'tanggal' & 'jumlah_terjual'
        horizon      : jumlah hari ke depan (1=besok, 7=minggu, 30=bulan)
        n_estimators : jumlah pohon Random Forest
        buffer_pct   : persentase buffer safety stock di atas prediksi

    Returns:
        dict berisi prediksi, stok anjuran, dan metadata
    """
    import pandas as pd
    import numpy as np

    daily = daily_raw.copy().sort_values('tanggal').reset_index(drop=True)
    n     = len(daily)
    vals  = daily['jumlah_terjual'].astype(float).values.tolist()
    last_date = pd.to_datetime(daily['tanggal'].iloc[-1])

    terjual_terakhir = int(vals[-1])
    rata_7hari       = round(float(np.mean(vals[-7:])) if n >= 7 else float(np.mean(vals)), 1)
    cap_max          = round(float(np.max(vals[-30:])) * 2) if n >= 5 else round(float(np.max(vals)) * 2)

    rf_model = lr_model = None
    metode   = 'Rata-rata'

    if n >= 10:
        try:
            feat_df = make_features(daily.copy())
            if len(feat_df) >= 5:
                X = feat_df[FEATURES_ITEM]
                y = feat_df['jumlah_terjual']
                lr_model = LinearRegression();                                       lr_model.fit(X, y)
                rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                rf_model.fit(X, y)
                metode = 'ML'
        except Exception as e:
            print(f'  ⚠️  Training error ({e}) → fallback ke rata-rata')

    def _predict_one(cur_vals, pred_date):
        n_c = len(cur_vals)
        if rf_model:
            feat = pd.DataFrame([{
                'hari_dalam_minggu': pred_date.weekday(),
                'hari_dalam_bulan':  pred_date.day,
                'minggu_ke':         int(pred_date.isocalendar()[1]),
                'lag_1':             float(cur_vals[-1]),
                'lag_2':             float(cur_vals[-2]) if n_c >= 2 else float(cur_vals[-1]),
                'lag_3':             float(cur_vals[-3]) if n_c >= 3 else float(cur_vals[-1]),
                'lag_7':             float(cur_vals[-7]) if n_c >= 7 else float(np.mean(cur_vals)),
                'rata_rata_3hari':   float(np.mean(cur_vals[-3:])) if n_c >= 3 else float(np.mean(cur_vals)),
                'rata_rata_7hari':   float(np.mean(cur_vals[-7:])) if n_c >= 7 else float(np.mean(cur_vals)),
            }])
            p_rf = max(0, min(round(float(rf_model.predict(feat)[0])), cap_max))
            p_lr = max(0, min(round(float(lr_model.predict(feat)[0])), cap_max))
        else:
            avg  = round(float(np.mean(cur_vals[-7:])) if len(cur_vals) >= 7 else float(np.mean(cur_vals)))
            p_rf = p_lr = avg
        return p_rf, p_lr

    cur_vals = vals.copy()
    preds_rf = []
    preds_lr = []

    for step in range(1, horizon + 1):
        pred_date = last_date + pd.Timedelta(days=step)
        p_rf, p_lr = _predict_one(cur_vals, pred_date)
        preds_rf.append(p_rf)
        preds_lr.append(p_lr)
        cur_vals.append(float(p_rf))

    total_rf    = sum(preds_rf)
    total_lr    = sum(preds_lr)
    prediksi    = preds_rf[0] if horizon == 1 else total_rf
    stok_anjuran = round(prediksi * (1 + buffer_pct / 100))

    return {
        'terjual_terakhir': terjual_terakhir,
        'rata_7hari':       rata_7hari,
        'pred_lr':          preds_lr[0] if horizon == 1 else total_lr,
        'pred_rf':          preds_rf[0] if horizon == 1 else total_rf,
        'prediksi':         prediksi,
        'avg_per_hari':     round(total_rf / horizon, 1),
        'stok_anjuran':     stok_anjuran,
        'metode':           metode,
        'last_date':        str(last_date.date()),
        'next_date':        str((last_date + pd.Timedelta(days=1)).date()),
        'end_date':         str((last_date + pd.Timedelta(days=horizon)).date()),
        'total_data':       n,
        'horizon':          horizon,
        'preds_harian_rf':  preds_rf,
    }

print("✅ Fungsi predict_item_horizon siap digunakan")
```

```python
# Cell 6.2 — Siapkan data per produk dari DataFrame

def build_daily_per_produk(df_clean, nama_produk):
    """Agregasi data harian untuk satu produk."""
    sub = df_clean[df_clean['nama_produk'] == nama_produk].copy()
    if sub.empty:
        return None, None

    daily = (sub.groupby('tanggal')
               .agg(jumlah_terjual=('jumlah', 'sum'))
               .reset_index()
               .sort_values('tanggal')
               .reset_index(drop=True))
    daily['tanggal'] = pd.to_datetime(daily['tanggal'])

    kat = sub['kategori_clean'].mode()[0]
    return daily, kat

# Uji dengan produk pertama
contoh_produk = 'Nasi'
daily_test, kat_test = build_daily_per_produk(df, contoh_produk)
if daily_test is not None:
    print(f"✅ Data produk '{contoh_produk}':")
    print(f"   Total hari : {len(daily_test)}")
    print(f"   Kategori   : {kat_test}")
    print(f"   Min/Max    : {daily_test['jumlah_terjual'].min()} / {daily_test['jumlah_terjual'].max()}")
    print(f"   Rata-rata  : {daily_test['jumlah_terjual'].mean():.1f} item/hari")
```

```python
# Cell 6.3 — Prediksi stok untuk semua produk (horizon: besok)

print("⏳ Menghitung prediksi stok untuk semua produk...\n")

BUFFER_PCT   = 15    # safety buffer 15%
HORIZON      = 1     # 1=besok, 7=minggu, 30=bulan
N_EST_ITEM   = 100   # jumlah pohon RF untuk prediksi per item

semua_produk  = df['nama_produk'].dropna().unique()
hasil_stok    = []

for produk in semua_produk:
    daily_p, kat_p = build_daily_per_produk(df, produk)
    if daily_p is None or len(daily_p) < 3:
        continue

    result = predict_item_horizon(
        daily_p, horizon=HORIZON,
        n_estimators=N_EST_ITEM, buffer_pct=BUFFER_PCT
    )
    result['produk']   = produk
    result['kategori'] = kat_p
    hasil_stok.append(result)

df_stok = pd.DataFrame(hasil_stok).sort_values('stok_anjuran', ascending=False).reset_index(drop=True)

print(f"✅ Prediksi selesai untuk {len(df_stok)} produk")
print(f"\n📋 Ringkasan prediksi stok besok (top 10 tertinggi):")
cols_show = ['produk', 'kategori', 'terjual_terakhir', 'rata_7hari',
             'pred_rf', 'stok_anjuran', 'metode']
print(df_stok[cols_show].head(10).to_string(index=False))
```

```python
# Cell 6.4 — Visualisasi prediksi stok besok per produk

df_mak_stok = df_stok[df_stok['kategori'] == 'Makanan'].head(20)
df_min_stok = df_stok[df_stok['kategori'] == 'Minuman']

fig, axes = plt.subplots(2, 1, figsize=(16, 12))
fig.suptitle(f'Rekomendasi Stok untuk Besok ({df_stok["next_date"].iloc[0]}) — Buffer {BUFFER_PCT}%',
             fontsize=13, fontweight='bold')

# Makanan
ax1 = axes[0]
x1  = range(len(df_mak_stok))
ax1.bar(x1, df_mak_stok['pred_rf'],     label='Prediksi RF',       color='#E74C3C', alpha=0.7)
ax1.bar(x1, df_mak_stok['stok_anjuran'] - df_mak_stok['pred_rf'],
        bottom=df_mak_stok['pred_rf'],  label=f'Buffer +{BUFFER_PCT}%', color='#F1948A', alpha=0.6)
ax1.set_title('Makanan (Top 20 tertinggi)')
ax1.set_xticks(x1)
ax1.set_xticklabels(df_mak_stok['produk'], rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('Jumlah Item')
ax1.legend()

# Minuman
ax2 = axes[1]
x2  = range(len(df_min_stok))
ax2.bar(x2, df_min_stok['pred_rf'],     label='Prediksi RF',       color='#3498DB', alpha=0.7)
ax2.bar(x2, df_min_stok['stok_anjuran'] - df_min_stok['pred_rf'],
        bottom=df_min_stok['pred_rf'],  label=f'Buffer +{BUFFER_PCT}%', color='#85C1E9', alpha=0.6)
ax2.set_title('Minuman')
ax2.set_xticks(x2)
ax2.set_xticklabels(df_min_stok['produk'], rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Jumlah Item')
ax2.legend()

plt.tight_layout()
plt.savefig('stok_besok.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Grafik stok besok disimpan sebagai 'stok_besok.png'")
```

```python
# Cell 6.5 — Prediksi multi-horizon: besok, 1 minggu, 1 bulan

print("⏳ Menghitung prediksi multi-horizon (1 hari / 7 hari / 30 hari)...\n")

horizons = {1: 'Besok', 7: '1 Minggu', 30: '1 Bulan'}
stok_multi = {h: [] for h in horizons}

for produk in semua_produk:
    daily_p, kat_p = build_daily_per_produk(df, produk)
    if daily_p is None or len(daily_p) < 3:
        continue

    for h in horizons:
        result = predict_item_horizon(
            daily_p, horizon=h,
            n_estimators=N_EST_ITEM, buffer_pct=BUFFER_PCT
        )
        result['produk']   = produk
        result['kategori'] = kat_p
        stok_multi[h].append(result)

# Pilih beberapa produk utama untuk perbandingan
produk_utama = ['Nasi', 'Ayam goreng serayu', 'Ayam goreng serayu - Paha',
                'Ayam goreng serayu - Dada', 'Teh manis dingin', 'Teh manis panas']

print("📊 PREDIKSI STOK MULTI-HORIZON:")
print(f"{'Produk':35s} {'Besok':>8s} {'1 Minggu':>10s} {'1 Bulan':>10s}")
print("-" * 70)

for produk in produk_utama:
    row_data = [produk]
    for h in [1, 7, 30]:
        df_h = pd.DataFrame(stok_multi[h])
        match = df_h[df_h['produk'] == produk]
        if not match.empty:
            row_data.append(match.iloc[0]['stok_anjuran'])
        else:
            row_data.append('-')
    print(f"{row_data[0]:35s} {str(row_data[1]):>8s} {str(row_data[2]):>10s} {str(row_data[3]):>10s}")

print(f"\n   (Sudah termasuk buffer safety stock +{BUFFER_PCT}%)")
```

```python
# Cell 6.6 — Visualisasi perbandingan multi-horizon untuk produk utama

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'Prediksi Stok Multi-Horizon per Produk (Buffer +{BUFFER_PCT}%)',
             fontsize=13, fontweight='bold')

produk_plot = produk_utama[:6]

for ax, produk in zip(axes.flat, produk_plot):
    stok_vals = []
    pred_vals = []
    xticks    = []

    for h, label in horizons.items():
        df_h  = pd.DataFrame(stok_multi[h])
        match = df_h[df_h['produk'] == produk]
        if not match.empty:
            stok_vals.append(match.iloc[0]['stok_anjuran'])
            pred_vals.append(match.iloc[0]['prediksi'])
            xticks.append(label)

    x = range(len(xticks))
    ax.bar(x, pred_vals,  label='Prediksi', color='#3498DB', alpha=0.7, width=0.4)
    ax.bar([i + 0.4 for i in x], stok_vals, label='Stok Anjuran',
           color='#E74C3C', alpha=0.7, width=0.4)
    ax.set_title(produk[:30], fontsize=9, fontweight='bold')
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(xticks, fontsize=9)
    ax.set_ylabel('Jumlah Item')
    ax.legend(fontsize=8)

    for i, (p, s) in enumerate(zip(pred_vals, stok_vals)):
        ax.text(i,       p + 0.5, str(p), ha='center', fontsize=8)
        ax.text(i + 0.4, s + 0.5, str(s), ha='center', fontsize=8, color='#C0392B')

plt.tight_layout()
plt.savefig('multi_horizon.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Grafik multi-horizon disimpan sebagai 'multi_horizon.png'")
```

---

---

# RINGKASAN & EXPORT

```python
# Cell 7.1 — Ringkasan akhir CRISP-DM

print("=" * 70)
print("   RINGKASAN AKHIR — CRISP-DM AYAM SERAYU BEKASI")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 — BUSINESS UNDERSTANDING                               │
│  Tujuan: Prediksi penjualan harian untuk manajemen stok         │
│  Target: Kurangi pemborosan & cegah kehabisan bahan baku        │
└─────────────────────────────────────────────────────────────────┘
""")

print(f"""┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2 — DATA UNDERSTANDING                                   │
│  Sumber    : CSV POS Pawoon                                     │
│  Baris     : {len(df):,} transaksi                                      │
│  Durasi    : {durasi} hari ({df['tanggal'].min()} → {df['tanggal'].max()})    │
│  Produk    : {df['nama_produk'].nunique()} item unik (29 Makanan + 10 Minuman)       │
└─────────────────────────────────────────────────────────────────┘
""")

print("""┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3 — DATA PREPARATION                                     │
│  Cleaning  : Parse tanggal, bersihkan angka (quotes/comma)      │
│  Agregasi  : Total harian per kategori & per produk             │
│  Features  : 11 fitur (temporal + lag + rolling avg)            │
│  Split     : Chronological 80/20 (bukan random)                 │
└─────────────────────────────────────────────────────────────────┘
""")

# Ambil model terbaik
best_mak = df_eval[df_eval['kategori']=='Makanan'].loc[df_eval[df_eval['kategori']=='Makanan']['RMSE'].idxmin()]
best_min = df_eval[df_eval['kategori']=='Minuman'].loc[df_eval[df_eval['kategori']=='Minuman']['RMSE'].idxmin()]

print(f"""┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4 — MODELING                                             │
│  Model 1   : Linear Regression (parametrik, interpretabel)      │
│  Model 2   : Random Forest (100 pohon, robust, non-parametrik)  │
│  Fitur     : 11 (kategori) / 9 (per-item, tanpa trend global)   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  PHASE 5 — EVALUATION                                           │
│  Makanan — Model terpilih  : {best_mak['model']:30s}           │
│             MAE             : {best_mak['MAE']} item/hari                     │
│             RMSE            : {best_mak['RMSE']} item/hari                    │
│  Minuman — Model terpilih  : {best_min['model']:30s}           │
│             MAE             : {best_min['MAE']} item/hari                      │
│             RMSE            : {best_min['RMSE']} item/hari                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  PHASE 6 — DEPLOYMENT                                           │
│  Prediksi per produk       : {len(df_stok)} item                            │
│  Horizon tersedia          : 1 hari / 7 hari / 30 hari          │
│  Safety buffer             : +{BUFFER_PCT}% di atas prediksi                │
│  Fallback                  : Rata-rata jika data < 10 hari       │
└─────────────────────────────────────────────────────────────────┘
""")
```

```python
# Cell 7.2 — Export hasil ke Excel

import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment

def style_header(ws, header_color='2D6A2D'):
    hfill = PatternFill(start_color=header_color, end_color=header_color, fill_type='solid')
    hfont = Font(color='FFFFFF', bold=True)
    for cell in ws[1]:
        cell.fill = hfill
        cell.font = hfont
        cell.alignment = Alignment(horizontal='center')

wb = openpyxl.Workbook()

# Sheet 1: Data Harian Makanan
ws1 = wb.active
ws1.title = 'Data Harian Makanan'
ws1.append(['Tanggal', 'Jumlah Terjual'])
style_header(ws1)
for _, row in daily_makanan.iterrows():
    ws1.append([str(row['tanggal']), int(row['jumlah_terjual'])])

# Sheet 2: Data Harian Minuman
ws2 = wb.create_sheet('Data Harian Minuman')
ws2.append(['Tanggal', 'Jumlah Terjual'])
style_header(ws2)
for _, row in daily_minuman.iterrows():
    ws2.append([str(row['tanggal']), int(row['jumlah_terjual'])])

# Sheet 3: Evaluasi Model
ws3 = wb.create_sheet('Evaluasi Model')
ws3.append(['Kategori', 'Model', 'MAE', 'RMSE', 'Model Terpilih'])
style_header(ws3)
for _, row in df_eval.iterrows():
    best_for_kat = df_eval[df_eval['kategori']==row['kategori']].loc[
        df_eval[df_eval['kategori']==row['kategori']]['RMSE'].idxmin(), 'model']
    ws3.append([
        row['kategori'], row['model'], row['MAE'], row['RMSE'],
        'Ya' if row['model'] == best_for_kat else ''
    ])

# Sheet 4: Prediksi Makanan (test set)
ws4 = wb.create_sheet('Prediksi Makanan')
ws4.append(['Tanggal', 'Aktual', 'Prediksi LR', 'Prediksi RF'])
style_header(ws4)
for tgl, act, lr, rf in zip(test_mak['tanggal'], y_test_mak,
                              pred_lr_mak, pred_rf_mak):
    ws4.append([str(tgl.date()), int(act), round(lr), round(rf)])

# Sheet 5: Prediksi Minuman (test set)
ws5 = wb.create_sheet('Prediksi Minuman')
ws5.append(['Tanggal', 'Aktual', 'Prediksi LR', 'Prediksi RF'])
style_header(ws5)
for tgl, act, lr, rf in zip(test_min['tanggal'], y_test_min,
                              pred_lr_min, pred_rf_min):
    ws5.append([str(tgl.date()), int(act), round(lr), round(rf)])

# Sheet 6: Stok Rekomendasi Besok
ws6 = wb.create_sheet('Stok Besok')
ws6.append(['No', 'Produk', 'Kategori', 'Terjual Terakhir',
            'Rata-rata 7 Hari', 'Prediksi LR', 'Prediksi RF',
            f'Stok Anjuran (+{BUFFER_PCT}%)', 'Metode', 'Data Tersedia'])
style_header(ws6)
for i, row in df_stok.iterrows():
    ws6.append([
        i+1, row['produk'], row['kategori'],
        row['terjual_terakhir'], row['rata_7hari'],
        row['pred_lr'], row['pred_rf'],
        row['stok_anjuran'], row['metode'], row['total_data']
    ])

# Simpan
EXCEL_FILE = 'Hasil_CRISDM_Ayam_Serayu.xlsx'
wb.save(EXCEL_FILE)
print(f"✅ File Excel berhasil disimpan: {EXCEL_FILE}")
print(f"   Sheet yang tersedia:")
for ws in wb.worksheets:
    print(f"   - {ws.title}")
```

```python
# Cell 7.3 — Download semua file dari Colab

from google.colab import files

print("📥 Mengunduh file hasil analisis...\n")

file_download = [
    EXCEL_FILE,
    'eksplorasi_data.png',
    'distribusi_jam.png',
    'distribusi_fitur.png',
    'korelasi_fitur.png',
    'feature_importance.png',
    'koefisien_lr.png',
    'aktual_vs_prediksi.png',
    'analisis_residual.png',
    'scatter_aktual_prediksi.png',
    'perbandingan_metrik.png',
    'stok_besok.png',
    'multi_horizon.png',
]

for f in file_download:
    if os.path.exists(f):
        files.download(f)
        print(f"   ✅ {f}")
    else:
        print(f"   ⚠️  {f} tidak ditemukan")

print("\n🎉 Selesai! Semua file berhasil diunduh.")
```

---

---

# LAMPIRAN — REFERENSI CEPAT

## Penjelasan Fitur

| Fitur | Deskripsi | Tipe |
|---|---|---|
| `hari_ke` | Urutan hari ke-N sejak awal data (tren jangka panjang) | Temporal |
| `hari_dalam_minggu` | 0=Senin, 6=Minggu (pola mingguan) | Temporal |
| `hari_dalam_bulan` | 1–31 (pola bulanan) | Temporal |
| `bulan` | Nomor bulan 1–12 (pola musiman) | Temporal |
| `minggu_ke` | Minggu ISO dalam setahun (1–53) | Temporal |
| `lag_1` | Penjualan 1 hari sebelumnya | Lag |
| `lag_2` | Penjualan 2 hari sebelumnya | Lag |
| `lag_3` | Penjualan 3 hari sebelumnya | Lag |
| `lag_7` | Penjualan 7 hari sebelumnya (pola mingguan) | Lag |
| `rata_rata_3hari` | Rata-rata bergerak 3 hari (shift 1 untuk anti-leakage) | Rolling |
| `rata_rata_7hari` | Rata-rata bergerak 7 hari (shift 1 untuk anti-leakage) | Rolling |

## Mengapa Chronological Split, Bukan Random?

- **Random split** bisa memasukkan data hari Jumat pekan depan ke training, lalu memprediksi hari Kamis pekan ini → model "bocor" melihat masa depan
- **Chronological split** mensimulasikan kondisi nyata: model hanya tahu data masa lalu saat melakukan prediksi
- Untuk time series / forecasting, **chronological split adalah standar industri**

## Mengapa Random Forest Unggul?

- **Linear Regression** mengekstrapolasi tren secara linear tak terbatas → bahaya dengan data pendek (2 bulan)
- **Random Forest** tidak bisa ekstrapolasi di luar rentang training → prediksi lebih konservatif dan aman
- RF lebih robust terhadap outlier (hari libur, event khusus)
- Mampu menangkap interaksi non-linear antar fitur

## Mengapa FEATURES_ITEM (9 fitur) Berbeda dari FEATURES (11 fitur)?

Untuk prediksi per-item, fitur `hari_ke` dan `bulan` dihapus karena:
- Dengan hanya 2 bulan data, Linear Regression cenderung over-fit tren jangka panjang
- Contoh: LR bisa prediksi 1.692 porsi nasi untuk 30 hari padahal aktual ~151/hari
- Tanpa fitur tren global, model fokus pada pola hari-dalam-minggu + autoregresif

## Formula Stok Anjuran

```
stok_anjuran = prediksi × (1 + buffer_pct / 100)

Contoh (buffer 15%):
  prediksi    = 100 item
  stok_anjuran = 100 × 1.15 = 115 item
```

## Safety Cap pada Random Forest

```python
cap_max = max(vals[-30:]) × 2
prediksi_final = min(prediksi_rf, cap_max)
```

Mencegah prediksi RF yang ekstrem ketika pola berubah drastis.

---

## Rumus Hitungan Manual (untuk verifikasi hasil sistem)

Bagian ini menyajikan seluruh rumus yang dipakai sistem beserta contoh perhitungan angka yang **bisa langsung dicocokan** dengan output di halaman web atau Colab.

---

### 1. MAE — Mean Absolute Error

**Rumus:**
```
MAE = (1/n) × Σ |Aktual_i − Prediksi_i|
```

**Langkah manual:**

| Hari | Aktual | Prediksi | |Aktual − Prediksi| |
|------|--------|----------|--------------------|
| 1    | 120    | 115      | 5                  |
| 2    | 95     | 102      | 7                  |
| 3    | 130    | 125      | 5                  |
| 4    | 110    | 118      | 8                  |
| 5    | 105    | 100      | 5                  |

```
MAE = (5 + 7 + 5 + 8 + 5) / 5
    = 30 / 5
    = 6.00
```

> Artinya: rata-rata kesalahan prediksi = **6 item per hari**

---

### 2. RMSE — Root Mean Squared Error

**Rumus:**
```
RMSE = √[ (1/n) × Σ (Aktual_i − Prediksi_i)² ]
```

**Langkah manual** (lanjut data di atas):

| Hari | |Aktual − Prediksi| | Error² |
|------|--------------------|----|
| 1    | 5                  | 25  |
| 2    | 7                  | 49  |
| 3    | 5                  | 25  |
| 4    | 8                  | 64  |
| 5    | 5                  | 25  |

```
MSE  = (25 + 49 + 25 + 64 + 25) / 5
     = 188 / 5
     = 37.6

RMSE = √37.6
     = 6.13
```

> RMSE lebih besar dari MAE → ada beberapa error yang besar (dihukum lebih berat).
> Sistem memilih model dengan **RMSE terkecil** sebagai model terpilih.

---

### 3. Linear Regression — Rumus Prediksi

**Rumus umum:**
```
Ŷ = β₀ + β₁X₁ + β₂X₂ + ... + β₁₁X₁₁
```

Dimana setiap Xᵢ adalah salah satu dari 11 fitur:

| Koefisien | Fitur              | Contoh nilai |
|-----------|--------------------|--------------|
| β₀        | Intercept          | 42.5         |
| β₁        | hari_ke            | 0.3          |
| β₂        | hari_dalam_minggu  | -2.1         |
| β₃        | hari_dalam_bulan   | 0.5          |
| β₄        | bulan              | 1.2          |
| β₅        | minggu_ke          | -0.4         |
| β₆        | lag_1              | 0.35         |
| β₇        | lag_2              | 0.15         |
| β₈        | lag_3              | 0.10         |
| β₉        | lag_7              | 0.20         |
| β₁₀       | rata_rata_3hari    | 0.25         |
| β₁₁       | rata_rata_7hari    | 0.18         |

**Contoh satu prediksi:**
```
Input hari besok:
  hari_ke=45, hari_dalam_minggu=2 (Rabu), hari_dalam_bulan=15,
  bulan=1, minggu_ke=3,
  lag_1=110, lag_2=95, lag_3=120,
  lag_7=105, rata_rata_3hari=108.3, rata_rata_7hari=107.1

Ŷ = 42.5
  + (0.3 × 45)    = 13.5
  + (-2.1 × 2)    = -4.2
  + (0.5 × 15)    = 7.5
  + (1.2 × 1)     = 1.2
  + (-0.4 × 3)    = -1.2
  + (0.35 × 110)  = 38.5
  + (0.15 × 95)   = 14.25
  + (0.10 × 120)  = 12.0
  + (0.20 × 105)  = 21.0
  + (0.25 × 108.3)= 27.075
  + (0.18 × 107.1)= 19.278
────────────────────────────
Ŷ ≈ 191.4 → dibulatkan = 191 item
```

> Nilai β (koefisien) aktual dari data Anda dapat dilihat di **Cell 4.4** output Colab.

---

### 4. Rolling Average (Rata-rata Bergerak)

**Rumus rata_rata_3hari** (digunakan sebagai fitur, bukan prediksi akhir):
```
rata_rata_3hari[t] = (jumlah[t-2] + jumlah[t-1] + jumlah[t]) / 3
                     (lalu di-shift 1 hari agar tidak bocor ke masa depan)
```

**Contoh:**
```
Data penjualan:  [100, 110, 95, 120, 105, ...]

rata_rata_3hari untuk hari ke-4 (nilai 120):
  = (100 + 110 + 95) / 3
  = 305 / 3
  = 101.67  ← digunakan sebagai fitur, bukan hari ke-4 itu sendiri
```

**Rumus rata_rata_7hari** (sama, window 7 hari):
```
rata_rata_7hari[t] = mean(jumlah[t-7 .. t-1])
```

---

### 5. Lag Feature

**Rumus:**
```
lag_1[t] = jumlah[t-1]   ← penjualan kemarin
lag_2[t] = jumlah[t-2]   ← penjualan 2 hari lalu
lag_3[t] = jumlah[t-3]
lag_7[t] = jumlah[t-7]   ← penjualan hari yang sama minggu lalu
```

**Contoh:**
```
Data: [90, 100, 110, 95, 120, 105, 115, 130]
                                              ↑ hari ini (indeks 7)

lag_1 = 115   (indeks 6)
lag_2 = 105   (indeks 5)
lag_3 = 120   (indeks 4)
lag_7 = 90    (indeks 0)
```

---

### 6. Stok Anjuran (Safety Stock)

**Rumus:**
```
stok_anjuran = prediksi × (1 + buffer_pct / 100)
```

**Contoh dengan buffer 15%:**
```
prediksi     = 100 item
stok_anjuran = 100 × (1 + 15/100)
             = 100 × 1.15
             = 115 item
```

**Tabel buffer berbeda:**

| Prediksi | Buffer 10% | Buffer 15% | Buffer 20% |
|----------|------------|------------|------------|
| 50       | 55         | 58         | 60         |
| 100      | 110        | 115        | 120        |
| 150      | 165        | 173        | 180        |
| 200      | 220        | 230        | 240        |

---

### 7. Safety Cap Random Forest

**Rumus:**
```
cap_max      = max(penjualan 30 hari terakhir) × 2
prediksi_cap = min(prediksi_rf, cap_max)
```

**Contoh:**
```
Penjualan 30 hari terakhir: maks = 145 item
cap_max = 145 × 2 = 290

Jika RF prediksi 350 (ekstrem) → dipotong jadi 290
Jika RF prediksi 120 (normal)  → tetap 120
```

---

### 8. Prediksi Fallback (Rata-rata, jika data < 10 hari)

**Rumus:**
```
Jika data < 10 hari → prediksi = mean(penjualan 7 hari terakhir)
                                  (atau mean semua data jika < 7 hari)
```

**Contoh produk baru dengan 5 hari data:**
```
Data: [12, 15, 10, 14, 13]
prediksi = (12 + 15 + 10 + 14 + 13) / 5 = 12.8 → dibulatkan = 13
stok_anjuran = 13 × 1.15 = 14.95 → dibulatkan = 15
```

---

### 9. Cell Verifikasi Manual di Colab

Salin cell berikut ke Colab untuk memverifikasi hasil sistem secara manual:

```python
# Cell Verifikasi — Hitung MAE & RMSE manual dan cocokkan dengan Cell 5.1

import numpy as np

# Ganti nilai ini dengan data aktual dari output Cell 5.2 (test set Makanan)
y_aktual  = list(y_test_mak.values)
y_pred_lr = list(pred_lr_mak)
y_pred_rf = list(pred_rf_mak)

def hitung_mae(aktual, prediksi):
    n = len(aktual)
    total_error = sum(abs(a - p) for a, p in zip(aktual, prediksi))
    return round(total_error / n, 2)

def hitung_rmse(aktual, prediksi):
    n = len(aktual)
    total_sq = sum((a - p) ** 2 for a, p in zip(aktual, prediksi))
    return round((total_sq / n) ** 0.5, 2)

mae_lr_manual  = hitung_mae(y_aktual, y_pred_lr)
rmse_lr_manual = hitung_rmse(y_aktual, y_pred_lr)
mae_rf_manual  = hitung_mae(y_aktual, y_pred_rf)
rmse_rf_manual = hitung_rmse(y_aktual, y_pred_rf)

print("=" * 55)
print("  VERIFIKASI MANUAL vs SISTEM (Makanan)")
print("=" * 55)
print(f"\n  {'':25s} {'MAE':>8s} {'RMSE':>8s}")
print(f"  {'-'*45}")
print(f"  {'Linear Regression (manual)':25s} {mae_lr_manual:>8.2f} {rmse_lr_manual:>8.2f}")
print(f"  {'Linear Regression (sistem)':25s} {df_eval.loc[0,'MAE']:>8.2f} {df_eval.loc[0,'RMSE']:>8.2f}")
print(f"  {'Random Forest (manual)':25s} {mae_rf_manual:>8.2f} {rmse_rf_manual:>8.2f}")
print(f"  {'Random Forest (sistem)':25s} {df_eval.loc[1,'MAE']:>8.2f} {df_eval.loc[1,'RMSE']:>8.2f}")
print(f"\n  ✅ Nilai harus identik jika perhitungan benar")

# Verifikasi stok anjuran satu produk secara manual
print("\n" + "=" * 55)
print("  VERIFIKASI STOK ANJURAN (contoh: produk 'Nasi')")
print("=" * 55)

BUFFER = 15
produk_cek = 'Nasi'
row = df_stok[df_stok['produk'] == produk_cek].iloc[0]

prediksi_sistem   = row['prediksi']
stok_sistem       = row['stok_anjuran']
stok_manual       = round(prediksi_sistem * (1 + BUFFER / 100))

print(f"\n  Prediksi sistem  : {prediksi_sistem}")
print(f"  Stok sistem      : {stok_sistem}")
print(f"  Stok manual      : {prediksi_sistem} × {1 + BUFFER/100} = {stok_manual}")
print(f"  Cocok?           : {'✅ YA' if stok_manual == stok_sistem else '❌ TIDAK — cek buffer_pct'}")
```
