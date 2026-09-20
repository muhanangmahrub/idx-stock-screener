# CLAUDE.md

Panduan untuk Claude Code saat mengerjakan proyek ini. Baca ini sebelum menulis atau mengubah kode.

## Tentang proyek

**IDX Stock Screener** — screener saham Bursa Efek Indonesia (IDX) yang menggabungkan
dua paradigma analisis:

- **Lapis 1 — Fundamental (value investing):** menyaring saham yang _layak_ dibeli
  berdasarkan rasio keuangan. Kerangka mengacu pada metode **Teguh Hidayat**.
- **Lapis 2 — Teknikal (chart pattern):** menentukan _timing_ masuk dengan mendeteksi
  pola chart klasik. Kerangka mengacu pada buku analisis teknikal **Edianto Ong**.

Alurnya: universe saham IDX → fundamental menyaring → watchlist saham layak →
teknikal mendeteksi pola & sinyal breakout → kandidat untuk verifikasi manual.

Ini **bukan** robot trading atau pemberi rekomendasi jual/beli. Output-nya adalah
kandidat untuk saya (pemilik) analisis manual. Setiap teks yang menghadap pengguna
harus memperjelas ini.

## Pemilik & konteks

- Pemilik adalah **QA Engineer**, bukan trader profesional. Terbiasa dengan pytest,
  test design, dan berpikir dalam kerangka spesifikasi → implementasi → verifikasi.
- Proyek ini untuk **belajar + portfolio**. Jadi kejelasan kode & bisa dijelaskan
  lebih penting daripada trik pintar. Kalau ada dua cara, pilih yang lebih mudah dibaca.
- Aturan analisis (angka PER yang dianggap murah, syarat konfirmasi pola, dsb.)
  **berasal dari buku yang sedang dibaca pemilik.** JANGAN mengarang angka atau aturan
  spesifik dari Teguh Hidayat / Edianto Ong. Kalau sebuah aturan belum diberikan,
  tanyakan — jangan menebak dan menuliskannya seolah dari buku.

## Tumpukan teknologi

- Python 3.11+
- `streamlit` — UI web
- `yfinance` — data harga & fundamental IDX (ticker format `XXXX.JK`)
- `scipy.signal.find_peaks` — deteksi swing high/low (fondasi deteksi pola)
- `plotly` — chart candlestick interaktif
- `pandas`, `numpy` — olah data
- `pytest` — unit test

Jangan menambah dependency baru tanpa alasan jelas. Kalau perlu, sebutkan kenapa dulu.

## Struktur yang dituju

```
idx-screener/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── app.py                    # entry Streamlit (UI saja, logika di modul)
├── screener/
│   ├── __init__.py
│   ├── data.py               # ambil data (harga & fundamental) via yfinance
│   ├── extrema.py            # get_extrema() — deteksi swing high/low
│   ├── trend.py              # classify_trend() — uptrend/downtrend/sideways dari urutan swing
│   ├── patterns.py           # detektor pola teknikal (double top, H&S, dst.)
│   ├── fundamental.py        # filter value investing (lapis 1)
│   └── plotting.py           # fungsi chart plotly
└── tests/
    ├── test_patterns.py      # unit test tiap detektor pola
    ├── test_trend.py         # unit test klasifikasi tren (data zigzag sintetis)
    └── test_fundamental.py   # unit test filter fundamental
```

Pemisahan logika dari UI itu wajib: `app.py` hanya merangkai UI dan memanggil modul
`screener/`. Ini yang membuat logika bisa di-unit-test tanpa menjalankan Streamlit —
poin penting untuk portfolio QA.

## Prinsip kerja (penting untuk vibe coding)

1. **Spesifikasi dulu, kode kemudian.** Untuk tiap pola atau filter baru, mulai dari
   aturan tertulis (dari buku pemilik) → terjemahkan ke kondisi kode → baru tulis test.
   Perlakukan aturan buku seperti _test spec_.

2. **Setiap detektor pola harus punya unit test.** Buat data sintetis yang jelas
   mengandung pola itu, pastikan detektor menemukannya; buat data yang jelas TIDAK
   mengandungnya, pastikan detektor tidak menghasilkan false positive. Ini cara pemilik
   memverifikasi, sejalan dengan latar QA-nya.

3. **Perubahan kecil & bertahap (surgical).** Jangan refactor besar-besaran tanpa
   diminta. Ubah seperlunya untuk tugas yang diminta. Kalau melihat perbaikan lain,
   sebutkan sebagai saran — jangan langsung kerjakan.

4. **Nyatakan asumsi.** Kalau sebuah aturan ambigu atau angkanya belum ada, tanya
   dulu. Jangan diam-diam memilih threshold lalu menuliskannya seolah final.

5. **Jelaskan sambil jalan.** Pemilik sedang belajar. Saat menulis logika non-trivial
   (regresi trendline, konfirmasi volume, rasio fundamental), sisipkan komentar singkat
   _kenapa_, bukan hanya _apa_. Hindari komentar yang cuma mengulang kode.

6. **Definisi selesai (definition of done):** kode jalan + ada unit test yang lulus +
   nama variabel jelas + teks pengguna menegaskan ini alat bantu, bukan rekomendasi.

## Gaya kode

- Nama deskriptif dalam bahasa Inggris untuk kode; boleh bahasa Indonesia untuk
  teks yang menghadap pengguna (UI) dan komentar penjelas.
- Fungsi kecil dengan satu tanggung jawab. Detektor pola mengembalikan dictionary
  dengan bentuk seragam: `{"type", "bias", "points", "neckline", "confirmed"}`.
- Untuk menambah pola baru: tulis fungsi `detect_xxx(prices, highs, lows, tol)`,
  lalu daftarkan di dictionary `DETECTORS`. Jangan mengubah alur `scan()`.
- Semua parameter sensitivitas (distance, prominence, toleransi) harus bisa diatur,
  bukan hardcoded — supaya bisa dikalibrasi terhadap contoh nyata di buku.

## Spesifikasi metode Teguh Hidayat (lapis 1)

Aturan berikut diekstrak dari materi milik pemilik (rekap + pseudo-code Investment
Planning Teguh Hidayat). Ini spesifikasi rujukan — implementasikan sebagai kode +
test saat diminta, jangan mengubah angkanya tanpa instruksi. Angka yang TIDAK ada
di sumber (mis. batas DER) ditandai sebagai asumsi netral yang wajib dikalibrasi,
bukan angka beliau.

### A. Screening awal (filter kuantitatif — sudah diimplementasi di fundamental.py)

- Likuiditas: nilai transaksi harian ≥ Rp5 miliar.
- Free float ≥ 15%.
- ROE disetahunkan ≥ 10% (ideal 20–30% tergantung sektor).
- Bukan saham bandar/gorengan → blacklist manual (bukan deteksi otomatis).

### B. Valuasi & harga best buy (jalur harga absolut — sudah di fundamental.py: compute_valuation)

Urutan hitung sesuai pseudo-code:

1. `pbv_base = roe_annualized / 10` (ROE 51% → PBV dasar 5,1x)
2. `fair_price_base = bvps * pbv_base`
3. Diskon risiko: `fair_price_adj = fair_price_base * (1 - 0.10 * jumlah_faktor_risiko)`
   (10% per faktor risiko makro/sektoral)
4. Margin of safety 35–50%: `best_buy = fair_price_adj * (1 - MoS)` (default MoS 0,35)
5. Toleransi beli maksimum: `max_buy = best_buy * 1.15` (+10–15% di atas best buy)

Catatan: kode saat ini (`valuation_label`) memakai jalur PBV (bandingkan PBV kini vs
PBV wajar). Jalur absolut di atas menambah BVPS, diskon risiko, dan toleransi +15%.
Keduanya konsisten (PBV = harga/BVPS). Saat implementasi, gabungkan — jangan buang
jalur PBV, tambahkan jalur harga absolut sebagai fungsi terpisah.

### C. Aturan valuasi murah (menggantikan aturan PBV tunggal lama)

- Blue chip: `PER ≤ 12` ATAU `PBV ≤ 0.7`.
- Small cap: `PER ≤ 8` ATAU `PBV ≤ 0.7`.

### D. Manajemen risiko posisi (sudah di fundamental.py: compute_position_limit)

- Batas beli per emiten ≤ 30% dari nilai transaksi harian emiten.
- Alokasi satu saham ≤ 20% dari total liquid net worth investor.

### E. Alokasi portofolio & cash (sudah di portfolio.py)

- Portofolio terfokus: target 5–6 saham (maks 10–12), sektor berbeda.
- Bobot per saham: min 5%, target 25–30%, maks 35% (ekstrem 50% keyakinan tinggi).
- Cash cadangan: min 10%, ideal 20%.

### F. Eksekusi & horizon (sudah di execution.py — sebagai status kondisi, bukan label BUY/SELL)

- Horizon: pendek <3 bln, menengah 3–12 bln, panjang 1–2 thn.
- BUY/average down bila `harga ≤ max_buy` DAN valuasi murah.
- CONSIDER_PROFIT_TAKING bila `harga ≥ fair_price`.
- HOLD selama fundamental bertumbuh.
- REBALANCE/EXIT bila holding ≥ 12 bln DAN growth tidak on-track.
- Evaluasi berkala tiap rilis laporan keuangan kuartalan.

### Batas otomasi (JANGAN dilanggar)

Kriteria kualitatif tetap manual, tidak boleh dipaksa jadi filter otomatis: deep
dive laporan keuangan, market leader/visibilitas merek, model bisnis simpel, GCG &
profil pemilik, kebiasaan right issue, katalis kebijakan/makro. Screener menampilkan
ini sebagai checklist untuk dijawab pemilik, bukan menilainya sendiri.

## Spesifikasi metode Edianto Ong (lapis 2)

Aturan berikut diberikan pemilik dari buku Edianto Ong. Sama seperti bagian
Teguh Hidayat: ini spesifikasi rujukan, jangan mengubah angkanya tanpa
instruksi. Aturan pola chart/candlestick (definisi, toleransi, konfirmasi)
BELUM diberikan — tanyakan saat akan mengimplementasikan detektor.

### A. Horizon trader

- Short term: < 3 minggu.
- Medium term: 3 minggu sampai beberapa bulan.
- Long term: > 1 tahun.

Pemilik cenderung **medium term**. Default UI dan detektor harus mengikuti
alur medium term di bawah, bukan short term.

### C. Definisi tren (sudah di trend.py: classify_trend)

- Uptrend: puncak maupun dasar yang terbentuk semakin lama semakin tinggi.
- Downtrend: puncak dan dasar yang terbentuk semakin lama semakin rendah.
- Sideways: puncak ke puncak dan dasar ke dasar (hampir) sama.
- Kombinasi lain (mis. puncak naik, dasar turun) dilaporkan "tidak jelas",
  tidak dipaksa ke salah satu.
- ASUMSI yang wajib dikalibrasi (bukan angka buku): toleransi "hampir sama"
  default 2%, jumlah puncak/dasar terakhir yang dibandingkan default 3.

### B. Alur timeframe untuk medium term (makro → presisi)

1. **Weekly chart, periode 3 tahun** — gambaran makro (tren besar).
2. **Daily chart, periode 1 tahun** — "dikompres" dari weekly untuk detail.
3. **Minutes chart** — hanya bila perlu presisi lebih, sebagai konfirmasi
   akhir (bukan titik awal analisis).

### D. Trendline (sudah di trend.py: select_anchor_points, build_trendline)

- Up-trendline: menghubungkan harga terendah (Low) lembah-lembah pada chart
  uptrend; level tempat uptrend diuji. Down-trendline: menghubungkan harga
  tertinggi (High) puncak-puncak pada chart downtrend.
- Syarat titik acuan "siap": pada uptrend, dasar A2 baru dipakai setelah
  level puncak terakhir sebelum A2 (resistance) dilewati harga. Sebagian
  technicalist cukup mensyaratkan 50% jarak vertikal A2 → resistance
  terlampaui. Downtrend cermin: puncak B2 siap setelah support (dasar
  terakhir sebelum B2) ditembus, atau 50% jaraknya.
- Pakai harga keseluruhan: High untuk puncak/resistance, Low untuk
  dasar/support — bukan Close.
- ASUMSI (bukan buku): default mazhab 100% (`DEFAULT_CONFIRMATION_RATIO`),
  bisa diganti 50% di UI; titik acuan pertama = titik awal tren, tidak perlu
  konfirmasi; bila 3+ titik tidak segaris dipakai regresi kuadrat terkecil.

### E. Penembusan trendline (sudah di trend.py: check_trendline_break)

- Aturan utama: penembusan sah (**valid break**) bila harga **penutupan**
  berada di luar garis — Close jauh lebih signifikan daripada pergerakan
  sementara intraday.
- Tembusan sementara oleh High/Low intraday yang Close-nya kembali ke dalam
  garis = **false break / whipsaw**, bukan penembusan.
- ASUMSI (bukan buku): diperiksa sejak bar setelah titik acuan terakhir
  (garis regresi bisa menyilang titik acuannya sendiri); Close tepat di
  garis belum dihitung di luar. Status hanya ditampilkan sebagai kondisi,
  bukan sinyal jual/beli.

Catatan implementasi (dari pengecekan yfinance, bukan dari buku): weekly &
daily tersedia sejak 2004; intraday dibatasi yfinance (1m: 7 hari, 5m–30m:
60 hari) dan bar 09:00 sering volume 0 (pre-opening) sehingga harus dibuang
sebelum deteksi pola. Pemetaan ke `get_price_history(ticker, period,
interval)`: weekly 3 thn = `("3y", "1wk")`, daily 1 thn = `("1y", "1d")`.

## Yang harus dihindari

- Jangan mengarang aturan/angka spesifik dari buku Teguh Hidayat atau Edianto Ong.
- Jangan mengubah screener jadi pemberi sinyal jual/beli otomatis.
- Jangan menambah dependency berat (TA-Lib, framework ML) tanpa dibahas dulu.
- Jangan menghapus atau melemahkan disclaimer "bukan rekomendasi" di UI.
- Jangan menaruh logika bisnis di `app.py`.
