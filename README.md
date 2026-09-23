# IDX Stock Screener

Screener saham Bursa Efek Indonesia (IDX) yang menggabungkan filter
fundamental (value investing, mengacu pada metode Teguh Hidayat) dengan
deteksi pola chart teknikal (mengacu pada Edianto Ong).

**Ini bukan robot trading atau pemberi rekomendasi jual/beli.** Output-nya
adalah kandidat saham untuk dianalisis manual.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Menjalankan aplikasi

```bash
streamlit run app.py
```

## Menjalankan test

```bash
pytest
```

## Status pengembangan

Lihat `CLAUDE.md` untuk spesifikasi lengkap dan status tiap bagian.

- [x] Filter fundamental awal (likuiditas, free float, ROE, blacklist)
- [x] Aturan valuasi murah (PER/PBV)
- [x] Valuasi harga absolut (BVPS, diskon risiko, margin of safety)
- [x] Manajemen risiko posisi (batas beli per emiten)
- [x] Alokasi portofolio & cash (tab Portofolio)
- [x] Eksekusi & horizon (bagian F) - sebagai status kondisi, bukan label BUY/SELL
- [x] Checklist kualitatif (dijawab manual, tidak dinilai otomatis)
- [x] Baca LK mentah (e-book Metode Analisis Fundamental): rasio ROE/ROA/PER/PBV + EDR/EER/EAR dari komponen LK, anualisasi per periode, tolak mutlak, warning kualitas laba & neraca
- [x] ROE dari laba TTM (bukan satu kuartal x4), peringatan saat ROE ekstrem membuat harga wajar tidak realistis
- [x] Harga nominal (bukan tersesuaikan dividen) untuk analisis teknikal; peringatan saat bar terakhir belum selesai
- [x] Rencana breakout tanpa lookahead: garis keluar hanya dari data sampai bar masuk
- [x] Level S/R punya jumlah sendiri, terpisah dari lookback tren, agar level lama ikut terlihat
- [x] Daftar blue chip manual (PER <= 12) vs small cap (PER <= 8) di screening massal
- [x] Isian pemilik tersimpan antar-sesi: blacklist, blue chip, checklist, portofolio (`user_data.json`)
- [x] Backtest walk-forward (`scripts/backtest_cli.py`) untuk mengukur dampak tiap perubahan aturan, dengan fundamental titik-waktu dari laporan tahunan
- [x] Perbandingan PER/PBV dengan median sektor (jalur opsional; default tetap jalur harga absolut/story)
- [x] Klasifikasi tren uptrend/downtrend/sideways dari urutan swing high/low (definisi Edianto Ong)
- [x] Up-/down-trendline dari Low dasar / High puncak, titik acuan harus 'siap' (resistance/support dilewati, atau 50% jaraknya)
- [x] Status penembusan trendline: valid break (Close di luar garis) vs false break/whipsaw (intraday saja), plus aturan 2nd day (Open sesi berikutnya sebagai konfirmasi akhir)
- [x] Level support/resistance horizontal: tembus sah via Close, peran berbalik setelah tembus, usia level ditampilkan; level dari Low lembah / High puncak sesuai buku
- [x] Pullback: uji ulang level yang sudah dilewati, dicatat bertahan (Close di dalam) atau gagal (tembus lagi)
- [x] Toleransi penembusan trendline per horizon (short 0,5-1,5%, medium 2-3%, long 3,5-5%; default 2%)
- [x] Channeling: channel line sejajar dari basic trendline; basic tertembus = awal perubahan tren, channel line tertembus = akselerasi
- [x] The Fan Principle (Bab 14): tiga trendline dari satu pangkal, reversal terkonfirmasi hanya saat garis ketiga tertembus, dua arah, peran garis berbalik
- [x] Validasi breakout & trading plan (Edianto Ong, contoh McD): breakout sah +1,5%, strong resistance (3x uji), masuk 2nd day, cut-loss 1,5% di bawah support baru, false breakout, keluar saat trendline patah
- [ ] Detektor pola chart teknikal (menunggu spesifikasi dari buku)
- [x] Screening massal universe IDX (corong: likuiditas IDX → blacklist → free float IDX → ROE yfinance → valuasi murah PER/PBV, batas small cap untuk semua)
- [x] UI Streamlit: Lapis 1, Portofolio, Checklist & Eksekusi tersambung; Lapis 2 baru swing high/low
- [x] Sumber data resmi IDX (free float & nilai transaksi dari idx.co.id, fallback estimasi yfinance)
