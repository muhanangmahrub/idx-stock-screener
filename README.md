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
- [ ] Detektor pola chart teknikal (menunggu spesifikasi dari buku)
- [ ] Screening massal universe IDX (filter likuiditas dari Ringkasan Saham)
- [x] UI Streamlit: Lapis 1, Portofolio, Checklist & Eksekusi tersambung; Lapis 2 baru swing high/low
- [x] Sumber data resmi IDX (free float & nilai transaksi dari idx.co.id, fallback estimasi yfinance)
