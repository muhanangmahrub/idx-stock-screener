"""Format angka untuk teks yang menghadap pengguna."""


def format_rupiah(value: float) -> str:
    """Rp1.234.567 - pemisah ribuan titik sesuai kebiasaan Indonesia."""
    return f"Rp{value:,.0f}".replace(",", ".")


_COMPACT_UNITS = ((1e12, "T"), (1e9, "M"), (1e6, "jt"))


def format_rupiah_compact(value: float) -> str:
    """Rp404,4 T / Rp478,9 M / Rp12,3 jt - untuk angka besar di ruang sempit.

    Singkatan mengikuti kebiasaan media keuangan Indonesia: T = triliun,
    M = miliar, jt = juta. Di bawah satu juta ditampilkan penuh.
    """
    for threshold, unit in _COMPACT_UNITS:
        if abs(value) >= threshold:
            return f"Rp{value / threshold:.1f} {unit}".replace(".", ",")
    return format_rupiah(value)
