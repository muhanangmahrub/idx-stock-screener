"""Penyimpanan sederhana berbasis file JSON untuk isian yang perlu bertahan.

Screener ini dipakai berulang untuk memantau emiten yang sama, jadi isian
seperti blacklist, daftar blue chip, jawaban checklist kualitatif, faktor
risiko, dan posisi portofolio tidak masuk akal kalau hilang tiap refresh.

Disengaja sesederhana mungkin: satu berkas JSON di folder proyek, tanpa
database dan tanpa dependency baru. Isinya data pemilik sendiri, bukan cache
- kegagalan baca/tulis tidak boleh menjatuhkan aplikasi, jadi error dilaporkan
sebagai nilai default (baca) atau False (tulis).
"""

import json
from pathlib import Path
from typing import Any

DEFAULT_STORE_PATH = Path(__file__).resolve().parent.parent / "user_data.json"

KEY_BLACKLIST = "blacklist"
KEY_BLUE_CHIPS = "blue_chips"
KEY_CHECKLIST = "checklist"
KEY_PORTFOLIO = "portfolio"
KEY_RISK_FACTORS = "risk_factors"


def load_store(path: Path | str = DEFAULT_STORE_PATH) -> dict[str, Any]:
    """Seluruh isi penyimpanan; dict kosong bila belum ada atau rusak."""
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_store(data: dict[str, Any], path: Path | str = DEFAULT_STORE_PATH) -> bool:
    """Tulis seluruh isi penyimpanan; False bila gagal (mis. folder read-only)."""
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
    except (OSError, TypeError):
        return False
    return True


def load_value(key: str, default: Any = None, path: Path | str = DEFAULT_STORE_PATH) -> Any:
    return load_store(path).get(key, default)


def save_value(key: str, value: Any, path: Path | str = DEFAULT_STORE_PATH) -> bool:
    """Perbarui satu kunci tanpa menghapus kunci lain."""
    data = load_store(path)
    data[key] = value
    return save_store(data, path)


def load_codes(key: str, path: Path | str = DEFAULT_STORE_PATH) -> set[str]:
    """Kumpulan kode emiten (blacklist / blue chip) dalam huruf besar."""
    value = load_value(key, [], path)
    if isinstance(value, str):
        value = value.split(",")
    return {str(code).strip().upper() for code in value if str(code).strip()}


def save_codes(key: str, codes: set[str] | list[str], path: Path | str = DEFAULT_STORE_PATH) -> bool:
    return save_value(key, sorted({str(c).strip().upper() for c in codes if str(c).strip()}), path)


def codes_to_text(codes: set[str]) -> str:
    """Kembali ke bentuk yang enak diedit di satu baris input."""
    return ", ".join(sorted(codes))
