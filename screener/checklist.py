"""Checklist kualitatif (CLAUDE.md "Batas otomasi", Teguh Hidayat).

Kriteria di sini SENGAJA tidak dinilai otomatis: screener hanya menampilkan
pertanyaannya dan merangkum jawaban pemilik. Tidak ada skor, tidak ada
lolos/gagal - hasilnya pengingat mana yang belum dicek.
"""

from dataclasses import dataclass

ANSWER_UNCHECKED = "Belum dicek"
ANSWER_YES = "Ya"
ANSWER_NO = "Tidak"
ANSWER_OPTIONS = (ANSWER_UNCHECKED, ANSWER_YES, ANSWER_NO)

QUALITATIVE_CHECKLIST: dict[str, str] = {
    "deep_dive": "Sudah deep dive laporan keuangan (bukan hanya rasio)?",
    "market_leader": "Market leader di industrinya / merek punya visibilitas?",
    "simple_business": "Model bisnis simpel dan bisa dijelaskan dalam satu kalimat?",
    "gcg": "GCG baik dan profil pemilik/manajemen bisa dipercaya?",
    "right_issue": "Tidak punya kebiasaan right issue yang merugikan pemegang saham?",
    "catalyst": "Ada katalis kebijakan/makro yang mendukung?",
}


@dataclass
class ChecklistSummary:
    yes: list[str]
    no: list[str]
    unchecked: list[str]

    @property
    def is_complete(self) -> bool:
        return not self.unchecked


def summarize_checklist(answers: dict[str, str]) -> ChecklistSummary:
    """Kelompokkan jawaban per item; item yang tidak dijawab dianggap belum dicek."""
    summary = ChecklistSummary(yes=[], no=[], unchecked=[])
    for item_key, question in QUALITATIVE_CHECKLIST.items():
        answer = answers.get(item_key, ANSWER_UNCHECKED)
        if answer == ANSWER_YES:
            summary.yes.append(question)
        elif answer == ANSWER_NO:
            summary.no.append(question)
        else:
            summary.unchecked.append(question)
    return summary
