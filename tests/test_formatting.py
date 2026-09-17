from screener.formatting import format_rupiah, format_rupiah_compact


def test_uses_dot_as_thousand_separator():
    assert format_rupiah(393_453_237_500) == "Rp393.453.237.500"


def test_rounds_to_whole_rupiah():
    assert format_rupiah(1234.6) == "Rp1.235"
    assert format_rupiah(0) == "Rp0"


class TestFormatRupiahCompact:
    def test_triliun(self):
        assert format_rupiah_compact(404_400_000_000_000) == "Rp404,4 T"

    def test_miliar(self):
        assert format_rupiah_compact(478_987_829_000) == "Rp479,0 M"

    def test_juta(self):
        assert format_rupiah_compact(12_345_678) == "Rp12,3 jt"

    def test_below_one_million_is_full(self):
        assert format_rupiah_compact(654_321) == "Rp654.321"
