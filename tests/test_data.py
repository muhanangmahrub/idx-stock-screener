from screener.data import _book_value_currency_matches_price


class TestBookValueCurrencyMatchesPrice:
    def test_same_currency_is_valid(self):
        assert _book_value_currency_matches_price({"currency": "IDR", "financialCurrency": "IDR"})

    def test_usd_statements_with_idr_price_is_invalid(self):
        """Kasus AMMN/BRPT: bookValue USD vs harga IDR -> priceToBook puluhan ribu."""
        assert not _book_value_currency_matches_price(
            {"currency": "IDR", "financialCurrency": "USD"}
        )

    def test_unknown_currency_is_assumed_valid(self):
        assert _book_value_currency_matches_price({"currency": "IDR"})
        assert _book_value_currency_matches_price({"financialCurrency": "USD"})
        assert _book_value_currency_matches_price({})
