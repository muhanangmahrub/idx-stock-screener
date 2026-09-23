"""Unit test penyimpanan isian pemilik (blacklist, blue chip, checklist, dll.)."""

import json

import pytest

from screener.storage import (
    KEY_BLACKLIST,
    KEY_BLUE_CHIPS,
    codes_to_text,
    load_codes,
    load_store,
    load_value,
    save_codes,
    save_store,
    save_value,
)


@pytest.fixture
def store(tmp_path):
    return tmp_path / "user_data.json"


class TestRoundTrip:
    def test_value_survives_save_and_load(self, store):
        assert save_value("checklist", {"gcg": "Ya"}, store)
        assert load_value("checklist", path=store) == {"gcg": "Ya"}

    def test_saving_one_key_keeps_the_others(self, store):
        save_value(KEY_BLACKLIST, ["ABCD"], store)
        save_value(KEY_BLUE_CHIPS, ["BBCA"], store)
        assert load_value(KEY_BLACKLIST, path=store) == ["ABCD"]
        assert load_value(KEY_BLUE_CHIPS, path=store) == ["BBCA"]

    def test_file_is_readable_json(self, store):
        save_store({"portfolio": [{"ticker": "BBCA.JK", "value": 1000}]}, store)
        assert json.loads(store.read_text())["portfolio"][0]["ticker"] == "BBCA.JK"


class TestCodes:
    def test_codes_are_normalised_to_upper_case(self, store):
        save_codes(KEY_BLUE_CHIPS, ["bbca", " tlkm ", "BBRI"], store)
        assert load_codes(KEY_BLUE_CHIPS, store) == {"BBCA", "TLKM", "BBRI"}

    def test_blank_entries_are_dropped(self, store):
        save_codes(KEY_BLACKLIST, ["ABCD", "", "   "], store)
        assert load_codes(KEY_BLACKLIST, store) == {"ABCD"}

    def test_comma_string_is_accepted_when_reading(self, store):
        save_value(KEY_BLACKLIST, "abcd, efgh", store)
        assert load_codes(KEY_BLACKLIST, store) == {"ABCD", "EFGH"}

    def test_codes_to_text_is_sorted_and_editable(self):
        assert codes_to_text({"TLKM", "BBCA"}) == "BBCA, TLKM"


class TestFailuresAreSoft:
    """Kegagalan penyimpanan tidak boleh menjatuhkan aplikasi."""

    def test_missing_file_reads_as_empty(self, tmp_path):
        assert load_store(tmp_path / "belum_ada.json") == {}
        assert load_value("apa pun", "bawaan", tmp_path / "belum_ada.json") == "bawaan"

    def test_corrupt_file_reads_as_empty(self, store):
        store.write_text("{ rusak")
        assert load_store(store) == {}

    def test_non_dict_content_reads_as_empty(self, store):
        store.write_text("[1, 2, 3]")
        assert load_store(store) == {}

    def test_unwritable_path_returns_false(self, tmp_path):
        assert save_store({"a": 1}, tmp_path / "tidak" / "ada" / "x.json") is False

    def test_unserialisable_value_returns_false(self, store):
        assert save_value("x", {1, 2, 3}, store) is False
