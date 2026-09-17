from screener.checklist import (
    ANSWER_NO,
    ANSWER_UNCHECKED,
    ANSWER_YES,
    QUALITATIVE_CHECKLIST,
    summarize_checklist,
)


def test_all_items_unchecked_by_default():
    summary = summarize_checklist({})
    assert summary.unchecked == list(QUALITATIVE_CHECKLIST.values())
    assert summary.is_complete is False


def test_groups_answers_and_keeps_question_text():
    answers = {
        "deep_dive": ANSWER_YES,
        "gcg": ANSWER_NO,
        "catalyst": ANSWER_UNCHECKED,
    }
    summary = summarize_checklist(answers)
    assert summary.yes == [QUALITATIVE_CHECKLIST["deep_dive"]]
    assert summary.no == [QUALITATIVE_CHECKLIST["gcg"]]
    assert len(summary.unchecked) == len(QUALITATIVE_CHECKLIST) - 2


def test_complete_when_every_item_answered():
    answers = {key: ANSWER_YES for key in QUALITATIVE_CHECKLIST}
    assert summarize_checklist(answers).is_complete is True
