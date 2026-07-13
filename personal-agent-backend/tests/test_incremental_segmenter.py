import pytest

from app.services.incremental_segmenter import IncrementalSegmenter


def test_emits_complete_sentence_at_strong_boundary():
    segmenter = IncrementalSegmenter()

    assert segmenter.feed("你好") == []
    assert segmenter.feed("呀。今天") == ["你好呀。"]
    assert segmenter.flush() == "今天"


def test_uses_recent_weak_boundary_after_soft_limit():
    segmenter = IncrementalSegmenter(soft_limit=8, hard_limit=14)

    assert segmenter.feed("这是一个很长的开场，后面还有内容") == ["这是一个很长的开场，"]
    assert segmenter.flush() == "后面还有内容"


def test_forces_split_at_hard_limit_without_punctuation():
    segmenter = IncrementalSegmenter(soft_limit=6, hard_limit=10)

    assert segmenter.feed("一二三四五六七八九十十一") == ["一二三四五六七八九十"]
    assert segmenter.flush() == "十一"


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["（微笑）你好。"], ["你好。"]),
        (["回答。|||主动话题"], ["回答。"]),
        (["嗯。", "这是答案。"], ["嗯。这是答案。"]),
    ],
)
def test_filters_non_speech_and_merges_tiny_segments(tokens, expected):
    segmenter = IncrementalSegmenter()
    emitted = []
    for token in tokens:
        emitted.extend(segmenter.feed(token))
    tail = segmenter.flush()
    if tail:
        emitted.append(tail)

    assert emitted == expected
