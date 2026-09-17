import json
from pathlib import Path

import pytest

from subforge.core.asr.asr_data import ASRDataSeg, ASRWord
from subforge.core.split.boundary import _repair_repeated_identifiers, normalize_boundaries
from subforge.core.split.boundary_detectors.entity import typed_technical_phrase
from subforge.core.split.boundary_features import extract_english_boundary_features


def cues(ids):
    data = json.loads((Path(__file__).parent / 'fixtures/ioniq_boundaries.json').read_text())
    return [ASRDataSeg(text=c['text'], start_time=c['start'], end_time=c['end'],
                       speaker_id=c['speaker'], words=[
                           ASRWord(text=w['text'], start_time=w['start'], end_time=w['end'],
                                   speaker_id=w['speaker']) for w in c['words']])
            for c in data if c['id'] in ids]


@pytest.mark.parametrize('number,phrase', [(35, 'Level 2 charger'), (39, 'IONIQ 5,'),
    (190, 'parking cameras'), (675, 'sound system test song'), (681, 'Bose systems'),
    (700, 'IONIQ 5 XRT')])
def test_actual_word_alignment_preserves_phrases_and_words(number, phrase):
    original = cues([5, 19, 27, number, number + 1])
    result = normalize_boundaries(original)
    tail = [c for c in result if c.start_time >= original[-2].start_time]
    assert any(phrase in c.text for c in tail)
    assert [w for c in result for w in c.words] == [w for c in original for w in c.words]
    assert all(c.text.split()[-1].lower() not in {'the', 'long-range'} for c in tail)
    assert [(c.text, c.start_time, c.end_time) for c in normalize_boundaries(result)] == [
        (c.text, c.start_time, c.end_time) for c in result]


@pytest.mark.parametrize('case', ['speaker', 'missing', 'mismatch', 'limit', 'single_reference'])
def test_identifier_repair_declines_unsafe_alignment(case):
    original = cues([5, 19, 700, 701])
    if case == 'speaker':
        original[-1].speaker_id = 'different'
    if case == 'missing':
        original[-1].words = []
    if case == 'mismatch':
        original[-1].text += ' extra'
    if case == 'single_reference':
        original = original[1:]
    assert _repair_repeated_identifiers(original, hard_max_words=4 if case == 'limit' else 22) == original


def test_edited_source_with_unmatched_raw_words_is_not_retimed():
    original = cues([5, 19, 596, 597])
    assert _repair_repeated_identifiers(original, hard_max_words=22) == original


def test_translated_cues_are_not_rebuilt():
    original = cues([5, 19, 700, 701])
    original[-1].translated_text = '保留译文'
    assert normalize_boundaries(original) == original


@pytest.mark.parametrize('left,right', [('Go to level', '2 of the building.'),
    ('We finished parking.', 'Cameras are useful.'), ('The sound system.', 'Test songs later.'),
    ('It is Level 2.', 'Charging begins.'), ('There are 305,', '310 is next.')])
def test_technical_detector_does_not_join_unrelated_sentences(left, right):
    assert not typed_technical_phrase(extract_english_boundary_features(left, right))
