#!/usr/bin/env python3
"""Test subtitle optimization and translation."""
import os
import sys

# Set environment variables
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['OPENAI_API_KEY'] = 'tp-c3fni8mj1ffe9wtj6vj0pkyttbv595sk0jg1xmdovqtaj045'
os.environ['OPENAI_BASE_URL'] = 'https://token-plan-cn.xiaomimimo.com/v1'

sys.path.insert(0, 'backend')

from subforge.core.asr.asr_data import ASRData
from subforge.core.translate.factory import TranslatorFactory
from subforge.core.translate.types import TargetLanguage, TranslatorType

# Load subtitle
srt_path = '/Users/guwenhan/Desktop/YouTube/2027 Nissan Z NISMO (6-Speed Manual) - POV More Driving Impressions.srt'
asr_data = ASRData.from_subtitle_file(srt_path)
print(f'Loaded {len(asr_data.segments)} segments')

# Test with just 3 segments
test_segments = asr_data.segments[:3]
print(f'Testing with {len(test_segments)} segments...')

# Create test ASRData
test_data = ASRData(test_segments)

# Create LLM translator
print('Creating LLM translator...')
translator = TranslatorFactory.create_translator(
    translator_type=TranslatorType.OPENAI,
    thread_num=1,
    batch_num=3,
    target_language=TargetLanguage.SIMPLIFIED_CHINESE,
    model='mimo-v2.5-pro',
    custom_prompt='',
    is_reflect=False,
)
print('Translating...')
result = translator.translate_subtitle(test_data)
print(f'Translation complete: {len(result.segments)} segments')
for seg in result.segments:
    print(f'  {seg.text} -> {seg.translated_text}')
