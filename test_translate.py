#!/usr/bin/env python3
"""Test subtitle optimization and translation.

Requires environment variables:
  OPENAI_API_KEY   — LLM API key
  OPENAI_BASE_URL  — LLM base URL (e.g. https://token-plan-cn.xiaomimimo.com/v1)
"""
import os
import sys

# Set proxy if needed
os.environ.setdefault('HTTP_PROXY', 'http://127.0.0.1:7897')
os.environ.setdefault('HTTPS_PROXY', 'http://127.0.0.1:7897')

if not os.environ.get('OPENAI_API_KEY'):
    print("ERROR: Set OPENAI_API_KEY environment variable first")
    sys.exit(1)

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
