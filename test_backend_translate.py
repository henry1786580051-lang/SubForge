#!/usr/bin/env python3
"""Test translation exactly as the backend does it.

Requires environment variables:
  OPENAI_API_KEY   — LLM API key
  OPENAI_BASE_URL  — LLM base URL (e.g. https://token-plan-cn.xiaomimimo.com/v1)

Usage:
  export OPENAI_API_KEY=your-key
  export OPENAI_BASE_URL=https://your-endpoint/v1
  python test_backend_translate.py
"""
import os
import sys
import traceback

# Set proxy if needed
os.environ.setdefault('HTTP_PROXY', 'http://127.0.0.1:7897')
os.environ.setdefault('HTTPS_PROXY', 'http://127.0.0.1:7897')

if not os.environ.get('OPENAI_API_KEY'):
    print("ERROR: Set OPENAI_API_KEY environment variable first")
    sys.exit(1)

sys.path.insert(0, 'backend')

print("Step 1: Import modules...")
from subforge.core.asr.asr_data import ASRData
from subforge.core.translate.factory import TranslatorFactory
from subforge.core.translate.types import TargetLanguage, TranslatorType

print("Step 2: Load subtitle file...")
srt_path = '/Users/guwenhan/Desktop/YouTube/2027 Nissan Z NISMO (6-Speed Manual) - POV More Driving Impressions.srt'
asr_data = ASRData.from_subtitle_file(srt_path)
print(f"  Loaded {len(asr_data.segments)} segments")

print("Step 3: Create translator...")
translator = TranslatorFactory.create_translator(
    translator_type=TranslatorType.OPENAI,
    thread_num=1,
    batch_num=10,
    target_language=TargetLanguage.SIMPLIFIED_CHINESE,
    model='mimo-v2.5-pro',
    custom_prompt='',
    is_reflect=False,
)
print(f"  Translator type: {type(translator).__name__}")

print("Step 4: Translate...")
try:
    result = translator.translate_subtitle(asr_data)
    print(f"  Translation complete: {len(result.segments)} segments")
    for seg in result.segments[:5]:
        print(f"    {seg.text} -> {seg.translated_text}")
except Exception as e:
    print(f"  TRANSLATION FAILED: {type(e).__name__}: {str(e)}")
    traceback.print_exc()
