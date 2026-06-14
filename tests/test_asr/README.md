# ASR tests

This directory covers the ASR engines currently supported by SubForge:

- WhisperX with MLX Whisper and forced alignment
- whisper.cpp
- Faster Whisper
- OpenAI-compatible Whisper APIs
- shared chunking, VAD, timestamp, and post-processing behavior

Run the suite with:

```bash
uv run pytest tests/test_asr -q
```

API integration tests require `OPENAI_BASE_URL` and `OPENAI_API_KEY`. Tests
that need external models, binaries, services, or large audio fixtures should
be marked `integration` or `slow` and skip cleanly when prerequisites are not
available.
