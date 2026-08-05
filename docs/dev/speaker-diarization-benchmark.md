# Speaker diarization benchmark

SubForge separates accuracy evaluation from unlabeled product regression. Do not
report agreement on a YouTube video as diarization accuracy unless speaker RTTM
ground truth exists.

## Dataset roles

| Dataset | Role | Tuning policy |
| --- | --- | --- |
| AMI IHM/SDM | Meeting and far-field accuracy | Tune on development split only |
| VoxConverse | Primary real-world YouTube accuracy | Tune on development split only |
| MSDWild subset | Overlap and multilingual stress | Optional; observe its license |
| Jubilee and user videos | Unlabeled stability regression | Never treat as ground truth |

Freeze thresholds before running official evaluation/test splits. Keep known-speaker
and automatic-speaker reports separate.

Official sources:

- [AMI corpus downloads](https://groups.inf.ed.ac.uk/ami/download/)
- [AMI diarization protocol](https://github.com/pyannote/AMI-diarization-setup)
- [VoxConverse](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/)
- [MSDWild](https://x-lance.github.io/MSDWILD/)

## Strict scoring

The benchmark uses zero collar and includes overlapping speech. It reports DER,
missed detection, false alarm, speaker confusion, JER, 250/500 ms boundary F1,
speaker count, fragmentation, short islands, and overlap duration.

Strict DER/JER must score Community-1 regular diarization. Exclusive diarization is
retained for SubForge word ownership and reported as a product diagnostic only. Its
DER and boundary F1 are not directly comparable to a regular overlapping reference,
because exclusive diarization deliberately keeps only one active speaker at a time.

AMI manual word annotations are used to score anonymous speaker labels after optimal
one-to-one mapping. Report all-word WDER for context, but use non-overlap WDER as the
release metric so unavoidable exclusive-output errors during crosstalk do not distort
the product decision.

```bash
uv run --extra whisperx python scripts/benchmark_diarization.py score \
  reference.rttm hypothesis.rttm --uem evaluation.uem --uri recording-id \
  --report artifacts/diarization/recording-id.json
```

Run Community-1 with a known speaker count:

```bash
uv run --extra whisperx python scripts/benchmark_diarization.py run audio.wav output.rttm \
  --num-speakers 5 --model-dir /path/to/model --reference reference.rttm \
  --uem evaluation.uem --uri recording-id --no-cache \
  --report artifacts/diarization/run.json
```

Automatic mode is bounded to 2-10 speakers:

```bash
uv run --extra whisperx python scripts/benchmark_diarization.py run audio.wav output.rttm \
  --auto-speakers --min-speakers 2 --max-speakers 10 --model-dir /path/to/model
```

Score production word ownership against AMI manual annotations:

```bash
uv run python scripts/benchmark_word_speakers.py \
  --meeting IB4010 --words-dir /path/to/ami/words \
  --reference-rttm reference.rttm --hypothesis-rttm exclusive.rttm \
  --uri IB4010.Mix-Headset --output artifacts/diarization/IB4010-words.json
```

Validate the production acoustic gate with exclusive assignment turns and the
matching regular diarization track used only for overlap protection:

```bash
uv run python scripts/benchmark_ami_speaker_embedding_gate.py \
  --meeting IB4010 --uri IB4010.Mix-Headset --audio audio.wav \
  --words-dir /path/to/ami/words --reference-rttm reference.rttm \
  --hypothesis-rttm exclusive.rttm --overlap-rttm regular.rttm \
  --model /path/to/community-1 --device mps --margin 0.10 \
  --verification-model-dir /path/to/models --confirmation-margin -0.05 \
  --output artifacts/diarization/IB4010-embedding-gate.json
```

## Unlabeled stability

Run the same source after a controlled perturbation or with overlapping windows,
then compare the RTTM files. The result is agreement, not accuracy.

```bash
uv run --extra whisperx python scripts/benchmark_diarization.py stability \
  baseline.rttm candidate.rttm --report artifacts/diarization/stability.json
```

Use the highest-disagreement windows as a compact regression set. A production
change is acceptable only when text and word timestamps remain unchanged, strict
DER does not regress materially on held-out data, unassigned words do not increase,
and runtime remains within the release budget.

## Release gate

Keep the current production behavior unless all applicable checks pass:

- ASR text and word timestamps are byte-for-byte unchanged.
- Held-out strict DER does not regress by more than 0.3 percentage points.
- Speaker confusion or speaker-attributed word error improves without materially
  increasing missed speech or false alarms.
- Unassigned words do not increase.
- Unlabeled short islands and speaker-run fragmentation do not increase.
- Single-speaker sources are not split into multiple speakers.
- Apple Silicon runs do not silently fall back from MPS to CPU.
- End-to-end runtime does not regress by more than 10 percent.

Direct grammar-based speaker reassignment remains disabled because AMI word-level
ablation showed a consistent regression. Grammar rules may only propose a candidate;
production accepts it when Community-1 exceeds the current speaker by the frozen
0.10 margin and an independent WeSpeaker ECAPA512-LM embedding does not strongly
contradict the move. For two-speaker recordings only, a narrow consensus rescue can
accept a 0.075 Community-1 margin when ECAPA independently reaches 0.15. This rescue
is disabled for three or more speakers. Overlap regions and speakers without stable
references are never changed. Any verifier failure must retain the conservative
labels and complete the transcription.
