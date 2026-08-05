# Speaker diarization baseline

Date: 2026-08-05

This baseline uses Community-1 on Apple MPS with zero collar and overlapping speech
included. It contains two VoxConverse development recordings and two AMI validation
meetings in both IHM and SDM conditions. Every recording is run with a known speaker
count and with automatic 2-10 speaker detection.

| Dataset and mode | Strict regular DER | Miss | False alarm | Confusion | RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| VoxConverse known | 8.81% | 1.13% | 2.60% | 5.08% | 0.042 |
| VoxConverse auto | 6.47% | 1.13% | 2.60% | 2.74% | 0.032 |
| AMI IHM known | 16.96% | 5.73% | 4.88% | 6.34% | 0.035 |
| AMI IHM auto | 16.96% | 5.73% | 4.88% | 6.34% | 0.040 |
| AMI SDM known | 18.53% | 6.65% | 4.79% | 7.09% | 0.036 |
| AMI SDM auto | 18.53% | 6.65% | 4.79% | 7.09% | 0.036 |

All 12 runs completed on MPS without CPU fallback. AMI automatic mode detected four
speakers correctly in every run. VoxConverse detected the four-speaker recording
correctly but estimated six speakers for a seven-speaker recording. The lower DER
for that automatic run means the seventh reference speaker contributes little speech;
forcing seven clusters increased confusion, so speaker-count accuracy and DER must
remain separate acceptance criteria.

The AMI scores are close to Community-1's published reference range. Global clustering
or segmentation threshold changes are therefore not justified by this sample.

## Word ownership

AMI manual word timings provide 24,882 scored words outside overlapping speech. The
same Community-1 exclusive turns were assigned to those words before and after
SubForge's boundary smoothing:

| Input | Legacy grammar smoothing | Conservative assignment | Dual-model verification |
| --- | ---: | ---: | ---: |
| AMI IHM IB4010 | 5.15% | 4.87% | 4.69% |
| AMI IHM IB4011 | 4.12% | 3.29% | 3.17% |
| AMI SDM IB4010 | 7.14% | 6.24% | 6.24% |
| AMI SDM IB4011 | 5.19% | 4.08% | 4.01% |
| **Word-weighted WDER** | **5.46%** | **4.70%** | **4.61%** |

Grammar-based moves degraded all four conditions, with continuation snapping causing
the largest regression. Production therefore uses those rules only to propose a
candidate change. Community-1 proposes the acoustic decision and WeSpeaker
ECAPA512-LM independently checks it with robust temporary centroids. The verifier
does not store enrolled voiceprints. A strong ECAPA contradiction vetoes the move;
a narrow two-speaker-only consensus path can recover candidates just below the
Community-1 threshold. Regions with overlapping speech or without a stable two-second
speaker reference are never changed. The gate reduced non-overlap errors from 1,169
to 1,146, a 2.0% relative improvement over conservative assignment, with no condition
regressing. Text and timestamps remain unchanged.

Two real-video regressions use the same production path. A 44:51 two-speaker interview
kept all 7,877 words and timestamps identical while preserving the existing boundary
fixes and recovering the missing start of "I'm glad to be here." An 18:03 interview
with six detected voice classes
kept all 3,766 words, timestamps, and speaker labels identical because no proposal met
the evidence threshold. ECAPA512-LM adds about 25 MB of external model data and runs
through ONNX Runtime on CPU; after initialization, a four-second clip takes about
7 ms on the tested Apple Silicon machine. The verifier reads only candidate and
temporary reference clips rather than loading a second full-audio copy.

Dataset preparation rejected no sampled records. The VoxConverse preparation script
still validates every timestamp against decoded audio duration because the upstream
preprocessed dataset has a reported timestamp/audio mismatch in some rows.
