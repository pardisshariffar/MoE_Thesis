# Results log

## 2026-10-10 — M1 Baseline (OLMoE-1B-7B-0125, untouched)

**Setup**
- GPU: RTX 5060 Ti, 16 GB
- Model loaded in fp16, no quantization
- LU data: wikimedia/wikipedia 20231101.lb (switched from Shdorsh/luxembourgish-lu which was too small)
- Seed: 42, single run

**Perplexity (lower = better)**
| Lang | PPL   |
|------|------:|
| en   |  5.43 |
| de   |  7.60 |
| nl   | 11.37 |
| lu   | 21.72 |

**LuxGen generation probe (chrF, 256 samples)**
- BLEU: 0.32
- chrF: 6.99
→ Model handles LU vocabulary (low-ish PPL) but cannot generate coherent LU (near-zero chrF).

**Routing analysis**
- JSD matrix: runs_baseline_2026-10-10/routing_analysis/baseline_jsd_matrix_heatmap.png
- Similarity vs layer: runs_baseline_2026-10-10/routing_analysis/baseline_similarity_vs_layer.png
- [TODO: look at the plots and note the shape]

**Notes for the thesis**
- The gap M3 should close is the generation gap, not the PPL gap. PPL on LU is only moderately worse than NL; chrF reveals the real problem.
- All four PPL numbers are from a single seed. Treat as point estimates until 3-seed re-run.
## 2026-10-29 — Full pipeline M1 / M2 / M3 (RTX 4090, 24 GB, fp16)

**Setup**
- GPU: rented RTX 4090, 24 GB (RunPod)
- Precision: fp16 (NOT 4-bit; 4-bit load exceeded 24 GB on the installed transformers version)
- Trainable: router (mlp.gate) + lm_head only; all 64 experts/layer frozen (~105 M params, 1.5%)
- Training: 1000 steps each for M2 and M3; learning rate 2e-6; BATCH_SIZE_MONO = BATCH_SIZE_PARALLEL = 1 with gradient accumulation
- Alignment (M3): JS divergence, layers 5-10, lambda = 0.1; pairs EN-DE, EN-NL, DE-NL
- Data: EN/DE/NL Wikipedia (wikimedia/wikipedia, 20231101) ~1 M tokens; LU held out (never trained or aligned)
- Eval: perplexity EN/DE/NL/LU; LU generation chrF/BLEU on 20% held-out split, 256 samples
- Seed: 42, single run
- Source files: runs/eval/comparison_metrics.csv, runs/eval/eval_{baseline,cont,align}.json, runs/routing_analysis/*_lang_jsd_matrix.csv

**Perplexity (lower = better)** — from comparison_metrics.csv
| Lang | M1 baseline | M2 cont | M3 align |
|------|------:|------:|------:|
| en   |  5.43 |  5.61 |  5.17 |
| de   |  7.60 |  6.64 |  6.24 |
| nl   | 11.37 |  9.72 |  8.95 |
| lu (held-out) | 21.72 | 30.04 | 35.71 |

**LU generation (chrF / BLEU)**
| Metric | M1 | M2 | M3 |
|--------|---:|---:|---:|
| chrF | 7.01 | 6.45 | 6.10 |
| BLEU | 0.32 | 0.24 | 0.20 |

**Routing (EN-LU JSD, mean layers 5-10)**
| | M1 | M2 | M3 |
|--|---:|---:|---:|
| EN-LU | 0.0415 | 0.0350 | 0.0334 (-19.5%) |

**Finding**
- Alignment improves every trained language (M3 best on EN/DE/NL) but harms held-out LU (PPL 21.72 -> 35.71).
- Routing transferred to LU (EN-LU divergence fell ~20%) yet LU degraded: the method reached the held-out language, and the transfer was harmful. Hypothesis (beneficial transfer) not supported.

**Expert-distribution analysis (from *_expert_load.csv)**
- Corrected full pairwise JSD matrix (the saved matrix had de-lu / nl-lu as 0.0 because only 4 pairs were computed; code fixed on 2026-11-13):
  en-de 0.0256->0.0205, en-nl 0.0240->0.0197, en-lu 0.0415->0.0334, de-nl 0.0043->0.0042, de-lu 0.0087->0.0086, nl-lu 0.0111->0.0092 (baseline -> align).
- Only pairs that include English converge (~20%); pairs without English barely move: alignment pulls routing toward English.
- DE/NL/LU share the same top experts in every middle layer (e.g. expert 33 in L8, expert 27 in L7); English uses a different set. LU-DE = 0.0087 vs LU-EN = 0.0415: LU natively routes with the continental Germanic cluster, not English.

**Limitations**
- Single seed; treat as point estimates until a 3-seed re-run.
- Convergence is in the overall 64-expert distribution; the identity of the very top experts changes little.

**Code state at this run**
- fp16 loader (use_quant only below 20 GB VRAM), batch size 1, trainable-params-only checkpoints. These fixes and the JSD-matrix pair fix were committed to the repo on 2026-11-13.
