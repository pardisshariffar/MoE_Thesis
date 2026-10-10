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