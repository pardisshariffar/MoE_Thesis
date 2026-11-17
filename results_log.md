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

## 2026-11-14 — Model 4 (router + LoRA on experts) implemented (not yet run)

**What M4 is**
- Same alignment objective, data, layers (5-10), lambda (0.1) and 1000-step schedule as M3, but each expert also carries a trainable low-rank adapter. Trainable set = router (mlp.gate) + lm_head + expert LoRA; the 6.8B expert backbone stays frozen. This is the router-only (M3) vs router+LoRA (M4) comparison.
- In transformers >=5.x the 64 experts per layer are fused into 3-D tensors inside one OlmoeExperts module (gate_up_proj [E, 2I, H], down_proj [E, H, I]), not separate nn.Linear modules, so off-the-shelf PEFT target_modules cannot attach to them. M4 adds LoRA directly to the fused weights via a module swap (expert_lora.py: LoRAExperts / inject_expert_lora). Router module class is untouched, so the routing hooks still fire; LoRA B is zero-initialised, so step 0 equals the base model.
- Config: LORA_R=8, LORA_ALPHA=16, LORA_DROPOUT=0.0, LEARNING_RATE_LORA=2e-6, MAX_STEPS_MODEL4=1000. Checkpoint: runs/checkpoints/model4_lora_align (trainable-params-only format; the eval loader detects the LoRA tensors and rebuilds the adapter before overlaying).

**Verified before running (CPU, tiny 2-layer/4-expert OLMoE)**
- LoRA injects into every MoE block; model is identical to base at init; trainable set is exactly router+lm_head+LoRA with the base experts frozen.
- The alignment gradient (through the real routing hooks + JSD loss) reaches both the expert LoRA and the router.
- Train -> save (fp16, trainable-only) -> rebuild -> eval-reload reproduces the trained model's logits (fp16 tolerance).

**How to run (rented >=24 GB GPU recommended; experts stay fp16 even under 4-bit, so 16 GB is too tight)**
- Train:  python models_and_training.py --mode lora_align
- Eval all four and rebuild the comparison:  python models_and_training.py --mode eval --eval-model all
- M4 joins comparison_metrics.csv as model_type "lora_align" once its checkpoint exists.
model_type,loaded_from,eval_json_path,luxgen_samples_path,ppl_en,ppl_de,ppl_nl,ppl_lu,ppl_mean,luxgen_bleu,luxgen_chrf,luxgen_token_f1_fallback
baseline,allenai/OLMoE-1B-7B-0125,/workspace/MoE_Thesis/runs/eval/eval_baseline.json,/workspace/MoE_Thesis/runs/eval/luxgen_samples_baseline.jsonl,5.427284770091097,7.602523482701854,11.369216538801641,21.71682167939589,11.52896161774762,0.3205212049734573,6.988888931509807,
cont,/workspace/MoE_Thesis/runs/checkpoints/model2_cont,/workspace/MoE_Thesis/runs/eval/eval_cont.json,/workspace/MoE_Thesis/runs/eval/luxgen_samples_cont.jsonl,5.71525888134181,6.71081548661989,9.720937384110556,30.399017394901612,13.136507286743466,0.23184640359471972,6.347310459426672,
align,/workspace/MoE_Thesis/runs/checkpoints/model3_align,/workspace/MoE_Thesis/runs/eval/eval_align.json,/workspace/MoE_Thesis/runs/eval/luxgen_samples_align.jsonl,5.244831986673494,6.274907368103452,8.969798632045121,34.95094226642259,13.860120063311163,0.17515537288766567,6.025998700310362,
lora_align,/workspace/MoE_Thesis/runs/checkpoints/model4_lora_align,/workspace/MoE_Thesis/runs/eval/eval_lora_align.json,/workspace/MoE_Thesis/runs/eval/luxgen_samples_lora_align.jsonl,5.009859344015603,5.946331630146233,8.506926535885231,33.408658488761034,13.217943999702026,0.22977145924081538,6.109306712357769,
