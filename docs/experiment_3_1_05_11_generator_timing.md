# 3.1.05_11: Generator-level timing modes

## Question

The 3.1.05_10 spatial-top-five timing selector failed independent validation even though its fixed-path timing Oracle was large. That experiment only warped completed trajectories. This experiment tests whether longitudinal timing becomes a useful **generator mode** when both its data target and the two-step diffusion decoder represent early/late progress explicitly.

## Intervention

- The original K67 generator, PCS scorer and Cost Ranker are unchanged. The 67 original candidates remain available in the final 201-candidate set.
- Each of the 67 spatial anchors has an early and late time-indexed seed. The new side branch runs the frozen two-step DDIM decoder on those 134 seeds. A scene-conditioned vector containing **all eight displacement intervals** enters both decoder layers at both denoising steps. Trainable per-layer pose residuals have zero initialization. No final trajectory warp is applied at inference.
- For each original candidate, the training-label builder scores that candidate and four offline time-profile variants in **one official metric-cache scoring call**. It chooses a distinct target within each early/late family only when NC, DAC, TTC, comfort and direction do not worsen and PDM loss is at most 0.02. Unsafe or near-identical variants have zero training weight. The two modes never share one human-GT XY target.
- Only the timing conditioner and pose adapters train. Original generator, perception, PCS and Ranker weights are frozen. A scene with no eligible timing targets is excluded from adapter training. The data cache records all K67 parent indices, without top-five filtering.

## Validation boundary

Use navtrain for labels and log-separated navtrain val for calibration/audit. The fixed 20-epoch adapter checkpoint is evaluated once; audit cannot pick an epoch or threshold. The validation compares 201 candidates in each arm: generated timing modes, two fixed post-generation warps, and two extra stochastic samples. It reports official candidate Oracle and the frozen PCS+Ranker selected PDM. Each arm picks its threshold only on calibration. `pass_for_navtest` requires positive and safe audit gain, no severe audit loss, at least 0.1 PDM point more audit Oracle gain, and at least 0.05 PDM point more audit selected gain than both equal-count controls. The navtest command rejects a failed decision.

The offline profile variants provide supervision, not proof that the learned branch is useful. A strong Oracle from the label cache only says the timing targets exist. The decisive results are the **new generated candidates**, their independent audit gain, and the equal-count controls. Existing 3.1.05_10 labels and checkpoints are intentionally not reused.

## Operational notes

Run `bash scripts/pcs/run_3_1_05_11.sh` for commands. The default physical training GPUs are UUIDs on PCI buses 34–37 (physical cards 0–3); the four Ada cards remain available for other runs. Preparation is CPU metric scoring and resumable by block. Training reads existing K67 candidate contexts and creates a separate teacher cache; it does not duplicate the 205 GiB candidate cache. See `training_and_eval_3_1_05_11.txt` for exact paths and steps.

Local training/smoke tests were not run; server unit tests and pilot are required before full cache preparation.
