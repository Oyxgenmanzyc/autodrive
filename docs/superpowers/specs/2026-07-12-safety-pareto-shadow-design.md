# 3.20.4 Safety-Pareto Shadow Design

## Objective

Replace the counterfactual selector's scalar total-cost comparison with a safety-first Pareto rule. The experiment remains inference-only shadow evaluation and never changes the trajectory returned to PDM.

## Evidence

The 3.20.3 navtest shadow evaluated 35 counterfactual trajectories. Three new TTC failures were selected with dynamic-clearance changes of -0.188 m, -0.014 m, and +0.005 m because total cost improved by as little as 1e-6. Thirty of the 35 candidates did not change brake onset. A candidate therefore needs a measurable safety improvement, not merely a numerically lower mixed cost.

## Selection Rule

Within the classifier top-3 modes, a non-base candidate is safety-Pareto eligible only when all conditions hold:

- Dynamic clearance improves by at least 0.10 m.
- Weighted clearance-plus-timing safety cost improves by at least 0.01.
- Map cost does not regress by more than 0.02.
- Comfort cost does not regress by more than 0.05.

Map and comfort are vetoes only. They cannot compensate for missing safety gain. Eligible candidates are ranked by safety-cost improvement, not total cost. Longitudinal candidates remain preferred; a lateral candidate is allowed only under the existing conditional lateral policy, expressed using safety improvement.

## Diagnostics

Record Pareto candidate count and the selected candidate's clearance gain, safety-cost gain, map regression, comfort regression, and Pareto validity. Preserve existing fields for comparison with earlier CSV files.

## Success Criteria

- A map-only or numerical-noise total-cost improvement cannot activate a counterfactual.
- A candidate that reduces dynamic clearance cannot activate a counterfactual.
- A meaningful clearance and safety-cost improvement remains eligible.
- The actual classifier trajectory remains unchanged.
- No trainable parameter, loss, cache schema, or checkpoint interface changes.
