# PATS on ALFWorld: a controlled reproduction

This repository reproduces the central causal claim of [PATS: Policy-Aware Training Scaffolding for Agentic Reinforcement Learning](https://alphaxiv.org/abs/2607.21419): temporary, policy-aware guidance should improve exploration, then transfer into better behavior after the guidance is removed.

**Assessment: partially reproduced.** On a fixed public ALFWorld subset, adaptive removal reached **6.25% scaffold-free held-out success (6/96)**, versus **4.17% (4/96)** for equal-budget no-scaffold RLVR and **3.13% (3/96)** for static cards. Policy-aware cards permanently retained also reached **6.25% (6/96)**, so the named control comparisons point in the predicted direction but removal itself is not isolated. The paper reports 80.71% for PATS, 67.86% for GRPO, and 70.24% for its static warm-start baseline; this reconstruction does not match those absolute numbers.

The support-gap diagnostic moved in the claimed direction: adaptive removal reduced mean absolute supported-versus-unsupported action NLL from 0.842 to 0.242 (71.2%), compared with 61.8% under permanent retention. Complete-trajectory diversity did **not** separate the methods (8/8 unique, 0% duplication), though removal had slightly greater post-warm-up action entropy than retention (2.056 vs 2.051).

This was deliberately downscaled to Qwen2.5-1.5B-Instruct, six fixed training tasks, twelve fixed `valid_unseen` tasks, 16 iterations, eight rollouts per task, and eight seeds per condition. We reconstructed missing evidence-card and removal details, used ALFWorld's audited public expert-plan metadata, and scored official admissible actions rather than generating unconstrained text. Formal runs used **OpenResearch Kubernetes**, **NVIDIA RTX PRO 6000 Blackwell** GPUs, a peak of **16 concurrent GPUs**, and **6.153805 elapsed wall-hours**.

- [Read the illustrated report](reports/pats-alfworld/report.md)
- [Explore the self-contained marimo notebook](notebooks/pats_alfworld_reproduction.py)
- [Inspect the frozen terminal-log evidence](reports/pats-alfworld/data/results.json)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/pats-policy-aware-training-scaffolding-for-agent/blob/main/notebooks/pats_alfworld_reproduction.py)

## Experiment log

Every formal experiment ran the exact command shown below. Links point to representative immutable branches; all seed/run provenance is frozen in the result JSON and notebook.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public report, notebook, figures, and frozen evidence | Not run as an experiment (publication surface) | Presentation only | — |
| [No scaffold, seed 0](https://github.com/alphaXiv/pats-policy-aware-training-scaffolding-for-agent/tree/orx/audited-no-scaffold-seed-0) | Equal-budget RLVR control; eight matched seeds | `bash scripts/run.sh` | 4.17% scaffold-free success (4/96) | Kubernetes, 1× RTX PRO 6000 Blackwell per seed |
| [Static cards, seed 0](https://github.com/alphaXiv/pats-policy-aware-training-scaffolding-for-agent/tree/orx/audited-static-audited-cards-seed-0) | Permanent unmodified evidence-card control; eight matched seeds | `bash scripts/run.sh` | 3.13% scaffold-free success (3/96) | Kubernetes, 1× RTX PRO 6000 Blackwell per seed |
| [Adaptive removal, seed 0](https://github.com/alphaXiv/pats-policy-aware-training-scaffolding-for-agent/tree/orx/audited-adaptive-removal-audited-cards-seed-0) | Policy-aware revision and competence-triggered removal; eight matched seeds | `bash scripts/run.sh` | 6.25% (6/96); directionally aligned vs named controls | Kubernetes, 1× RTX PRO 6000 Blackwell per seed |
| [Cards retained, seed 0](https://github.com/alphaXiv/pats-policy-aware-training-scaffolding-for-agent/tree/orx/audited-adaptive-retained-audited-cards-seed-0) | Same policy-aware training, removal disabled; eight matched seeds | `bash scripts/run.sh` | 6.25% (6/96); ties adaptive removal | Kubernetes, 1× RTX PRO 6000 Blackwell per seed |
| [Valid extension, adaptive seed 4](https://github.com/alphaXiv/pats-policy-aware-training-scaffolding-for-agent/tree/orx/valid-extended-adaptive-seed-4) | Fresh committed-config extension for seeds 4–7 in every arm | `bash scripts/run.sh` | Completed; reduced the preliminary adaptive estimate to 6.25% | Kubernetes, 1× RTX PRO 6000 Blackwell per seed |
| [Threshold sweep, 0.00 seed 0](https://github.com/alphaXiv/pats-policy-aware-training-scaffolding-for-agent/tree/orx/removal-threshold-0-00-seed-0) | Controller sensitivity at thresholds 0.00, 0.10, 0.15, 0.20, and primary 0.25; four seeds each | `bash scripts/run.sh` | Success rises 4.17% → 10.42%; aligned robustness result | Kubernetes, 1× RTX PRO 6000 Blackwell per seed |

## Reproduce or inspect

Formal training is configured through `configs/` and launched by:

```bash
bash scripts/run.sh
```

The command installs the public ALFWorld dependencies, materializes the fixed subset, trains the selected condition, and prints one terminal `FINAL_METRICS` JSON record. To regenerate the frozen figures after the listed runs are available locally:

```bash
python -m pip install -r requirements-analysis.txt
python analysis/build_artifacts.py
```

The notebook opens published evidence without requiring expensive reruns:

```bash
marimo edit notebooks/pats_alfworld_reproduction.py
```

See the [report](reports/pats-alfworld/report.md) for the protocol, claim-by-claim comparison, figures, interpretation, and limitations.
