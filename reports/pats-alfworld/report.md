# PATS on ALFWorld: do temporary hints become unaided skill?

Training an agent to solve household tasks by trial and error can stall before it ever discovers a reward. PATS proposes giving the learner temporary, experience-based hints, then withdrawing them as behavior improves so that the policy—not the hints—does the work. We tested that causal story on a fixed public ALFWorld subset by asking whether adaptive removal beats equal-budget training and permanent guidance when every final evaluation hides the hints.

**Verdict: partially reproduced.** Adaptive removal achieved **10.42%** scaffold-free held-out success (5/48 episodes), compared with **4.17%** for both no-scaffold RLVR and static cards (2/48 each), and **6.25%** with policy-aware cards permanently retained (3/48). The ordering matches the claim, but the experiment is far smaller than the paper and the advantage is only three, three, and two episodes respectively.

**Scope.** This is a controlled reduced-scale test: four matched seeds, six fixed training tasks, twelve fixed `valid_unseen` tasks, and 16 training iterations per seed. It is not a numerical replication of the paper's full ALFWorld campaign.

![Primary held-out result](images/headline_success.png)

**How to read this figure.** Bars average four seeds, dots show each seed, and whiskers are seed standard deviations. All conditions used the same training and evaluation tasks and rollout budget; “without cards” means no evidence card was present at final evaluation. Adaptive removal leads, but the overlapping seed outcomes show why the assessment is partial.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/pats-policy-aware-training-scaffolding-for-agent/blob/main/notebooks/pats_alfworld_reproduction.py)

## What was tested

The reconstruction uses the public ALFWorld 0.3.5 TextWorld environment and Qwen2.5-1.5B-Instruct with a rank-16 LoRA adapter. One fixed training task from each of six task families supplies binary environment reward; two unseen tasks per family are held out. The policy scores ALFWorld's official admissible actions, samples eight complete trajectories per task and iteration, and receives a group-relative REINFORCE update from the same environmental reward in every arm.

Because no author code is available, evidence cards had to be reconstructed. We executed ALFWorld's public expert-plan metadata once per training task, accepted it only when the environment reported success, and compressed the trace into a task-family card. Static training never changes that card. Policy-aware training revises cards from recent outcomes; the adaptive arm removes a family card after a four-iteration warm-up when its success moving average crosses 0.25, while the retained arm applies the same training but never removes cards.

The consequential code path is:

```text
verified ALFWorld trace → evidence card → grouped rollouts
environment reward → group-relative LoRA update
recent competence → revise/remove card → card-free held-out evaluation
```

![Training reward dynamics](images/training_success.png)

Cards improved exploration during training: mean grouped-rollout success was 6.05% with static cards, 5.53% with retained policy-aware cards, 4.59% with adaptive removal, and 0.98% without scaffolding. Adaptive reward falls between permanent guidance and no guidance because five of 24 family cards were removed across its four seeds.

## Claim-by-claim evidence

| Claim | Paper | This reproduction | Assessment |
|---|---:|---:|---|
| PATS improves ALFWorld success over equal-budget RL | 80.71% vs 67.86% GRPO | 10.42% vs 4.17% | **Aligned direction; partial at reduced scale** |
| PATS improves over static warm-start guidance | 80.71% vs 70.24% | 10.42% vs 4.17% static cards | **Aligned direction; control is reconstructed** |
| Removing support closes the response-likelihood gap | gap magnitude 0.1170 → 0.0723 (38.2%) | 0.844 → 0.264 (68.7%); retained ends at 0.353 | **Aligned, but retention also narrows it** |
| Removal preserves greater rollout diversity than retained guidance | greater diversity reported | both 8.0/8 unique; 0% duplication | **Inconclusive at a ceiling** |

![Teacher-forced likelihood diagnostic](images/likelihood_gap.png)

The signed quantity is unsupported negative log-likelihood minus supported negative log-likelihood; zero means the card no longer changes the fixed expert action's likelihood. Adaptive removal moved closest to zero on average, but some seeds crossed zero and retained guidance also narrowed the gap. Thus the diagnostic supports internalization during scaffolded training, not a removal-only mechanism.

![Complete-trajectory diversity](images/rollout_diversity.png)

Every arm reached the maximum measured diversity: all eight action sequences in every rollout group were unique. This rules out trajectory collapse in this setup, but it cannot support the paper's claimed diversity advantage over permanent guidance.

## Robustness to the removal rule

![Removal-threshold sensitivity](images/threshold_sensitivity.png)

The reconstructed controller is not indifferent to timing. Removing all cards at iteration 4 produced 4.17% success; competence thresholds 0.10, 0.15, and 0.20 produced 6.25%, 8.33%, and 10.42%. The predeclared 0.25 threshold also reached 10.42%, while retaining slightly more cards (4.75 rather than 4.25 on average). The monotonic rise and plateau support gradual, evidence-triggered withdrawal over abrupt scheduled removal, though each point still contains only 48 episodes.

## Interpretation and limits

The strongest evidence is the matched ordering: adaptive removal beats all three controls on the same 48 held-out episodes, while also producing the largest reduction in absolute support gap. Yet the result is fragile in absolute terms. Seed outcomes for adaptive removal were 16.7%, 0%, 16.7%, and 8.3%; no-scaffold and static controls each succeeded only twice. The fixed tasks improve comparability but do not create 48 independent task draws.

This reconstruction also differs materially from the paper. It uses six rather than hundreds of training tasks, 16 rather than 150 iterations, candidate-action scoring rather than unconstrained generation, and an oracle-like audited expert trace rather than the paper's learned experience bank. It does not test WebShop or search question answering. These substitutions explain the lower absolute success and prevent a full numerical claim.

An additional balanced seed-4–7 extension was externally cancelled before any terminal metrics and is excluded. All reported evidence was produced on **OpenResearch Kubernetes** using **NVIDIA RTX PRO 6000 Blackwell** GPUs, with a peak of **16 GPUs concurrently allocated** and **4.375616 hours** of actual elapsed campaign time. Terminal logs are the evidence source; the [self-contained notebook](../../notebooks/pats_alfworld_reproduction.py) exposes every included seed, secondary campaign, and provenance record without rerunning training.

## Bottom line

At this scale, temporary policy-aware scaffolding produced the paper's predicted ranking and a stronger support-gap contraction, so the central effect is partially reproduced. The data do not establish a diversity advantage, and the small number of successes leaves substantial uncertainty. A full reproduction would need the paper's full 150-step ALFWorld setup, author-equivalent card generation and removal, unconstrained action generation, and more evaluation seeds.

Experiment lineage: [no scaffold](https://github.com/alphaXiv/pats-policy-aware-training-scaffolding-for-agent/tree/orx/audited-no-scaffold-seed-0), [static cards](https://github.com/alphaXiv/pats-policy-aware-training-scaffolding-for-agent/tree/orx/audited-static-audited-cards-seed-0), [adaptive removal](https://github.com/alphaXiv/pats-policy-aware-training-scaffolding-for-agent/tree/orx/audited-adaptive-removal-audited-cards-seed-0), [cards retained](https://github.com/alphaXiv/pats-policy-aware-training-scaffolding-for-agent/tree/orx/audited-adaptive-retained-audited-cards-seed-0), and [threshold sensitivity](https://github.com/alphaXiv/pats-policy-aware-training-scaffolding-for-agent/tree/orx/removal-threshold-0-20-seed-0).
