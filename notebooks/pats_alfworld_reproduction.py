import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import textwrap
    import urllib.request

    import marimo as mo

    return json, mo, textwrap, urllib


@app.cell
def _(mo, textwrap):
    mo.md(textwrap.dedent(r"""# Can temporary hints become permanent skill?

Reinforcement-learning agents can fail simply because they do not discover
rewarding behavior early enough. **PATS** proposes temporary, experience-based
guidance that is revised and removed as the policy becomes capable, with the goal
of leaving a better unaided policy behind. This notebook walks through a reduced,
controlled reproduction on the public ALFWorld household-task environment.

The notebook opens the already-produced evidence; it does **not** rerun training.
Every formal metric came from Kubernetes terminal logs.
"""))
    return


@app.cell
def _(json, urllib):
    results_url = (
        "https://raw.githubusercontent.com/alphaXiv/"
        "pats-policy-aware-training-scaffolding-for-agent/main/"
        "reports/pats-alfworld/data/results.json"
    )
    with urllib.request.urlopen(results_url) as response:
        results = json.load(response)
    return results, results_url


@app.cell
def _(mo, results):
    primary = results["aggregates"]["primary"]
    labels = {
        "none": "No scaffold",
        "static": "Static cards",
        "adaptive": "Adaptive removal",
        "adaptive_retain": "Cards retained",
    }
    rows = []
    for _condition in ["none", "static", "adaptive", "adaptive_retain"]:
        item = primary[_condition]
        rows.append(
            {
                "Training condition": labels[_condition],
                "Scaffold-free success": (
                    f"{100 * item['heldout_scaffold_free_success']['mean']:.1f}%"
                ),
                "Seed SD": (
                    f"{100 * item['heldout_scaffold_free_success']['sd']:.1f} pp"
                ),
                "Supported success": (
                    f"{100 * item['heldout_supported_success']['mean']:.1f}%"
                ),
                "Final cards": f"{item['final_active_cards']['mean']:.2f} / 6",
            }
        )
    mo.vstack(
        [
            mo.md("## Verdict and primary result"),
            mo.md(
                "**Partially reproduced.** Adaptive removal reached 6/96 "
                "scaffold-free successes, versus 4/96 without scaffolding and "
                "3/96 with static cards, but tied retained policy-aware cards "
                "at 6/96. The named comparisons align directionally; the "
                "specific benefit of removal remains unresolved."
            ),
            mo.ui.table(rows, selection=None),
        ]
    )
    return labels, primary


@app.cell
def _(mo):
    headline_url = (
        "https://raw.githubusercontent.com/alphaXiv/"
        "pats-policy-aware-training-scaffolding-for-agent/main/"
        "reports/pats-alfworld/images/headline_success.png"
    )
    mo.md(
        f"""
        ![Held-out scaffold-free success by training condition]({headline_url})

        Bars average eight matched seeds; dots are individual seeds and whiskers are
        seed standard deviations. Every seed uses the same 12 held-out task files,
        and cards are hidden for this evaluation.
        """
    )
    return


@app.cell
def _(mo, textwrap):
    mo.md(textwrap.dedent(r"""## What was reconstructed

The paper does not release an implementation, so the experiment makes the missing
choices explicit. A Qwen2.5-1.5B-Instruct policy scores ALFWorld's official
admissible actions. Six fixed training tasks—one from each task family—provide
binary environment reward; twelve fixed `valid_unseen` tasks are held out.

For each training task, the environment's public expert-plan metadata is executed
once and accepted only if ALFWorld reports true success. That audited trajectory
becomes a compact evidence card. Static training always keeps the card; adaptive
training revises it from recent success/failure and removes it family-by-family
after a four-iteration warm-up when the success moving average reaches 0.25. All
arms use the same grouped environmental-reward update and rollout budget.

```text
audited successful trace → evidence card → grouped rollouts
                            ↓
            environment reward → LoRA policy update
                            ↓
             revise / remove when competent
                            ↓
              evaluate held-out tasks with no card
```
"""))
    return


@app.cell
def _(mo):
    training_url = (
        "https://raw.githubusercontent.com/alphaXiv/"
        "pats-policy-aware-training-scaffolding-for-agent/main/"
        "reports/pats-alfworld/images/training_success.png"
    )
    cards_url = (
        "https://raw.githubusercontent.com/alphaXiv/"
        "pats-policy-aware-training-scaffolding-for-agent/main/"
        "reports/pats-alfworld/images/active_cards.png"
    )
    mo.vstack(
        [
            mo.md("## Training and removal dynamics"),
            mo.md(f"![Grouped rollout reward by iteration]({training_url})"),
            mo.md(f"![Number of active evidence cards by iteration]({cards_url})"),
        ]
    )
    return


@app.cell
def _(mo):
    likelihood_url = (
        "https://raw.githubusercontent.com/alphaXiv/"
        "pats-policy-aware-training-scaffolding-for-agent/main/"
        "reports/pats-alfworld/images/likelihood_gap.png"
    )
    diversity_url = (
        "https://raw.githubusercontent.com/alphaXiv/"
        "pats-policy-aware-training-scaffolding-for-agent/main/"
        "reports/pats-alfworld/images/rollout_diversity.png"
    )
    mo.vstack(
        [
            mo.md("## Internalization and diversity diagnostics"),
            mo.md(
                f"""
                ![Supported versus unsupported teacher-forced NLL gap]({likelihood_url})

                Positive values mean the fixed expert action is more likely with a
                card; movement toward zero after removal would be consistent with
                internalization.
                """
            ),
            mo.md(
                f"""
                ![Complete-trajectory and action-distribution diversity]({diversity_url})

                The left panel shows a complete-trajectory ceiling: every arm has
                8/8 unique sequences and 0% duplication. The right panel adds
                post-warm-up action entropy; adaptive removal is only 0.005 nats
                above retained cards, so this is weak directional evidence rather
                than a clear diversity advantage.
                """
            ),
        ]
    )
    return


@app.cell
def _(labels, mo):
    condition_selector = mo.ui.dropdown(
        options=list(labels),
        value="adaptive",
        label="Inspect primary condition",
    )
    condition_selector
    return (condition_selector,)


@app.cell
def _(condition_selector, labels, mo, results):
    records = results["runs"]["primary"][condition_selector.value]
    seed_rows = [
        {
            "Seed": record["seed"],
            "Scaffold-free success": (
                f"{100 * record['heldout_scaffold_free_success']:.1f}%"
            ),
            "Supported success": f"{100 * record['heldout_supported_success']:.1f}%",
            "Final NLL gap": f"{record['final_nll_gap']:.3f}",
            "Final cards": record["final_active_cards"],
            "Run wall time": f"{record['wall_hours']:.3f} h",
        }
        for record in records
    ]
    mo.vstack(
        [
            mo.md(f"### Seed-level evidence: {labels[condition_selector.value]}"),
            mo.ui.table(seed_rows, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    robustness_url = (
        "https://raw.githubusercontent.com/alphaXiv/"
        "pats-policy-aware-training-scaffolding-for-agent/main/"
        "reports/pats-alfworld/images/scale_robustness.png"
    )
    mo.md(
        f"""
        ## Robustness and limits

        ![Result across generic, short trace-card, and primary campaigns]({robustness_url})

        This is a deliberately small claim test, not a reproduction of the paper's
        full 150-step campaign or its WebShop and search experiments. The public
        expert trace is stronger and more oracle-like than the paper's learned
        experience bank, while the candidate-action policy is narrower than free
        text generation. Eight seeds and 12 held-out tasks expose large binomial and
        training variance, so effect sizes should be read as evidence for this
        reconstruction only.
        """
    )
    return


@app.cell
def _(mo):
    threshold_url = (
        "https://raw.githubusercontent.com/alphaXiv/"
        "pats-policy-aware-training-scaffolding-for-agent/main/"
        "reports/pats-alfworld/images/threshold_sensitivity.png"
    )
    mo.md(
        f"""
        ## Controller-threshold sensitivity

        ![Scaffold-free success and final active cards by removal threshold]({threshold_url})

        The pre-specified 0.25 threshold did not remove a card at the first eligible
        checkpoint, so a transparent sensitivity grid was queued. Threshold 0 is a
        controlled scheduled-removal diagnostic; 0.10–0.20 remain
        competence-triggered variants. These results diagnose whether conclusions
        depend on an arbitrary reconstructed controller setting.
        """
    )
    return


@app.cell
def _(mo, results):
    _campaigns = [
        ("0.00", "threshold_0.00"),
        ("0.10", "threshold_0.10"),
        ("0.15", "threshold_0.15"),
        ("0.20", "threshold_0.20"),
        ("0.25", "primary"),
    ]
    _threshold_rows = []
    for _threshold, _campaign in _campaigns:
        _item = results["aggregates"][_campaign]["adaptive"]
        _threshold_rows.append(
            {
                "Removal threshold": _threshold,
                "Scaffold-free success": (
                    f"{100 * _item['heldout_scaffold_free_success']['mean']:.2f}%"
                ),
                "Successes / episodes": (
                    f"{_item['success_count']} / {_item['evaluation_episodes']}"
                ),
                "Final active cards": f"{_item['final_active_cards']['mean']:.2f} / 6",
            }
        )
    mo.ui.table(_threshold_rows, selection=None)
    return


@app.cell
def _(mo, results, results_url):
    compute = results["compute"]
    mo.md(
        f"""
        ## Provenance

        - Evidence backend: **{compute['backend']}**
        - GPU: **{compute['gpu_model']}**
        - Peak concurrent GPUs: **{compute['peak_gpu_count']}**
        - Actual elapsed campaign time: **{compute['wall_hours']:.6f} hours**
        - Frozen evidence JSON: [{results_url}]({results_url})
        - Paper: [PATS, arXiv 2607.21419](https://arxiv.org/abs/2607.21419)

        Run IDs, task-subset hashes, individual wall times, and terminal metrics are
        preserved in the frozen JSON. The exact command on every formal branch was
        `bash scripts/run.sh`, submitted with the Kubernetes backend.
        """
    )
    return


if __name__ == "__main__":
    app.run()
