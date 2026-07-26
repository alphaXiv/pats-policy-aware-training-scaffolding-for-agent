#!/usr/bin/env python3
"""Freeze terminal-log evidence and build the public PATS reproduction figures."""

from __future__ import annotations

import json
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "pats-alfworld"
IMAGES = OUT / "images"
DATA = OUT / "data"

RUNS = {
    "generic": {
        "none": [
            "683f2ab5-a8f8-4964-88f8-9c4b19de879e",
            "babbb048-2e62-42f8-ba55-843f1a31e411",
            "03a4f31d-ac9c-4a67-b888-0af2b75fa662",
            "e5a58c5c-74e6-4398-8de3-71c9df9917cc",
        ],
        "static": [
            "1bf09d5b-4fe0-4da7-8088-4815cc6ea04c",
            "7d627b80-b3ba-4983-b182-9b300f4a9eef",
            "69f00b26-3c1d-41cd-a0ba-6e8015dd263a",
            "47327bff-9fc5-4f18-a933-604be5134e4e",
        ],
        "adaptive": [
            "587a721a-26a1-451b-a5e0-5ede5ef7c229",
            "09488052-646d-4204-9d5b-c7ec83d3532c",
            "91f17e0e-3ef9-4337-a44b-cbcfa49cdb1e",
            "816ed759-fccd-4d79-9b90-cc34e79c105f",
        ],
        "adaptive_retain": [
            "b6e50f89-9b7d-4b0a-a850-cf8dc260070a",
            "229e1154-d4e6-4441-9696-15da72490528",
            "5b9977ed-5464-459f-ac6e-389a61f69dc1",
            "97ecbbe7-7014-43c0-8cea-8fdbd7001f91",
        ],
    },
    "trace_short": {
        "none": [
            "82b5daa9-1aba-4900-b131-e90cba29330f",
            "a7f05c7e-d3fa-4af0-8874-9cbb9233b803",
            "ad247eb9-372f-43b2-bf9a-140c61c96fa0",
            "67c6bc48-109e-4012-986f-dacf7bd50b92",
        ],
        "static": [
            "36e5760b-382c-4eff-9108-08ee3426d052",
            "1fe14bc4-f750-4821-86fb-cc9ff271c9dd",
            "d9e7ad6a-8be3-40b3-a518-876fd455ff0a",
            "e8066424-f0ad-4825-9a8c-dd6070403906",
        ],
        "adaptive": [
            "c85cfea1-5d48-4b85-8b87-9bc674bbb6f3",
            "e76394b2-ccc0-4c8c-8c89-f5f5e8741180",
            "f8f78e43-7378-4d29-a4f7-3334e17bf0a1",
            "c65702a6-ce28-4b90-bc89-dfc272279fec",
        ],
        "adaptive_retain": [
            "ba327d7c-0f19-48cd-b24f-e39ef76d961b",
            "3a4edc06-4c5a-4f7b-a1c6-4efd9eb5ba03",
            "ce95af02-e217-43a3-bc6c-83ab84eeaa7b",
            "55cb1a27-9f29-41ac-83e0-a3d8516f3d55",
        ],
    },
    "primary": {
        "none": [
            "e7020c16-8cbf-4da6-820c-d46a201fa0cf",
            "651e1af2-ad2d-4658-aab1-7f6c075053bf",
            "f66e52ba-0225-4e61-8093-cb48de37dbd9",
            "764c75b4-278c-40ca-bdad-75fee52670a8",
        ],
        "static": [
            "0bd69902-17bf-4528-a21a-ede75e6c9959",
            "3cc154cd-a773-4e03-adb8-0351366b0127",
            "cc484f2b-5161-42ed-841c-9148179feefd",
            "8801e7f8-7555-4fe8-8ca3-982d5cc507a6",
        ],
        "adaptive": [
            "079b763b-50a5-43b9-830e-f6ac4aa4bfb3",
            "f8b1f161-43c9-404e-8f3b-c3583b798d48",
            "2e0fb7f8-293a-41dd-be1f-a10ca86357af",
            "095d71a3-2252-4847-a254-9a3767185734",
        ],
        "adaptive_retain": [
            "554407bf-4f38-4d03-97a2-97f3ab18df22",
            "4c5224a5-99bf-46c3-bfe8-04b00b336d02",
            "0f1d2aad-bc2c-4678-97d1-846e8988c171",
            "68364acb-3481-4ae3-a335-1c11ad1f2651",
        ],
    },
    "threshold_0.00": {
        "adaptive": [
            "c16dda8f-afce-424b-93a7-adb268cdadcc",
            "6f1b3f6c-2477-4ecb-8185-1bfd39a7c8ed",
            "356d1d18-0a1a-425e-b5de-be05fe8c1627",
            "efa73521-3d91-4063-8e5a-87c9092efa72",
        ],
    },
    "threshold_0.10": {
        "adaptive": [
            "dae05733-c095-403f-8bdc-5cf082b970dc",
            "524b0c27-4f61-4004-98c2-d920c46b9289",
            "5b19e663-e00d-4980-a1fc-48aa9e3713d5",
            "9b97e43f-7ceb-4b6b-8dae-992bebfa7694",
        ],
    },
    "threshold_0.15": {
        "adaptive": [
            "e48725c0-76b2-4c64-8cf9-015d6cc4973b",
            "af9caf04-a20f-43b1-b4a3-a04b5e771cfb",
            "3998b2f7-a981-4696-a358-e335734928be",
            "4f065d30-f277-4041-9acf-39121ffe0c29",
        ],
    },
    "threshold_0.20": {
        "adaptive": [
            "3c8045f2-9b21-4022-a2a5-0a66cf662938",
            "827e9308-1592-4f3e-93dd-592acc1c7104",
            "9f511036-8a7a-4ab8-a7b0-dcadc105bb93",
            "6a113155-9db9-4750-91cd-9bdfb22f77c0",
        ],
    },
}

LABELS = {
    "none": "No scaffold",
    "static": "Static cards",
    "adaptive": "Adaptive removal",
    "adaptive_retain": "Cards retained",
}
COLORS = {
    "none": "#65758B",
    "static": "#E49A35",
    "adaptive": "#1D8A74",
    "adaptive_retain": "#8A5BA5",
}
ORDER = ["none", "static", "adaptive", "adaptive_retain"]


def terminal_metrics(run_id: str) -> dict:
    proc = subprocess.run(
        ["orx", "logs", run_id, "--bytes", "1000000"],
        check=True,
        capture_output=True,
        text=True,
    )
    matches = []
    for line in proc.stdout.splitlines():
        if '"event": "FINAL_METRICS"' not in line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("event") == "FINAL_METRICS":
            matches.append(item)
    if len(matches) != 1:
        raise RuntimeError(f"{run_id}: expected one FINAL_METRICS line, found {len(matches)}")
    metrics = matches[0]
    if metrics.get("backend") != "kubernetes":
        raise RuntimeError(f"{run_id}: evidence was not produced on Kubernetes")
    return metrics


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate(records: list[dict]) -> dict:
    fields = [
        "heldout_scaffold_free_success",
        "heldout_supported_success",
        "heldout_scaffold_free_mean_tokens",
        "heldout_supported_mean_tokens",
        "initial_nll_gap",
        "final_nll_gap",
        "relative_nll_gap_reduction",
        "final_active_cards",
        "wall_hours",
    ]
    result = {}
    for field in fields:
        values = [float(record[field]) for record in records]
        result[field] = {
            "mean": statistics.mean(values),
            "sd": sample_sd(values),
            "values": values,
        }
    initial_abs = [abs(float(record["initial_nll_gap"])) for record in records]
    final_abs = [abs(float(record["final_nll_gap"])) for record in records]
    per_seed_abs_reduction = [
        1.0 - final / initial for initial, final in zip(initial_abs, final_abs)
    ]
    result["absolute_nll_gap"] = {
        "initial_mean": statistics.mean(initial_abs),
        "final_mean": statistics.mean(final_abs),
        "mean_reduction": statistics.mean(per_seed_abs_reduction),
        "pooled_reduction": 1.0 - statistics.mean(final_abs) / statistics.mean(initial_abs),
        "per_seed_reduction": per_seed_abs_reduction,
    }
    result["rollout_diversity"] = {
        "unique_sequences_per_group_mean": statistics.mean(
            point["unique_sequences_per_group"]
            for record in records
            for point in record["training_dynamics"]
        ),
        "duplication_rate_mean": statistics.mean(
            point["duplication_rate"]
            for record in records
            for point in record["training_dynamics"]
        ),
    }
    result["success_count"] = round(
        sum(float(record["heldout_scaffold_free_success"]) for record in records) * 12
    )
    result["evaluation_episodes"] = 12 * len(records)
    return result


def style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D8DEE8", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def save_figure(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(IMAGES / name, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def headline(primary: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(ORDER))
    values = [
        100 * statistics.mean(r["heldout_scaffold_free_success"] for r in primary[c])
        for c in ORDER
    ]
    errors = [
        100 * sample_sd([r["heldout_scaffold_free_success"] for r in primary[c]])
        for c in ORDER
    ]
    ax.bar(
        x,
        values,
        yerr=errors,
        capsize=5,
        color=[COLORS[c] for c in ORDER],
        width=0.66,
        edgecolor="white",
    )
    for idx, condition in enumerate(ORDER):
        seeds = [100 * r["heldout_scaffold_free_success"] for r in primary[condition]]
        offsets = np.linspace(-0.18, 0.18, len(seeds))
        ax.scatter(
            idx + offsets,
            seeds,
            s=34,
            color="#172033",
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
    ax.set_xticks(x, [LABELS[c] for c in ORDER])
    ax.set_ylabel("Held-out success without cards (%)")
    ax.set_title("Primary result: scaffold-free ALFWorld success after matched training")
    ax.set_ylim(0, max(value + error for value, error in zip(values, errors)) * 1.16)
    ax.text(
        0.01,
        0.97,
        "Bars: mean across 4 seeds · dots: seed results · whiskers: seed SD · n=12 tasks/seed",
        transform=ax.transAxes,
        va="top",
        color="#4A5568",
        fontsize=9,
    )
    style_axes(ax)
    save_figure(fig, "headline_success.png")


def dynamics(primary: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for condition in ORDER:
        by_step = defaultdict(list)
        for record in primary[condition]:
            for point in record["training_dynamics"]:
                by_step[int(point["step"])].append(float(point["train_success"]))
        steps = sorted(by_step)
        means = [100 * statistics.mean(by_step[step]) for step in steps]
        ax.plot(
            steps,
            means,
            marker="o",
            linewidth=2.2,
            markersize=4.5,
            color=COLORS[condition],
            label=LABELS[condition],
        )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Environment success in grouped rollouts (%)")
    ax.set_title("Reward dynamics under matched environmental-reward training")
    ax.legend(frameon=False, ncol=2)
    style_axes(ax)
    save_figure(fig, "training_success.png")


def likelihood(primary: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for condition in ORDER:
        by_step = defaultdict(list)
        for record in primary[condition]:
            for point in record["checkpoints"]:
                by_step[int(point["step"])].append(float(point["nll_gap"]))
        steps = sorted(by_step)
        means = [statistics.mean(by_step[step]) for step in steps]
        ax.plot(
            steps,
            means,
            marker="o",
            linewidth=2.2,
            color=COLORS[condition],
            label=LABELS[condition],
        )
    ax.axhline(0, color="#172033", linewidth=1, alpha=0.65)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Unsupported NLL − supported NLL")
    ax.set_title("Teacher-forced support gap across checkpoints")
    ax.legend(frameon=False, ncol=2)
    style_axes(ax)
    save_figure(fig, "likelihood_gap.png")


def controller(primary: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for condition in ["adaptive", "adaptive_retain"]:
        by_step = defaultdict(list)
        for record in primary[condition]:
            for point in record["training_dynamics"]:
                by_step[int(point["step"])].append(float(point["active_cards"]))
        steps = sorted(by_step)
        means = [statistics.mean(by_step[step]) for step in steps]
        ax.plot(
            steps,
            means,
            marker="o",
            linewidth=2.4,
            color=COLORS[condition],
            label=LABELS[condition],
        )
    ax.set_ylim(-0.2, 6.4)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Mean active task-family cards (of 6)")
    ax.set_title("Adaptive controller removal versus permanent retention")
    ax.legend(frameon=False)
    style_axes(ax)
    save_figure(fig, "active_cards.png")


def diversity(primary: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(ORDER))
    unique = []
    duplication = []
    for condition in ORDER:
        unique.append(
            statistics.mean(
                point["unique_sequences_per_group"]
                for record in primary[condition]
                for point in record["training_dynamics"]
            )
        )
        duplication.append(
            100
            * statistics.mean(
                point["duplication_rate"]
                for record in primary[condition]
                for point in record["training_dynamics"]
            )
        )
    bars = ax.bar(x, unique, color=[COLORS[c] for c in ORDER], width=0.66)
    ax.set_xticks(
        x,
        ["No\nscaffold", "Static\ncards", "Adaptive\nremoval", "Cards\nretained"],
    )
    ax.set_ylabel("Unique action sequences per group (maximum 8)")
    ax.set_title("Rollout diversity across all training checkpoints")
    for bar, dup in zip(bars, duplication):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.12,
            f"{dup:.1f}% duplicated",
            ha="center",
            fontsize=9,
            color="#4A5568",
        )
    ax.set_ylim(0, 8.8)
    style_axes(ax)
    save_figure(fig, "rollout_diversity.png")


def scale_check(all_records: dict[str, dict[str, list[dict]]]) -> None:
    campaigns = ["generic", "trace_short", "primary"]
    campaign_labels = ["Generic cards\n6×4×15", "Trace cards\n6×4×15", "Trace cards\n16×8×20"]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(campaigns))
    width = 0.19
    for idx, condition in enumerate(ORDER):
        values = [
            100
            * statistics.mean(
                record["heldout_scaffold_free_success"]
                for record in all_records[campaign][condition]
            )
            for campaign in campaigns
        ]
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            color=COLORS[condition],
            label=LABELS[condition],
        )
    ax.set_xticks(x, campaign_labels)
    ax.set_ylabel("Held-out scaffold-free success (%)")
    ax.set_title("Robustness to card strength and reduced training scale")
    ax.legend(frameon=False, ncol=2)
    style_axes(ax)
    save_figure(fig, "scale_robustness.png")


def threshold_sensitivity(all_records: dict[str, dict[str, list[dict]]]) -> None:
    thresholds = [0.00, 0.10, 0.15, 0.20, 0.25]
    campaign_keys = [
        "threshold_0.00",
        "threshold_0.10",
        "threshold_0.15",
        "threshold_0.20",
        "primary",
    ]
    success = []
    active = []
    for campaign in campaign_keys:
        condition_records = all_records[campaign]["adaptive"]
        success.append(
            100
            * statistics.mean(
                record["heldout_scaffold_free_success"]
                for record in condition_records
            )
        )
        active.append(
            statistics.mean(record["final_active_cards"] for record in condition_records)
        )
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(
        np.arange(len(thresholds)),
        success,
        width=0.62,
        color=COLORS["adaptive"],
        alpha=0.88,
        label="Scaffold-free success",
    )
    ax.set_xticks(np.arange(len(thresholds)), [f"{value:.2f}" for value in thresholds])
    ax.set_xlabel("Competence threshold for card removal")
    ax.set_ylabel("Held-out scaffold-free success (%)")
    ax2 = ax.twinx()
    ax2.plot(
        np.arange(len(thresholds)),
        active,
        color="#172033",
        marker="o",
        linewidth=2.2,
        label="Final active cards",
    )
    ax2.set_ylabel("Final active cards (of 6)")
    ax2.set_ylim(-0.2, 6.4)
    ax.set_title("Controller-threshold sensitivity")
    lines, labels = [], []
    for axis in (ax, ax2):
        axis_lines, axis_labels = axis.get_legend_handles_labels()
        lines.extend(axis_lines)
        labels.extend(axis_labels)
    ax.legend(lines, labels, frameon=False, loc="upper left")
    style_axes(ax)
    ax2.spines["top"].set_visible(False)
    save_figure(fig, "threshold_sensitivity.png")


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, list[dict]]] = {}
    provenance = []
    for campaign, conditions in RUNS.items():
        records[campaign] = {}
        for condition, run_ids in conditions.items():
            records[campaign][condition] = []
            for run_id in run_ids:
                metrics = terminal_metrics(run_id)
                if metrics["condition"] != condition:
                    raise RuntimeError(
                        f"{run_id}: expected {condition}, got {metrics['condition']}"
                    )
                records[campaign][condition].append(metrics)
                provenance.append(
                    {
                        "campaign": campaign,
                        "condition": condition,
                        "seed": metrics["seed"],
                        "run_id": run_id,
                        "subset_sha256": metrics["subset_sha256"],
                        "backend": metrics["backend"],
                        "gpu_model": metrics["gpu_model"],
                        "wall_hours": metrics["wall_hours"],
                    }
                )

    frozen = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "paper": {
            "id": "2607.21419",
            "alfworld_success_pats": 0.8071,
            "alfworld_success_grpo": 0.6786,
            "alfworld_success_static_warm_start": 0.7024,
            "controlled_unsupported_success_pats": 0.2297,
            "controlled_unsupported_success_no_band": 0.1766,
            "nll_gap_initial": 0.1170,
            "nll_gap_final": 0.0723,
        },
        "compute": {
            "backend": "kubernetes",
            "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
            "peak_gpu_count": 16,
            "wall_hours": 4.375616,
            "campaign_start_utc": "2026-07-26T14:18:14.602Z",
            "campaign_end_utc": "2026-07-26T18:40:46.818Z",
        },
        "runs": records,
        "aggregates": {
            campaign: {
                condition: aggregate(condition_records)
                for condition, condition_records in conditions.items()
            }
            for campaign, conditions in records.items()
        },
        "provenance": provenance,
    }
    (DATA / "results.json").write_text(json.dumps(frozen, indent=2) + "\n")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "figure.facecolor": "white",
        }
    )
    headline(records["primary"])
    dynamics(records["primary"])
    likelihood(records["primary"])
    controller(records["primary"])
    diversity(records["primary"])
    scale_check(records)
    threshold_sensitivity(records)
    print(DATA / "results.json")
    for image_path in sorted(IMAGES.glob("*.png")):
        print(image_path)


if __name__ == "__main__":
    main()
