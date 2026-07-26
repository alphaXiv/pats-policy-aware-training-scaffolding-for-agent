from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
TASK_FAMILIES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)

BASE_CARDS = {
    "pick_and_place_simple": (
        "Place one object: search likely receptacles, take the named object, "
        "go to the destination, then put the object in or on it."
    ),
    "look_at_obj_in_light": (
        "Examine under light: find and take the named object, go to the desk "
        "lamp, then use the desk lamp while holding the object."
    ),
    "pick_clean_then_place_in_recep": (
        "Clean before placing: take the named object, go to a sink basin, "
        "clean it with the sink basin, then carry it to the destination."
    ),
    "pick_heat_then_place_in_recep": (
        "Heat before placing: take the named object, put it in a microwave, "
        "heat it with the microwave, retrieve it, then place it at the destination."
    ),
    "pick_cool_then_place_in_recep": (
        "Cool before placing: take the named object, put it in a refrigerator, "
        "cool it with the refrigerator, retrieve it, then place it at the destination."
    ),
    "pick_two_obj_and_place": (
        "Place two objects: locate each matching object separately and move both "
        "to the requested destination; verify that both placements complete."
    ),
}


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def family_from_path(path: str) -> str:
    parent = Path(path).parent.parent.name
    for family in TASK_FAMILIES:
        if parent.startswith(family):
            return family
    joined = str(path)
    for family in TASK_FAMILIES:
        if family in joined:
            return family
    raise ValueError(f"Could not infer task family from {path}")


def load_env_config(max_steps: int) -> dict[str, Any]:
    with (ROOT / "configs" / "alfworld.yaml").open() as handle:
        config = yaml.safe_load(handle)
    config["dagger"]["training"]["max_nb_steps_per_episode"] = max_steps
    return config


def collect_fixed_tasks(
    config: dict[str, Any],
    split: str,
    per_family: int,
) -> list[str]:
    manager = AlfredTWEnv(config, train_eval=split)
    grouped: dict[str, list[str]] = defaultdict(list)
    for game_file in sorted(manager.game_files):
        grouped[family_from_path(game_file)].append(game_file)
    selected: list[str] = []
    for family in TASK_FAMILIES:
        if len(grouped[family]) < per_family:
            raise RuntimeError(f"Need {per_family} {family} tasks, found {len(grouped[family])}")
        selected.extend(grouped[family][:per_family])
    del manager
    return selected


def make_task_env(config: dict[str, Any], split: str, game_file: str):
    manager = AlfredTWEnv(config, train_eval=split)
    manager.game_files = [game_file]
    manager.num_games = 1
    return manager.init_env(batch_size=1)


def normalize_commands(
    commands: list[str],
    max_candidates: int,
    goal: str,
    evidence: str | None = None,
) -> list[str]:
    commands = sorted(set(str(x).strip() for x in commands if str(x).strip()))
    if len(commands) <= max_candidates:
        return commands
    words = {
        w
        for w in re.findall(r"[a-z]+", goal.lower())
        if len(w) > 3 and w not in {"your", "task", "then", "with", "into", "onto", "under"}
    }
    anchors = ("look", "inventory", "open", "close", "take", "put", "go to", "clean", "heat", "cool", "use")

    evidence_text = (evidence or "").lower()

    def priority(command: str) -> tuple[int, int, int, str]:
        lower = command.lower()
        evidence_match = int(lower in evidence_text)
        overlap = sum(word in lower for word in words)
        anchored = sum(lower.startswith(anchor) for anchor in anchors)
        return (-evidence_match, -overlap, -anchored, lower)

    kept = sorted(commands, key=priority)[:max_candidates]
    for essential in ("look", "inventory"):
        if essential in commands and essential not in kept:
            kept[-1] = essential
    return sorted(set(kept))


def clean_observation(text: str, limit: int = 1400) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[-limit:]


def build_prompt(
    goal: str,
    observation: str,
    history: list[tuple[str, str]],
    commands: list[str],
    card: str | None,
) -> str:
    experience = ""
    if card:
        experience = (
            "\nRetrieved Relevant Experience (temporary training support):\n"
            f"- {card}\n"
        )
    recent = "\n".join(
        f"Previous action: {action}\nEnvironment: {clean_observation(obs, 500)}"
        for action, obs in history[-2:]
    )
    return (
        "You are acting in the public ALFWorld text environment. Choose exactly one "
        "admissible command that makes progress. Do not explain.\n"
        f"Task and initial scene: {clean_observation(goal)}\n"
        f"{experience}"
        f"{recent}\n"
        f"Current observation: {clean_observation(observation)}\n"
        "Admissible commands:\n- "
        + "\n- ".join(commands)
        + "\nAnswer with one command.\nAction:"
    )


class CandidatePolicy:
    def __init__(self, config: dict[str, Any]):
        model_name = config["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        lora = LoraConfig(
            r=int(config["lora_rank"]),
            lora_alpha=int(config["lora_rank"]) * 2,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, lora)
        self.model.config.use_cache = False
        self.device = next(self.model.parameters()).device

    def _scores(self, prompt: str, candidates: list[str], grad: bool) -> torch.Tensor:
        prefix = prompt + " "
        texts = [prefix + command for command in candidates]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=True,
        ).to(self.device)
        prefix_ids = self.tokenizer(
            prefix,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            add_special_tokens=True,
        )["input_ids"]
        prefix_len = min(prefix_ids.shape[1], encoded["input_ids"].shape[1] - 1)
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            logits = self.model(**encoded).logits[:, :-1].float()
            targets = encoded["input_ids"][:, 1:]
            token_lp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            positions = torch.arange(token_lp.shape[1], device=self.device)[None, :]
            mask = (positions >= max(prefix_len - 1, 0)) & encoded["attention_mask"][:, 1:].bool()
            lengths = mask.sum(dim=1).clamp_min(1)
            return (token_lp * mask).sum(dim=1) / lengths

    def choose(
        self,
        prompt: str,
        candidates: list[str],
        temperature: float,
        generator: torch.Generator,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        self.model.eval()
        scores = self._scores(prompt, candidates, grad=False)
        probs = F.softmax(scores / max(temperature, 1e-4), dim=0)
        if deterministic:
            index = int(torch.argmax(probs).item())
        else:
            index = int(torch.multinomial(probs, 1, generator=generator).item())
        entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
        return index, float(scores[index].item()), entropy

    def update(self, records: list[dict[str, Any]], optimizer, max_records: int) -> float:
        if not records:
            return 0.0
        self.model.train()
        random.shuffle(records)
        records = records[:max_records]
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for record in records:
            scores = self._scores(record["prompt"], record["candidates"], grad=True)
            log_policy = F.log_softmax(scores, dim=0)[record["choice"]]
            loss = -float(record["advantage"]) * log_policy / len(records)
            loss.backward()
            total += float(loss.detach().item())
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        return total

    def action_nll(self, prompt: str, action: str) -> float:
        self.model.eval()
        return float(-self._scores(prompt, [action], grad=False)[0].item())


def extract_expert_action(info: dict[str, Any]) -> str | None:
    for key in ("extra.expert_plan", "expert_plan"):
        value = info.get(key)
        if value and value[0]:
            plan = value[0]
            if isinstance(plan, (list, tuple)) and plan:
                return str(plan[0])
            if isinstance(plan, str):
                return plan
    return None


def audited_expert_card(env, family: str, max_steps: int) -> tuple[str, bool, int]:
    _, info = env.reset()
    actions: list[str] = []
    won = False
    for _ in range(max_steps):
        action = extract_expert_action(info)
        if not action:
            break
        _, _, dones, info = env.step([action])
        actions.append(action)
        won = bool(info.get("won", [False])[0])
        if won or bool(dones[0]):
            break
    if won:
        trace = " → ".join(actions)
        card = (
            f"{BASE_CARDS[family]} Audited successful public-environment trace "
            f"for this training task: {trace}."
        )
        return card[:1500], True, len(actions)
    return BASE_CARDS[family], False, len(actions)


def rollout(
    policy: CandidatePolicy,
    env,
    family: str,
    card: str | None,
    max_steps: int,
    max_candidates: int,
    temperature: float,
    generator: torch.Generator,
    deterministic: bool = False,
    collect_records: bool = True,
) -> dict[str, Any]:
    obs, info = env.reset()
    goal = str(obs[0])
    observation = goal
    history: list[tuple[str, str]] = []
    actions: list[str] = []
    records: list[dict[str, Any]] = []
    entropies: list[float] = []
    prompt_tokens = 0
    invalid = 0
    won = False
    for _ in range(max_steps):
        commands = normalize_commands(
            list(info["admissible_commands"][0]), max_candidates, goal, card
        )
        prompt = build_prompt(goal, observation, history, commands, card)
        prompt_tokens += len(policy.tokenizer.encode(prompt, add_special_tokens=False))
        choice, _, entropy = policy.choose(
            prompt, commands, temperature, generator, deterministic=deterministic
        )
        action = commands[choice]
        next_obs, _, dones, next_info = env.step([action])
        next_text = str(next_obs[0])
        won = bool(next_info.get("won", [False])[0])
        invalid += int("nothing happens" in next_text.lower())
        if collect_records:
            records.append({"prompt": prompt, "candidates": commands, "choice": choice})
        actions.append(action)
        entropies.append(entropy)
        history.append((action, next_text))
        observation = next_text
        info = next_info
        if bool(dones[0]) or won:
            break
    return {
        "family": family,
        "reward": int(won),
        "actions": actions,
        "records": records,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "invalid": invalid,
        "steps": len(actions),
        "tokens": prompt_tokens + sum(len(policy.tokenizer.encode(a)) for a in actions),
    }


def evidence_revision(
    family: str,
    trajectories: list[dict[str, Any]],
    base_card: str | None = None,
) -> str:
    base = base_card or BASE_CARDS[family]
    successful = [trajectory for trajectory in trajectories if trajectory["reward"]]
    failed = [trajectory for trajectory in trajectories if not trajectory["reward"]]
    additions: list[str] = []
    if successful:
        shortest = min(successful, key=lambda x: len(x["actions"]))
        verbs = [action.split()[0] for action in shortest["actions"]]
        compact = " → ".join(dict.fromkeys(verbs))
        additions.append(f"Recent success used this action progression: {compact}.")
    if failed:
        repeated = Counter(action for item in failed for action in item["actions"]).most_common(1)
        if repeated:
            additions.append(
                f"Recent failures overused “{repeated[0][0]}”; change location or prerequisite after repeated feedback."
            )
    return " ".join([base, *additions])[:1500]


def diversity_summary(groups: list[list[dict[str, Any]]]) -> dict[str, float]:
    unique = []
    duplication = []
    reward_std = []
    entropy = []
    for group in groups:
        sequences = [tuple(item["actions"]) for item in group]
        count = len(set(sequences))
        unique.append(count)
        duplication.append(1.0 - count / max(len(group), 1))
        reward_std.append(float(np.std([item["reward"] for item in group])))
        entropy.extend(item["entropy"] for item in group)
    return {
        "unique_sequences_per_group": float(np.mean(unique)),
        "duplication_rate": float(np.mean(duplication)),
        "reward_std": float(np.mean(reward_std)),
        "action_entropy": float(np.mean(entropy)),
    }


def build_diagnostic_records(
    policy: CandidatePolicy,
    envs: list[Any],
    task_files: list[str],
    max_candidates: int,
) -> list[dict[str, Any]]:
    records = []
    for env, path in zip(envs, task_files):
        obs, info = env.reset()
        goal = str(obs[0])
        raw_commands = list(info["admissible_commands"][0])
        expert = extract_expert_action(info)
        commands = normalize_commands(raw_commands, max_candidates, goal)
        if expert and expert in raw_commands and expert not in commands:
            commands[-1] = expert
            commands = sorted(set(commands))
        if expert and expert in commands:
            records.append(
                {
                    "family": family_from_path(path),
                    "goal": goal,
                    "observation": goal,
                    "commands": commands,
                    "action": expert,
                }
            )
    if not records:
        raise RuntimeError("No fixed expert actions available for likelihood diagnostic")
    return records


def likelihood_gap(
    policy: CandidatePolicy,
    diagnostic: list[dict[str, Any]],
    cards: dict[str, str],
) -> dict[str, float]:
    free_nll = []
    supported_nll = []
    for item in diagnostic:
        prompt_free = build_prompt(
            item["goal"], item["observation"], [], item["commands"], None
        )
        prompt_supported = build_prompt(
            item["goal"],
            item["observation"],
            [],
            item["commands"],
            cards[item["family"]],
        )
        free_nll.append(policy.action_nll(prompt_free, item["action"]))
        supported_nll.append(policy.action_nll(prompt_supported, item["action"]))
    free = float(np.mean(free_nll))
    supported = float(np.mean(supported_nll))
    return {
        "unsupported_nll": free,
        "supported_nll": supported,
        "nll_gap": free - supported,
        "nll_examples": len(free_nll),
    }


def evaluate(
    policy: CandidatePolicy,
    config: dict[str, Any],
    task_files: list[str],
    split: str,
    cards: dict[str, str],
    with_cards: bool,
    generator: torch.Generator,
) -> dict[str, Any]:
    outcomes = []
    per_family: dict[str, list[int]] = defaultdict(list)
    for path in task_files:
        family = family_from_path(path)
        env = make_task_env(config, split, path)
        result = rollout(
            policy,
            env,
            family,
            cards[family] if with_cards else None,
            int(config["_run"]["max_env_steps"]),
            int(config["_run"]["max_candidates"]),
            0.4,
            generator,
            deterministic=False,
            collect_records=False,
        )
        env.close()
        outcomes.append(result)
        per_family[family].append(result["reward"])
    return {
        "success": float(np.mean([item["reward"] for item in outcomes])),
        "mean_tokens": float(np.mean([item["tokens"] for item in outcomes])),
        "mean_steps": float(np.mean([item["steps"] for item in outcomes])),
        "per_family": {key: float(np.mean(value)) for key, value in per_family.items()},
        "n": len(outcomes),
    }


def main() -> None:
    started = time.time()
    with (ROOT / "experiment_config.json").open() as handle:
        run_config = json.load(handle)
    if run_config.get("smoke"):
        run_config.update(
            iterations=1,
            group_size=2,
            max_env_steps=3,
            train_tasks_per_family=1,
            eval_tasks_per_family=1,
            max_update_decisions=4,
        )
    seed = int(run_config["seed"])
    seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-capable experiment started without CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    emit(
        "run_start",
        config=run_config,
        backend="kubernetes",
        gpu_model=gpu_name,
        visible_gpus=torch.cuda.device_count(),
        torch=torch.__version__,
    )
    rollout_horizon = int(run_config["max_env_steps"])
    environment_horizon = (
        max(50, rollout_horizon)
        if bool(run_config.get("expert_trace_cards", False))
        else rollout_horizon
    )
    env_config = load_env_config(environment_horizon)
    env_config["_run"] = run_config
    train_files = collect_fixed_tasks(
        env_config, "train", int(run_config["train_tasks_per_family"])
    )
    eval_files = collect_fixed_tasks(
        env_config,
        "eval_out_of_distribution",
        int(run_config["eval_tasks_per_family"]),
    )
    subset_hash = hashlib.sha256(
        "\n".join(train_files + eval_files).encode()
    ).hexdigest()[:16]
    emit(
        "task_subset",
        train=[str(Path(path).parent.parent.name) for path in train_files],
        eval=[str(Path(path).parent.parent.name) for path in eval_files],
        subset_sha256=subset_hash,
    )
    policy = CandidatePolicy(run_config)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.model.parameters() if parameter.requires_grad],
        lr=float(run_config["learning_rate"]),
    )
    train_envs = [
        make_task_env(env_config, "train", path) for path in train_files
    ]
    cards = dict(BASE_CARDS)
    if bool(run_config.get("expert_trace_cards", False)):
        for env, path in zip(train_envs, train_files):
            family = family_from_path(path)
            cards[family], won, trace_steps = audited_expert_card(
                env, family, max(50, int(run_config["max_env_steps"]) * 3)
            )
            emit(
                "audited_expert_card",
                family=family,
                trace_success=won,
                trace_steps=trace_steps,
                card_characters=len(cards[family]),
            )
    active = {family: run_config["condition"] != "none" for family in TASK_FAMILIES}
    competence = {family: 0.0 for family in TASK_FAMILIES}
    generator = torch.Generator(device=policy.device)
    generator.manual_seed(seed + 1701)
    diagnostic = build_diagnostic_records(
        policy, train_envs, train_files, int(run_config["max_candidates"])
    )
    initial_gap = likelihood_gap(policy, diagnostic, cards)
    emit("checkpoint", step=0, active_cards=sum(active.values()), **initial_gap)
    checkpoints = [{"step": 0, **initial_gap}]
    all_dynamics = []

    for iteration in range(1, int(run_config["iterations"]) + 1):
        iteration_groups = []
        update_records = []
        family_groups: dict[str, list[dict[str, Any]]] = {}
        for env, path in zip(train_envs, train_files):
            family = family_from_path(path)
            group = []
            for _ in range(int(run_config["group_size"])):
                result = rollout(
                    policy,
                    env,
                    family,
                    cards[family] if active[family] else None,
                    int(run_config["max_env_steps"]),
                    int(run_config["max_candidates"]),
                    float(run_config["temperature"]),
                    generator,
                )
                group.append(result)
            rewards = np.asarray([item["reward"] for item in group], dtype=np.float32)
            std = float(rewards.std())
            if std > 0:
                advantages = (rewards - rewards.mean()) / (std + 1e-4)
                for trajectory, advantage in zip(group, advantages):
                    for record in trajectory["records"]:
                        record["advantage"] = float(advantage)
                        update_records.append(record)
            family_groups[family] = group
            iteration_groups.append(group)

        loss = policy.update(
            update_records, optimizer, int(run_config["max_update_decisions"])
        )
        stats = diversity_summary(iteration_groups)
        mean_success = float(
            np.mean([item["reward"] for group in iteration_groups for item in group])
        )

        for family, group in family_groups.items():
            group_success = float(np.mean([item["reward"] for item in group]))
            competence[family] = 0.5 * competence[family] + 0.5 * group_success
            if run_config["condition"] in {"adaptive", "adaptive_retain"}:
                cards[family] = evidence_revision(family, group, cards[family])
            if (
                run_config["condition"] == "adaptive"
                and bool(run_config["adaptive_remove"])
                and active[family]
                and iteration >= int(run_config.get("adaptive_remove_after", 2))
                and competence[family]
                >= float(run_config.get("adaptive_threshold", 0.30))
            ):
                active[family] = False
                emit(
                    "card_removed",
                    step=iteration,
                    family=family,
                    competence=competence[family],
                )
        dynamics = {
            "step": iteration,
            "train_success": mean_success,
            "loss": loss,
            "mixed_groups": sum(
                0 < sum(item["reward"] for item in group) < len(group)
                for group in iteration_groups
            ),
            "active_cards": sum(active.values()),
            **stats,
        }
        all_dynamics.append(dynamics)
        emit("training_step", **dynamics)
        if iteration in {2, 4, 8, 12, int(run_config["iterations"])}:
            gap = likelihood_gap(policy, diagnostic, cards)
            checkpoints.append({"step": iteration, **gap})
            emit("checkpoint", step=iteration, active_cards=sum(active.values()), **gap)

    for env in train_envs:
        env.close()
    scaffold_free = evaluate(
        policy,
        env_config,
        eval_files,
        "eval_out_of_distribution",
        cards,
        False,
        generator,
    )
    supported = evaluate(
        policy,
        env_config,
        eval_files,
        "eval_out_of_distribution",
        cards,
        True,
        generator,
    )
    elapsed = (time.time() - started) / 3600.0
    final_gap = checkpoints[-1]["nll_gap"]
    initial_gap_value = checkpoints[0]["nll_gap"]
    gap_reduction = (
        (initial_gap_value - final_gap) / abs(initial_gap_value)
        if abs(initial_gap_value) > 1e-9
        else 0.0
    )
    summary = {
        "condition": run_config["condition"],
        "seed": seed,
        "backend": "kubernetes",
        "gpu_model": gpu_name,
        "gpu_count": 1,
        "wall_hours": elapsed,
        "subset_sha256": subset_hash,
        "heldout_scaffold_free_success": scaffold_free["success"],
        "heldout_supported_success": supported["success"],
        "heldout_scaffold_free_mean_tokens": scaffold_free["mean_tokens"],
        "heldout_supported_mean_tokens": supported["mean_tokens"],
        "initial_nll_gap": initial_gap_value,
        "final_nll_gap": final_gap,
        "relative_nll_gap_reduction": gap_reduction,
        "final_active_cards": sum(active.values()),
        "checkpoints": checkpoints,
        "training_dynamics": all_dynamics,
        "scaffold_free_eval": scaffold_free,
        "supported_eval": supported,
    }
    emit("FINAL_METRICS", **summary)
    print("PATS_REPRODUCTION_COMPLETE", flush=True)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
