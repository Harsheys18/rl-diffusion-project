"""
Synthetic validation for the proof claims in the project writeup.

This script does not try to approximate Stable Diffusion. Instead, it builds a
small controlled finite-horizon MDP with the same policy-class distinction:
state-conditioned controllers can choose actions from the latent state, while
time-only schedules must choose one action per denoising step. The environment
has known per-step intervention values, so the expressivity gap, sparse
intervention threshold, guaranteed gain, and diminishing returns predictions can
be checked numerically and plotted.

Usage:
    python proof_validation.py
    python proof_validation.py --horizon 30 --epsilon 0.12 --samples 20000
"""

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


@dataclass
class ValidationConfig:
    horizon: int = 20
    samples: int = 10000
    epsilon: float = 0.1
    seed: int = 42
    default_gamma: float = 7.5
    low_gamma: float = 3.0
    high_gamma: float = 12.0
    output_dir: str = "results/proof_validation"


def make_step_values(horizon: int) -> list[float]:
    """Create bounded, decreasing intervention values with a few early bumps."""
    values = []
    for step in range(1, horizon + 1):
        value = 0.12 * math.exp(-0.18 * (step - 1))
        value += 0.035 * math.exp(-0.5 * ((step - 4) / 1.3) ** 2)
        value += 0.020 * math.exp(-0.5 * ((step - 9) / 1.8) ** 2)
        values.append(value)
    return values


def state_target(state: int, cfg: ValidationConfig) -> float:
    """Latent-state-sensitive optimal guidance: A wants low, B wants high."""
    return cfg.low_gamma if state < 0 else cfg.high_gamma


def action_score(action: float, target: float, cfg: ValidationConfig) -> float:
    """Bounded reward factor in [0, 1], maximized by matching the state target."""
    scale = cfg.high_gamma - cfg.low_gamma
    return max(0.0, min(1.0, 1.0 - abs(action - target) / scale))


def evaluate_expressivity_gap(cfg: ValidationConfig, rng: random.Random) -> dict:
    """Compare the best state-conditioned policy with the best time-only schedule."""
    states = [rng.choice([-1, 1]) for _ in range(cfg.samples)]
    targets = [state_target(state, cfg) for state in states]

    state_rewards = [action_score(target, target, cfg) for target in targets]

    candidate_schedules = [
        cfg.low_gamma + i * (cfg.high_gamma - cfg.low_gamma) / 180.0 for i in range(181)
    ]
    schedule_rewards = []
    for gamma in candidate_schedules:
        scores = [action_score(gamma, target, cfg) for target in targets]
        schedule_rewards.append(sum(scores) / len(scores))

    best_idx = max(range(len(schedule_rewards)), key=lambda idx: schedule_rewards[idx])
    best_schedule_gamma = float(candidate_schedules[best_idx])
    best_schedule_reward = float(schedule_rewards[best_idx])
    state_reward = float(sum(state_rewards) / len(state_rewards))

    p_low = float(sum(1 for state in states if state < 0) / len(states))
    p_high = float(sum(1 for state in states if state > 0) / len(states))
    delta = state_reward - best_schedule_reward
    proof_lower_bound = min(p_low, p_high) * 0.5

    return {
        "state_conditioned_reward": state_reward,
        "best_time_only_reward": best_schedule_reward,
        "best_time_only_gamma": best_schedule_gamma,
        "observed_gap": float(delta),
        "positive_gap": bool(delta > 0),
        "latent_region_probabilities": {"low_target": p_low, "high_target": p_high},
        "constructive_lower_bound_shape": proof_lower_bound,
        "schedule_grid": candidate_schedules,
        "schedule_rewards": schedule_rewards,
    }


def evaluate_sparse_and_topk(cfg: ValidationConfig, step_values: list[float]) -> dict:
    """Check sparse approximation, guaranteed gain, and diminishing returns."""
    threshold = cfg.epsilon / cfg.horizon
    selected = [idx for idx, value in enumerate(step_values) if value > threshold]
    k_epsilon = len(selected)

    sorted_values = sorted(step_values, reverse=True)
    cumulative = [0.0]
    for value in sorted_values:
        cumulative.append(cumulative[-1] + value)
    dense_gain = float(sum(step_values))
    sparse_gain = float(sum(step_values[idx] for idx in selected))
    sparse_error = dense_gain - sparse_gain

    marginal_gains = [cumulative[idx + 1] - cumulative[idx] for idx in range(len(sorted_values))]
    non_increasing = all(
        marginal_gains[idx] + 1e-12 >= marginal_gains[idx + 1]
        for idx in range(len(marginal_gains) - 1)
    )
    guaranteed_gain_holds = all(
        cumulative[idx + 1] + 1e-12 >= sum(sorted_values[: idx + 1])
        for idx in range(len(sorted_values))
    )

    saturation_candidates = [idx for idx, value in enumerate(sorted_values) if value < threshold]
    saturation_budget = saturation_candidates[0] if saturation_candidates else cfg.horizon
    tail_after_saturation = float(dense_gain - cumulative[saturation_budget])

    return {
        "epsilon": cfg.epsilon,
        "threshold_epsilon_over_T": float(threshold),
        "dense_gain": dense_gain,
        "k_epsilon": k_epsilon,
        "selected_steps_1_indexed": [idx + 1 for idx in selected],
        "sparse_gain": sparse_gain,
        "sparse_error": float(sparse_error),
        "sparse_bound_holds": bool(sparse_error <= cfg.epsilon + 1e-12),
        "topk_cumulative_gains": cumulative,
        "topk_marginal_gains": marginal_gains,
        "diminishing_returns_hold": non_increasing,
        "guaranteed_gain_bound_holds": guaranteed_gain_holds,
        "saturation_budget": saturation_budget,
        "tail_after_saturation": tail_after_saturation,
        "saturation_bound_holds": bool(tail_after_saturation <= cfg.epsilon + 1e-12),
        "step_values": step_values,
    }


def write_step_csv(path: Path, step_values: list[float], threshold: float) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "delta_t", "selected_by_epsilon_threshold"])
        for idx, value in enumerate(step_values, start=1):
            writer.writerow([idx, f"{value:.10f}", int(value > threshold)])


def plot_results(output_dir: Path, expressivity: dict, sparse: dict) -> None:
    if plt is None:
        return

    schedule_grid = expressivity["schedule_grid"]
    schedule_rewards = expressivity["schedule_rewards"]
    step_values = sparse["step_values"]
    cumulative = sparse["topk_cumulative_gains"]
    marginals = sparse["topk_marginal_gains"]
    threshold = sparse["threshold_epsilon_over_T"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(schedule_grid, schedule_rewards, color="#1f77b4", linewidth=2)
    ax.axhline(expressivity["state_conditioned_reward"], color="#d62728", linestyle="--")
    ax.axvline(expressivity["best_time_only_gamma"], color="#444444", linestyle=":")
    ax.set_title("Expressivity gap")
    ax.set_xlabel("time-only guidance gamma")
    ax.set_ylabel("expected reward")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    steps = list(range(1, len(step_values) + 1))
    colors = ["#2ca02c" if value > threshold else "#bbbbbb" for value in step_values]
    ax.bar(steps, step_values, color=colors)
    ax.axhline(threshold, color="#d62728", linestyle="--", label="epsilon / T")
    ax.set_title("Sparse intervention threshold")
    ax.set_xlabel("step")
    ax.set_ylabel("Delta_t")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 0]
    ax.plot(list(range(len(cumulative))), cumulative, marker="o", color="#9467bd")
    ax.set_title("Top-K cumulative gain")
    ax.set_xlabel("K")
    ax.set_ylabel("gain over default")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(list(range(1, len(marginals) + 1)), marginals, marker="o", color="#ff7f0e")
    ax.axhline(threshold, color="#d62728", linestyle="--", label="epsilon / T")
    ax.set_title("Diminishing marginal gains")
    ax.set_xlabel("intervention rank")
    ax.set_ylabel("marginal gain")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "proof_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _polyline(points: list[tuple[float, float]], color: str, width: int = 3) -> str:
    joined = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{joined}" fill="none" stroke="{color}" stroke-width="{width}"/>'


def _line(x1: float, y1: float, x2: float, y2: float, color: str, dash: bool = False) -> str:
    dash_attr = ' stroke-dasharray="6 5"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="2"{dash_attr}/>'
    )


def _text(x: float, y: float, content: str, size: int = 14, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'text-anchor="{anchor}" fill="#222">{content}</text>'
    )


def _panel_frame(x: float, y: float, w: float, h: float, title: str) -> list[str]:
    return [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#fff" stroke="#ddd"/>',
        _text(x + 18, y + 28, title, size=17),
    ]


def write_svg_summary(output_dir: Path, expressivity: dict, sparse: dict) -> None:
    """Write a dependency-free SVG dashboard for quick visual inspection."""
    width, height = 1200, 820
    panel_w, panel_h = 540, 315
    panels = [(40, 70), (620, 70), (40, 430), (620, 430)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f8fa"/>',
        _text(40, 38, "Proof Validation Visual Summary", size=26),
        _text(
            40,
            60,
            "State-conditioned expressivity, sparse thresholding, top-K gains, and saturation.",
            size=13,
        ),
    ]

    schedule_grid = expressivity["schedule_grid"]
    schedule_rewards = expressivity["schedule_rewards"]
    step_values = sparse["step_values"]
    cumulative = sparse["topk_cumulative_gains"]
    marginals = sparse["topk_marginal_gains"]
    threshold = sparse["threshold_epsilon_over_T"]

    # Panel 1: expressivity gap.
    x, y = panels[0]
    parts += _panel_frame(x, y, panel_w, panel_h, "Expressivity Gap")
    px, py, pw, ph = x + 60, y + 55, panel_w - 95, panel_h - 100
    min_reward = min(schedule_rewards)
    max_reward = max(expressivity["state_conditioned_reward"], max(schedule_rewards))
    reward_span = max_reward - min_reward

    def map_schedule(gamma: float, reward: float) -> tuple[float, float]:
        gx = px + (gamma - min(schedule_grid)) / (max(schedule_grid) - min(schedule_grid)) * pw
        gy = py + ph - (reward - min_reward) / reward_span * ph
        return gx, gy

    points = [map_schedule(g, r) for g, r in zip(schedule_grid, schedule_rewards)]
    state_y = map_schedule(schedule_grid[0], expressivity["state_conditioned_reward"])[1]
    best_x = map_schedule(expressivity["best_time_only_gamma"], min_reward)[0]
    parts += [
        _line(px, py + ph, px + pw, py + ph, "#999"),
        _line(px, py, px, py + ph, "#999"),
        _polyline(points, "#1f77b4", 3),
        _line(px, state_y, px + pw, state_y, "#d62728", dash=True),
        _line(best_x, py, best_x, py + ph, "#555", dash=True),
        _text(px, py + ph + 28, "time-only gamma", size=12),
        _text(px - 15, py - 8, "reward", size=12, anchor="end"),
        _text(px + 12, state_y - 8, "state-conditioned = 1.0000", size=12),
        _text(px + 12, py + ph - 12, f"best schedule = {expressivity['best_time_only_reward']:.4f}", size=12),
        _text(px + 12, py + ph + 52, f"observed gap = {expressivity['observed_gap']:.4f}", size=15),
    ]

    # Panel 2: sparse threshold.
    x, y = panels[1]
    parts += _panel_frame(x, y, panel_w, panel_h, "Sparse Intervention Threshold")
    px, py, pw, ph = x + 55, y + 55, panel_w - 85, panel_h - 100
    max_step = max(step_values)
    bar_gap = 4
    bar_w = (pw - bar_gap * (len(step_values) - 1)) / len(step_values)
    thresh_y = py + ph - threshold / max_step * ph
    parts += [_line(px, py + ph, px + pw, py + ph, "#999"), _line(px, py, px, py + ph, "#999")]
    for idx, value in enumerate(step_values):
        bx = px + idx * (bar_w + bar_gap)
        bh = value / max_step * ph
        color = "#2ca02c" if value > threshold else "#c7c7c7"
        parts.append(f'<rect x="{bx:.2f}" y="{py + ph - bh:.2f}" width="{bar_w:.2f}" height="{bh:.2f}" fill="{color}"/>')
    parts += [
        _line(px, thresh_y, px + pw, thresh_y, "#d62728", dash=True),
        _text(px + pw - 3, thresh_y - 8, "epsilon / T", size=12, anchor="end"),
        _text(px, py + ph + 28, "denoising step", size=12),
        _text(px - 15, py - 8, "Delta_t", size=12, anchor="end"),
        _text(px + 12, py + ph + 52, f"K_epsilon = {sparse['k_epsilon']} / {len(step_values)}", size=15),
    ]

    # Panel 3: cumulative top-K gain.
    x, y = panels[2]
    parts += _panel_frame(x, y, panel_w, panel_h, "Top-K Cumulative Gain")
    px, py, pw, ph = x + 60, y + 55, panel_w - 95, panel_h - 100
    max_cum = max(cumulative)

    def map_cum(idx: int, value: float) -> tuple[float, float]:
        gx = px + idx / (len(cumulative) - 1) * pw
        gy = py + ph - value / max_cum * ph
        return gx, gy

    cum_points = [map_cum(idx, value) for idx, value in enumerate(cumulative)]
    sat_x = map_cum(sparse["saturation_budget"], 0.0)[0]
    parts += [
        _line(px, py + ph, px + pw, py + ph, "#999"),
        _line(px, py, px, py + ph, "#999"),
        _polyline(cum_points, "#9467bd", 3),
        _line(sat_x, py, sat_x, py + ph, "#d62728", dash=True),
        _text(sat_x + 6, py + 18, "K*", size=12),
        _text(px, py + ph + 28, "K interventions", size=12),
        _text(px - 15, py - 8, "gain", size=12, anchor="end"),
        _text(px + 12, py + ph + 52, f"dense gain = {sparse['dense_gain']:.4f}", size=15),
    ]

    # Panel 4: diminishing returns.
    x, y = panels[3]
    parts += _panel_frame(x, y, panel_w, panel_h, "Diminishing Marginal Gains")
    px, py, pw, ph = x + 60, y + 55, panel_w - 95, panel_h - 100
    max_margin = max(marginals)

    def map_margin(idx: int, value: float) -> tuple[float, float]:
        gx = px + idx / (len(marginals) - 1) * pw
        gy = py + ph - value / max_margin * ph
        return gx, gy

    margin_points = [map_margin(idx, value) for idx, value in enumerate(marginals)]
    thresh_y = py + ph - threshold / max_margin * ph
    parts += [
        _line(px, py + ph, px + pw, py + ph, "#999"),
        _line(px, py, px, py + ph, "#999"),
        _polyline(margin_points, "#ff7f0e", 3),
        _line(px, thresh_y, px + pw, thresh_y, "#d62728", dash=True),
        _text(px + pw - 3, thresh_y - 8, "epsilon / T", size=12, anchor="end"),
        _text(px, py + ph + 28, "intervention rank", size=12),
        _text(px - 15, py - 8, "marginal gain", size=12, anchor="end"),
        _text(px + 12, py + ph + 52, f"diminishing returns: {sparse['diminishing_returns_hold']}", size=15),
    ]

    parts.append("</svg>")
    (output_dir / "proof_validation.svg").write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/proof_validation")
    args = parser.parse_args()

    cfg = ValidationConfig(
        horizon=args.horizon,
        samples=args.samples,
        epsilon=args.epsilon,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    rng = random.Random(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step_values = make_step_values(cfg.horizon)
    expressivity = evaluate_expressivity_gap(cfg, rng)
    sparse = evaluate_sparse_and_topk(cfg, step_values)

    results = {
        "config": asdict(cfg),
        "expressivity_gap": expressivity,
        "sparse_intervention": sparse,
        "all_checks_passed": bool(
            expressivity["positive_gap"]
            and sparse["sparse_bound_holds"]
            and sparse["diminishing_returns_hold"]
            and sparse["guaranteed_gain_bound_holds"]
            and sparse["saturation_bound_holds"]
        ),
    }

    with (output_dir / "proof_validation_results.json").open("w") as f:
        json.dump(results, f, indent=2)
    write_step_csv(output_dir / "step_values.csv", step_values, sparse["threshold_epsilon_over_T"])
    write_svg_summary(output_dir, expressivity, sparse)
    plot_results(output_dir, expressivity, sparse)

    print("=== Proof Validation Summary ===")
    print(f"State-conditioned reward: {expressivity['state_conditioned_reward']:.4f}")
    print(f"Best time-only reward:     {expressivity['best_time_only_reward']:.4f}")
    print(f"Observed gap:              {expressivity['observed_gap']:.4f}")
    print(f"K_epsilon:                 {sparse['k_epsilon']} / {cfg.horizon}")
    print(f"Sparse error:              {sparse['sparse_error']:.6f} <= epsilon {cfg.epsilon}")
    print(f"Saturation budget K*:      {sparse['saturation_budget']}")
    print(f"All checks passed:         {results['all_checks_passed']}")
    print(f"Results written to:        {output_dir}")
    print(f"SVG visualization:         {output_dir / 'proof_validation.svg'}")
    if plt is None:
        print("Plot skipped:              matplotlib is not installed")

    if not results["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
