#!/usr/bin/env python3
"""GUI tool to estimate resonator collision/crossing fractions and plot outcomes."""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from functools import lru_cache
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter


@dataclass
class Results:
    threshold: float
    collision_fraction_mc: float
    crossed_fraction_mc: float
    collision_fraction_analytic: float
    crossed_fraction_analytic: float


@dataclass
class Params:
    spacing: float
    mu_df: float
    alpha_df: float
    beta_df: float
    q: float
    widths: float
    num_resonators: int
    trials: int
    seed: int | None
    plot_trials: int
    plot_bins: int
    f0_hz: float
    plot_count: int
    plot_center_index: int


def resonator_frequency_hz(index: int, f_center_hz: float, spacing: float, center_index: int) -> float:
    return f_center_hz * ((1.0 + spacing) ** (index - center_index))


def generalized_normal_pdf(x: np.ndarray, mu: float, alpha: float, beta: float) -> np.ndarray:
    return np.asarray(stats.gennorm.pdf(x, beta, loc=mu, scale=alpha), dtype=float)


def generalized_normal_std(alpha: float, beta: float) -> float:
    return float(stats.gennorm.std(beta, loc=0.0, scale=alpha))


def alpha_from_sigma_beta(sigma: float, beta: float) -> float:
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    if beta <= 0.0:
        raise ValueError("beta must be > 0")
    return float(sigma * math.sqrt(math.gamma(1.0 / beta) / math.gamma(3.0 / beta)))


def gap_sigma_from_resonator_sigma(sigma_resonator: float) -> float:
    if sigma_resonator <= 0.0:
        raise ValueError("resonator sigma must be > 0")
    return float(math.sqrt(2.0) * sigma_resonator)


def resonator_sigma_from_gap_sigma(sigma_gap: float) -> float:
    if sigma_gap <= 0.0:
        raise ValueError("gap sigma must be > 0")
    return float(sigma_gap / math.sqrt(2.0))


def sample_generalized_normal_array(
    rng: np.random.Generator, mu: float, alpha: float, beta: float, size: int | tuple[int, ...]
) -> np.ndarray:
    return np.asarray(stats.gennorm.rvs(beta, loc=mu, scale=alpha, size=size, random_state=rng), dtype=float)


@lru_cache(maxsize=32)
def make_distribution_grid(alpha: float, beta: float, tail_prob: float = 1e-12, points: int = 8193) -> tuple[np.ndarray, np.ndarray]:
    if alpha <= 0.0:
        raise ValueError("alpha_df must be > 0")
    if beta <= 0.0:
        raise ValueError("beta_df must be > 0")
    if points < 5 or points % 2 == 0:
        raise ValueError("points must be an odd integer >= 5")
    support = alpha * ((-math.log(tail_prob)) ** (1.0 / beta))
    x = np.linspace(-support, support, points)
    pdf = generalized_normal_pdf(x, 0.0, alpha, beta)
    area = float(np.trapezoid(pdf, x))
    if area <= 0.0:
        raise ValueError("Invalid generalized-normal PDF area")
    pdf /= area
    return x, pdf


def cdf_from_pdf_grid(x: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    cdf = np.empty_like(x)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * dx)
    if cdf[-1] <= 0.0:
        raise ValueError("Invalid CDF grid")
    cdf /= cdf[-1]
    cdf[-1] = 1.0
    return cdf


@lru_cache(maxsize=32)
def diff_distribution_grid(alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, pdf = make_distribution_grid(alpha=alpha, beta=beta)
    dx = x[1] - x[0]
    y = np.linspace(x[0] + x[0], x[-1] + x[-1], 2 * len(x) - 1)
    diff_pdf = np.convolve(pdf, pdf, mode="full") * dx
    diff_cdf = cdf_from_pdf_grid(y, diff_pdf)
    return y, diff_pdf, diff_cdf


def analytic_fractions(spacing: float, alpha_df: float, beta_df: float, q_factor: float, widths: float) -> tuple[float, float, float]:
    diff_x, _, diff_cdf = diff_distribution_grid(alpha=alpha_df, beta=beta_df)
    threshold = widths / q_factor
    collision = float(np.interp(threshold - spacing, diff_x, diff_cdf, left=0.0, right=1.0))
    crossed = float(np.interp(-spacing, diff_x, diff_cdf, left=0.0, right=1.0))
    return threshold, collision, crossed


def monte_carlo_fractions(
    spacing: float,
    mu_df: float,
    alpha_df: float,
    beta_df: float,
    q_factor: float,
    widths: float,
    num_resonators: int,
    trials: int,
    seed: int | None,
) -> tuple[float, float, float]:
    if num_resonators < 2:
        raise ValueError("num_resonators must be at least 2")
    if trials < 1:
        raise ValueError("trials must be >= 1")

    rng = np.random.default_rng(seed)
    threshold = widths / q_factor
    pair_count = num_resonators - 1

    dfs = sample_generalized_normal_array(rng, mu_df, alpha_df, beta_df, size=(trials, num_resonators))
    final_spacings = spacing + (dfs[:, 1:] - dfs[:, :-1])
    total_pairs = final_spacings.size
    collide_hits = int(np.count_nonzero(final_spacings <= threshold))
    crossed_hits = int(np.count_nonzero(final_spacings <= 0.0))

    return threshold, collide_hits / total_pairs, crossed_hits / total_pairs


def selected_indices(num_resonators: int, plot_count: int, plot_center_index: int) -> tuple[list[int], int]:
    if plot_count < 1:
        raise ValueError("plot_count must be >= 1")

    center_index = plot_center_index if plot_center_index >= 0 else num_resonators // 2
    if center_index < 0 or center_index >= num_resonators:
        raise ValueError("plot_center_index out of range")

    half = plot_count // 2
    i0 = max(0, center_index - half)
    i1 = min(num_resonators, i0 + plot_count)
    i0 = max(0, i1 - plot_count)
    return list(range(i0, i1)), center_index


def expanded_plot_indices(num_resonators: int, visible_indices: list[int], offscreen_count: int = 2) -> list[int]:
    if offscreen_count < 0:
        raise ValueError("offscreen_count must be >= 0")
    if not visible_indices:
        raise ValueError("visible_indices must not be empty")

    i0 = max(0, visible_indices[0] - offscreen_count)
    i1 = min(num_resonators, visible_indices[-1] + offscreen_count + 1)
    return list(range(i0, i1))


def sample_final_frequencies(
    spacing: float,
    mu_df: float,
    alpha_df: float,
    beta_df: float,
    num_resonators: int,
    trials: int,
    seed: int | None,
    selected: list[int],
    f_center_hz: float,
    center_index: int,
) -> tuple[list[float], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    starts = [resonator_frequency_hz(i, f_center_hz, spacing, center_index) for i in range(num_resonators)]
    mu_hz = f_center_hz * mu_df
    alpha_hz = f_center_hz * alpha_df
    selected_arr = np.asarray(selected, dtype=int)
    base = np.asarray(starts, dtype=float)[selected_arr]
    drifts = sample_generalized_normal_array(rng, mu_hz, alpha_hz, beta_df, size=(trials, selected_arr.size))
    by_resonator = [np.asarray(base[j] + drifts[:, j], dtype=float) for j in range(selected_arr.size)]
    return np.ravel(base[np.newaxis, :] + drifts).tolist(), by_resonator


def theory_curve_binned(
    spacing: float,
    mu_df: float,
    alpha_df: float,
    beta_df: float,
    f_center_hz: float,
    center_index: int,
    edge_selected: list[int],
    plotted_selected: list[int],
    samples: list[float],
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu_hz = f_center_hz * mu_df
    alpha_hz = f_center_hz * alpha_df
    dist_x_hz, dist_pdf_hz = make_distribution_grid(alpha=alpha_hz, beta=beta_df)
    dist_cdf_hz = cdf_from_pdf_grid(dist_x_hz, dist_pdf_hz)
    edge_starts = np.asarray(
        [resonator_frequency_hz(i, f_center_hz, spacing, center_index) for i in edge_selected], dtype=float
    )
    plotted_starts = np.asarray(
        [resonator_frequency_hz(i, f_center_hz, spacing, center_index) for i in plotted_selected], dtype=float
    )
    # Keep x-range fixed to resonance centers: 3 fully-visible resonances plus
    # 2 half-visible resonances whose centers sit on the left/right plot edges.
    x_min = float(edge_starts[0])
    x_max = float(edge_starts[-1])

    bin_edges = np.histogram_bin_edges(samples, bins=bins, range=(x_min, x_max))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    theory_pdf_bins = np.zeros_like(bin_centers, dtype=float)
    for start_hz in plotted_starts:
        for i in range(bin_centers.size):
            p_hi = float(np.interp(bin_edges[i + 1] - start_hz - mu_hz, dist_x_hz, dist_cdf_hz, left=0.0, right=1.0))
            p_lo = float(np.interp(bin_edges[i] - start_hz - mu_hz, dist_x_hz, dist_cdf_hz, left=0.0, right=1.0))
            theory_pdf_bins[i] += p_hi - p_lo
    theory_pdf_bins /= float(len(plotted_selected))
    theory_pdf_bins /= bin_widths

    return bin_edges, bin_centers, theory_pdf_bins


def spacing_distribution(
    spacing: float,
    mu_df: float,
    alpha_df: float,
    beta_df: float,
    pair_count: int,
    trials: int,
    seed: int | None,
    bins: int,
    widths: float,
    q: float,
) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray]:
    if pair_count < 1:
        raise ValueError("Need at least two plotted resonators to show spacing distribution.")
    if trials < 1:
        raise ValueError("plot_trials must be >= 1")

    rng = np.random.default_rng(seed)
    n_samples = pair_count * trials
    deltas = sample_generalized_normal_array(rng, mu_df, alpha_df, beta_df, size=n_samples) - sample_generalized_normal_array(
        rng, mu_df, alpha_df, beta_df, size=n_samples
    )
    samples = (spacing + deltas).tolist()
    threshold = widths / q

    x_min = min(min(samples), threshold, 0.0)
    x_max = max(samples)
    pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    x_min -= pad
    x_max += pad

    bin_edges = np.histogram_bin_edges(samples, bins=bins, range=(x_min, x_max))
    x = np.linspace(x_min, x_max, 1200)
    diff_x, diff_pdf, _ = diff_distribution_grid(alpha=alpha_df, beta=beta_df)
    pdf = np.interp(x - spacing, diff_x, diff_pdf, left=0.0, right=0.0)
    return samples, bin_edges, x, pdf


def run_model(
    params: Params,
) -> tuple[
    Results,
    list[float],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    threshold_mc, collision_mc, crossed_mc = monte_carlo_fractions(
        spacing=params.spacing,
        mu_df=params.mu_df,
        alpha_df=params.alpha_df,
        beta_df=params.beta_df,
        q_factor=params.q,
        widths=params.widths,
        num_resonators=params.num_resonators,
        trials=params.trials,
        seed=params.seed,
    )
    threshold_an, collision_an, crossed_an = analytic_fractions(
        spacing=params.spacing,
        alpha_df=params.alpha_df,
        beta_df=params.beta_df,
        q_factor=params.q,
        widths=params.widths,
    )

    results = Results(
        threshold=threshold_mc,
        collision_fraction_mc=collision_mc,
        crossed_fraction_mc=crossed_mc,
        collision_fraction_analytic=collision_an,
        crossed_fraction_analytic=crossed_an,
    )

    fully_visible_selected, center_index = selected_indices(
        num_resonators=params.num_resonators,
        plot_count=params.plot_count,
        plot_center_index=params.plot_center_index,
    )
    edge_selected = expanded_plot_indices(params.num_resonators, fully_visible_selected, offscreen_count=1)
    plotted_selected = expanded_plot_indices(params.num_resonators, edge_selected, offscreen_count=1)

    samples, samples_by_resonator = sample_final_frequencies(
        spacing=params.spacing,
        mu_df=params.mu_df,
        alpha_df=params.alpha_df,
        beta_df=params.beta_df,
        num_resonators=params.num_resonators,
        trials=params.plot_trials,
        seed=params.seed + 1000 if params.seed is not None else None,
        selected=plotted_selected,
        f_center_hz=params.f0_hz,
        center_index=center_index,
    )

    bin_edges, bin_centers, theory_pdf_bins = theory_curve_binned(
        spacing=params.spacing,
        mu_df=params.mu_df,
        alpha_df=params.alpha_df,
        beta_df=params.beta_df,
        f_center_hz=params.f0_hz,
        center_index=center_index,
        edge_selected=edge_selected,
        plotted_selected=plotted_selected,
        samples=samples,
        bins=params.plot_bins,
    )

    spacing_samples, spacing_edges, spacing_x, spacing_pdf = spacing_distribution(
        spacing=params.spacing,
        mu_df=params.mu_df,
        alpha_df=params.alpha_df,
        beta_df=params.beta_df,
        pair_count=len(fully_visible_selected) - 1,
        trials=params.plot_trials,
        seed=params.seed + 2000 if params.seed is not None else None,
        bins=params.plot_bins,
        widths=params.widths,
        q=params.q,
    )

    return (
        results,
        samples,
        samples_by_resonator,
        bin_edges,
        bin_centers,
        theory_pdf_bins,
        spacing_samples,
        spacing_edges,
        spacing_x,
        spacing_pdf,
    )


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Resonator Drift Collision Model")
        self.root.geometry("1250x760")

        self.defaults = {
            "spacing": "0.0018",
            "alpha_df": "1.414213562e-4",
            "beta_df": "2.0",
            "sigma_df": "1e-4",
            "sigma_gap_df": "1.414213562e-4",
            "q": "1e5",
            "widths": "10",
            "num_resonators": "1000",
            "trials": "1000",
            "seed": "1",
            "plot_trials": "10000",
            "plot_bins": "180",
            "f0_hz": "1e9",
            "plot_count": "3",
            "plot_center_index": "-1",
        }
        self.entries: dict[str, ttk.Entry] = {}
        self.y_scale_var = tk.StringVar(value="linear")
        self._updating_scale = False

        self._build_layout()
        self._sync_sigma_from_alpha_beta()
        self._run_and_plot()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.root, padding=10)
        controls.grid(row=0, column=0, sticky="ns")

        plot_frame = ttk.Frame(self.root, padding=10)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.rowconfigure(1, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Inputs", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")

        labels = [
            ("spacing", "Relative spacing"),
            ("alpha_df", "Drift alpha"),
            ("beta_df", "Drift beta (2 = normal dist)"),
            ("sigma_df", "Drift sigma (resonator)"),
            ("sigma_gap_df", "Drift sigma (gap change)"),
            ("q", "Q"),
            ("widths", "Threshold widths"),
            ("num_resonators", "Num resonators"),
            ("trials", "MC trials"),
            ("seed", "Seed (blank=random)"),
            ("plot_trials", "Plot trials/res"),
            ("plot_bins", "Plot bins"),
            ("f0_hz", "Center freq (Hz)"),
            ("plot_count", "Plotted resonators"),
            ("plot_center_index", "Plot center index"),
        ]

        for r, (key, label) in enumerate(labels, start=1):
            ttk.Label(controls, text=label).grid(row=r, column=0, sticky="w", pady=2)
            entry = ttk.Entry(controls, width=18)
            entry.insert(0, self.defaults[key])
            entry.grid(row=r, column=1, sticky="ew", padx=(8, 0), pady=2)
            self.entries[key] = entry

        self.entries["alpha_df"].bind("<Return>", self._on_alpha_or_beta_changed)
        self.entries["alpha_df"].bind("<FocusOut>", self._on_alpha_or_beta_changed)
        self.entries["beta_df"].bind("<Return>", self._on_alpha_or_beta_changed)
        self.entries["beta_df"].bind("<FocusOut>", self._on_alpha_or_beta_changed)
        self.entries["sigma_df"].bind("<Return>", self._on_sigma_changed)
        self.entries["sigma_df"].bind("<FocusOut>", self._on_sigma_changed)
        self.entries["sigma_gap_df"].bind("<Return>", self._on_gap_sigma_changed)
        self.entries["sigma_gap_df"].bind("<FocusOut>", self._on_gap_sigma_changed)

        button_row = len(labels) + 1
        ttk.Button(controls, text="Run", command=self._run_and_plot).grid(row=button_row, column=0, sticky="ew", pady=(10, 4))
        ttk.Button(controls, text="Reset", command=self._reset_defaults).grid(row=button_row, column=1, sticky="ew", pady=(10, 4), padx=(8, 0))

        scale_row = button_row + 1
        ttk.Label(controls, text="Y-axis scale").grid(row=scale_row, column=0, sticky="w", pady=(8, 2))
        scale_frame = ttk.Frame(controls)
        scale_frame.grid(row=scale_row, column=1, sticky="w", padx=(8, 0), pady=(8, 2))
        ttk.Radiobutton(scale_frame, text="Linear", variable=self.y_scale_var, value="linear", command=self._run_and_plot).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Radiobutton(scale_frame, text="Log", variable=self.y_scale_var, value="log", command=self._run_and_plot).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )

        self.results_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.results_var, justify="left").grid(
            row=scale_row + 1, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )

        ttk.Label(plot_frame, text="Histogram + Theory Curve", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        self.figure = Figure(figsize=(8.5, 6.8), dpi=100)
        self.ax_freq = self.figure.add_subplot(211)
        self.ax_spacing = self.figure.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

    def _reset_defaults(self) -> None:
        for key, value in self.defaults.items():
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, value)
        self._sync_sigma_from_alpha_beta()

    def _set_entry_value(self, key: str, value: float) -> None:
        entry = self.entries[key]
        entry.delete(0, tk.END)
        entry.insert(0, f"{value:.12g}")

    def _sync_sigma_from_alpha_beta(self) -> None:
        if self._updating_scale:
            return
        try:
            alpha = float(self.entries["alpha_df"].get())
            beta = float(self.entries["beta_df"].get())
            if alpha <= 0.0 or beta <= 0.0:
                return
            sigma = generalized_normal_std(alpha, beta)
            self._updating_scale = True
            self._set_entry_value("sigma_df", sigma)
            self._set_entry_value("sigma_gap_df", gap_sigma_from_resonator_sigma(sigma))
        finally:
            self._updating_scale = False

    def _sync_alpha_from_sigma_beta(self) -> None:
        if self._updating_scale:
            return
        try:
            sigma = float(self.entries["sigma_df"].get())
            beta = float(self.entries["beta_df"].get())
            alpha = alpha_from_sigma_beta(sigma, beta)
            self._updating_scale = True
            self._set_entry_value("alpha_df", alpha)
            self._set_entry_value("sigma_gap_df", gap_sigma_from_resonator_sigma(sigma))
        finally:
            self._updating_scale = False

    def _sync_from_gap_sigma_beta(self) -> None:
        if self._updating_scale:
            return
        try:
            sigma_gap = float(self.entries["sigma_gap_df"].get())
            beta = float(self.entries["beta_df"].get())
            sigma = resonator_sigma_from_gap_sigma(sigma_gap)
            alpha = alpha_from_sigma_beta(sigma, beta)
            self._updating_scale = True
            self._set_entry_value("sigma_df", sigma)
            self._set_entry_value("alpha_df", alpha)
        finally:
            self._updating_scale = False

    def _on_alpha_or_beta_changed(self, _event: object | None = None) -> None:
        self._sync_sigma_from_alpha_beta()
        self._run_and_plot()

    def _on_sigma_changed(self, _event: object | None = None) -> None:
        self._sync_alpha_from_sigma_beta()
        self._run_and_plot()

    def _on_gap_sigma_changed(self, _event: object | None = None) -> None:
        self._sync_from_gap_sigma_beta()
        self._run_and_plot()

    def _read_params(self) -> Params:
        seed_text = self.entries["seed"].get().strip()
        seed_val = None if seed_text == "" else int(seed_text)

        params = Params(
            spacing=float(self.entries["spacing"].get()),
            mu_df=0.0,
            alpha_df=float(self.entries["alpha_df"].get()),
            beta_df=float(self.entries["beta_df"].get()),
            q=float(self.entries["q"].get()),
            widths=float(self.entries["widths"].get()),
            num_resonators=int(self.entries["num_resonators"].get()),
            trials=int(self.entries["trials"].get()),
            seed=seed_val,
            plot_trials=int(self.entries["plot_trials"].get()),
            plot_bins=int(self.entries["plot_bins"].get()),
            f0_hz=float(self.entries["f0_hz"].get()),
            plot_count=int(self.entries["plot_count"].get()),
            plot_center_index=int(self.entries["plot_center_index"].get()),
        )
        if params.alpha_df <= 0.0:
            raise ValueError("Drift alpha must be > 0")
        if params.beta_df <= 0.0:
            raise ValueError("Drift beta must be > 0")
        return params

    def _run_and_plot(self) -> None:
        try:
            params = self._read_params()
            (
                results,
                samples,
                samples_by_resonator,
                bin_edges,
                bin_centers,
                theory_pdf_bins,
                spacing_samples,
                spacing_edges,
                spacing_x,
                spacing_pdf,
            ) = run_model(params)
        except Exception as exc:
            messagebox.showerror("Input/Model Error", str(exc))
            return

        self.results_var.set(
            "\n".join(
                [
                    f"Collision threshold (rel): {results.threshold:.6g}",
                    "",
                    f"MC Results (out of {params.num_resonators} resonators)",
                    f"Within threshold: {results.collision_fraction_mc * params.num_resonators:.1f}",
                    f"Crossed:          {results.crossed_fraction_mc * params.num_resonators:.1f}",
                    "",
                    f"Theory Results (out of {params.num_resonators} resonators)",
                    f"Within threshold: {results.collision_fraction_analytic * params.num_resonators:.1f}",
                    f"Crossed:          {results.crossed_fraction_analytic * params.num_resonators:.1f}",
                ]
            )
        )

        threshold = params.widths / params.q

        self.ax_freq.clear()
        # Stacked per-resonator histogram: each resonator contributes area 1.
        datasets = [arr.tolist() for arr in samples_by_resonator]
        weights = [np.full(arr.size, 1.0 / params.plot_trials) for arr in samples_by_resonator]
        cmap = np.asarray(np.linspace(0.0, 1.0, max(len(datasets), 1)))
        hist_colors = [tuple(c) for c in plt.cm.rainbow(cmap)]
        self.ax_freq.hist(
            datasets,
            bins=bin_edges,
            density=False,
            weights=weights,
            stacked=True,
            alpha=0.75,
            color=hist_colors[: len(datasets)],
            edgecolor="black",
            linewidth=0.2,
            label=[f"Res {i+1}" for i in range(len(datasets))] if len(datasets) <= 12 else None,
        )
        top_bin_widths = np.asarray(bin_edges[1:] - bin_edges[:-1])
        n_top_res = max(1, int(len(samples) / max(1, params.plot_trials)))
        theory_counts_per_bin = np.asarray(theory_pdf_bins) * top_bin_widths * n_top_res
        top_bin_khz = float(top_bin_widths[0] / 1e3) if len(top_bin_widths) > 0 else 0.0
        self.ax_freq.plot(
            bin_centers,
            theory_counts_per_bin,
            linestyle="-",
            marker="None",
            linewidth=2.2,
            color="tab:red",
            label="Theory curve",
        )
        self.ax_freq.set_xlabel("Final frequency (GHz)")
        self.ax_freq.set_ylabel("# resonators per bin")
        self.ax_freq.set_title(
            f"Final Frequency Distribution (visible window, +2 off-screen resonators each side, bin = {top_bin_khz:.3g} kHz)"
        )
        self.ax_freq.grid(alpha=0.25)
        if len(datasets) <= 12:
            self.ax_freq.legend()
        self.ax_freq.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v / 1e9:.6f}"))
        self.ax_freq.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
        if self.y_scale_var.get() == "log":
            self.ax_freq.set_yscale("log", nonpositive="clip")
        else:
            self.ax_freq.set_yscale("linear")

        self.ax_spacing.clear()
        self.ax_spacing.hist(
            spacing_samples,
            bins=spacing_edges,
            density=False,
            weights=np.full(len(spacing_samples), params.num_resonators / len(spacing_samples)),
            alpha=0.35,
            color="0.7",
            label="Spacing histogram",
        )
        spacing_centers = 0.5 * (spacing_edges[:-1] + spacing_edges[1:])
        diff_x, _, diff_cdf = diff_distribution_grid(alpha=params.alpha_df, beta=params.beta_df)
        theory_counts = np.asarray([
            params.num_resonators
            * (
                float(np.interp(spacing_edges[i + 1] - params.spacing, diff_x, diff_cdf, left=0.0, right=1.0))
                - float(np.interp(spacing_edges[i] - params.spacing, diff_x, diff_cdf, left=0.0, right=1.0))
            )
            for i in range(len(spacing_centers))
        ])
        spacing_bin_width_rel = float(spacing_edges[1] - spacing_edges[0]) if len(spacing_edges) > 1 else 0.0
        spacing_bin_khz = spacing_bin_width_rel * params.f0_hz / 1e3
        self.ax_spacing.plot(spacing_centers, theory_counts, color="tab:red", linewidth=2.2, label="Theory curve")
        self.ax_spacing.axvspan(spacing_edges[0], threshold, color="tab:orange", alpha=0.15, label="Collision region")
        self.ax_spacing.axvline(threshold, color="tab:orange", linestyle="--", linewidth=1.8, label="Threshold")
        self.ax_spacing.axvline(0.0, color="black", linestyle=":", linewidth=1.6, label="Crossing boundary")
        self.ax_spacing.set_xlabel("Adjacent final spacing (relative units)")
        self.ax_spacing.set_ylabel("# resonators per bin")
        self.ax_spacing.set_title(
            f"Adjacent-Pair Spacing with Collision Threshold (sum = Num resonators, bin = {spacing_bin_khz:.3g} kHz)"
        )
        self.ax_spacing.grid(alpha=0.25)
        self.ax_spacing.legend()
        if self.y_scale_var.get() == "log":
            self.ax_spacing.set_yscale("log", nonpositive="clip")
        else:
            self.ax_spacing.set_yscale("linear")

        self.figure.tight_layout()
        self.canvas.draw_idle()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
