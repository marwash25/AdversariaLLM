#!/usr/bin/env python3
"""
Visualize per-pair alpha_zero ``cross`` vectors (1D triu ordering) as symmetric heatmaps
or rank plots.

1) In memory:

    sys.path.insert(0, "scripts")
    from visualize_hessian_upperbd import plot_precomputed_cross
    plot_precomputed_cross(normalized_cross_vals, raw=cross_vals, out_path="cross.png", style="heatmap")

2) From ``hessian_upperbd_at_zero(..., debug_save_cross="bounds.pt")``:

    python scripts/visualize_hessian_upperbd.py --precomputed-pt bounds.pt --out cross.png

3) Toy demo:

    PYTHONPATH=src python scripts/visualize_hessian_upperbd.py --out cross_vis.png
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

PlotStyle = Literal["heatmap", "rank", "both"]


def infer_nb_from_pair_count(num_pairs: int) -> int:
    """Recover ``nb`` from len(cross) = nb * (nb - 1) // 2."""
    disc = 1 + 8 * num_pairs
    root = math.isqrt(disc)
    if root * root != disc:
        raise ValueError(
            f"Tensor length {num_pairs} is not n*(n-1)/2 for integer n (expected unordered pairs)."
        )
    return (1 + root) // 2


def pairwise_upper_vec_to_symmetric_matrix(cross: Tensor) -> tuple[np.ndarray, int]:
    """Map triu pair ordering (same as ``torch.triu_indices(nb, nb, offset=1)``) to a symmetric matrix."""
    v = cross.detach().float().cpu().reshape(-1)
    nb = infer_nb_from_pair_count(int(v.numel()))
    upper = torch.triu_indices(nb, nb, offset=1, device="cpu")
    i = upper[0].numpy()
    j = upper[1].numpy()
    M = np.full((nb, nb), np.nan, dtype=np.float64)
    arr = v.numpy()
    M[i, j] = arr
    M[j, i] = arr
    return M, nb


def _heatmap_vmin_vmax(M: np.ndarray) -> tuple[float, float]:
    finite = M[np.isfinite(M)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    if lo < 0.0 < hi:
        lim = max(-lo, hi)
        return -lim, lim
    return lo, hi


def _plot_heatmap_ax(ax, M: np.ndarray, title: str, *, grid_b: Optional[int]) -> None:
    masked = np.ma.masked_invalid(M)
    vmin, vmax = _heatmap_vmin_vmax(M)
    im = ax.imshow(masked, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("flat index v2")
    ax.set_ylabel("flat index v1")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    nb = M.shape[0]
    if grid_b is not None and grid_b > 0:
        for t in range(grid_b, nb, grid_b):
            ax.axhline(t - 0.5, color="k", lw=0.35, alpha=0.4)
            ax.axvline(t - 0.5, color="k", lw=0.35, alpha=0.4)


def _plot_rank_values(ax, t: Tensor, *, color: str, title: str, ylabel: str) -> None:
    x = t.detach().float().cpu().numpy().ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    x.sort()
    ax.plot(np.arange(x.size), x, color=color, lw=1.2)
    ax.set_title(title)
    ax.set_xlabel("rank (sorted)")
    ax.set_ylabel(ylabel)
    ax.axhline(float(x.min()), color=color, ls="--", lw=0.8, alpha=0.6)


def plot_precomputed_cross(
    cross: Tensor,
    *,
    raw: Optional[Tensor] = None,
    out_path: Union[str, Path],
    suptitle: str = "",
    style: PlotStyle = "heatmap",
    grid_b: Optional[int] = None,
) -> Path:
    """Plot ``cross`` (and optional ``raw``). Flat indices follow row-major (i,j) in [n]×[b]."""
    out_path = Path(out_path)

    def _heatmaps() -> plt.Figure:
        Mc, nbc = pairwise_upper_vec_to_symmetric_matrix(cross)
        if raw is not None:
            Mr, nbr = pairwise_upper_vec_to_symmetric_matrix(raw)
            if nbr != nbc:
                raise ValueError(f"raw implies nb={nbr} but cross implies nb={nbc}")
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
            _plot_heatmap_ax(axes[0], Mr, "Raw: F(v1)+F(v2)−F({v1,v2})", grid_b=grid_b)
            _plot_heatmap_ax(axes[1], Mc, "cross (e.g. ÷ w[j1]w[j2])", grid_b=grid_b)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
            _plot_heatmap_ax(ax, Mc, "cross", grid_b=grid_b)
        return fig

    def _ranks() -> plt.Figure:
        if raw is not None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            _plot_rank_values(axes[0], raw, color="steelblue", title="Raw", ylabel="value")
            _plot_rank_values(axes[1], cross, color="seagreen", title="cross", ylabel="value")
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            _plot_rank_values(ax, cross, color="seagreen", title="cross", ylabel="value")
        return fig

    if style == "heatmap":
        fig = _heatmaps()
    elif style == "rank":
        fig = _ranks()
    else:
        Mc, _ = pairwise_upper_vec_to_symmetric_matrix(cross)
        if raw is not None:
            Mr, _ = pairwise_upper_vec_to_symmetric_matrix(raw)
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            _plot_heatmap_ax(axes[0, 0], Mr, "Raw (heatmap)", grid_b=grid_b)
            _plot_heatmap_ax(axes[0, 1], Mc, "cross (heatmap)", grid_b=grid_b)
            _plot_rank_values(axes[1, 0], raw, color="steelblue", title="Raw (rank)", ylabel="value")
            _plot_rank_values(axes[1, 1], cross, color="seagreen", title="cross (rank)", ylabel="value")
        else:
            fig, axes = plt.subplots(2, 1, figsize=(7, 8.5))
            _plot_heatmap_ax(axes[0], Mc, "cross (heatmap)", grid_b=grid_b)
            _plot_rank_values(axes[1], cross, color="seagreen", title="cross (rank)", ylabel="value")

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path.resolve()


def _load_precomputed(path: Path) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Return (normalized_cross_vals, cross_vals, meta) from a debug save dict."""
    obj: Any = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected torch.save dict in {path}, got {type(obj).__name__}")

    meta: dict[str, Any] = {}
    for key in ("n", "b", "time_taken"):
        v = obj.get(key)
        if isinstance(v, (int, float)):
            meta[key] = v

    normalized_cross_vals = obj.get("normalized_cross_vals")
    if not isinstance(normalized_cross_vals, Tensor):
        raise KeyError(f"{path} missing required tensor key 'normalized_cross_vals'")

    cross_vals = obj.get("cross_vals")
    if not isinstance(cross_vals, Tensor):
        raise KeyError(f"{path} missing required tensor key 'cross_vals'")

    hessian_upperbd = obj.get("hessian_upperbd")
    if isinstance(hessian_upperbd, Tensor):
        meta["hessian_upperbd"] = hessian_upperbd

    return normalized_cross_vals, cross_vals, meta


def _run_toy_demo(out: Path, n: int, k: int, style: PlotStyle) -> None:
    from llm_quick_check.attacks.submodular_utils.lattice_functions import CallableLatticeFunction
    from llm_quick_check.attacks.submodular_utils.setfn_reductions import BinarySubmodularSetFnReduction

    device = torch.device("cpu")
    assert k > 1 and (k & (k - 1)) == 0, "k must be a power of 2 for BinarySubmodularSetFnReduction"

    def F_batch(x: Tensor) -> tuple[Tensor, int]:
        vals = 0.01 * (x.sum(dim=1).float() ** 2)
        return vals, 0

    F = CallableLatticeFunction(k, n, F_batch)
    red = BinarySubmodularSetFnReduction(F, k, n, device)

    singleton_vals, _ = red.eval_singletons()
    pair_vals, rows, cols, _ = red.eval_all_pairs()
    flat_v1 = rows[0] * red.b + cols[0]
    flat_v2 = rows[1] * red.b + cols[1]
    j1, j2 = cols[0], cols[1]
    w = red.map.weights
    raw = singleton_vals[flat_v1] + singleton_vals[flat_v2] - pair_vals
    denom = (w[j1] * w[j2]).to(dtype=raw.dtype, device=raw.device)
    cross_n = raw / denom

    path = plot_precomputed_cross(
        cross_n,
        raw=raw,
        out_path=out,
        suptitle=f"Toy BinarySubmodularSetFnReduction n={n}, k={k}, b={red.b}, num_pairs={cross_n.numel()}",
        style=style,
        grid_b=red.b,
    )
    print(f"Wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot hessian_upperbd debug saves (heatmap or rank).")
    p.add_argument("--out", type=Path, default=ROOT / "hessian_upperbd_demo.png")
    p.add_argument("--precomputed-pt", type=Path, default=None, help="debug_save_cross .pt dict")
    p.add_argument("--suptitle", type=str, default="")
    p.add_argument("--style", choices=("heatmap", "rank", "both"), default="heatmap")
    p.add_argument("--n", type=int, default=3, help="(toy only) sequence positions")
    p.add_argument("--k", type=int, default=16, help="(toy only) vocab per slot; power of 2")
    args = p.parse_args()

    if args.precomputed_pt is not None:
        normalized_cross_vals, cross_vals, meta = _load_precomputed(args.precomputed_pt)
        grid_b = meta.get("b") if isinstance(meta.get("b"), int) else None
        path = plot_precomputed_cross(
            normalized_cross_vals,
            raw=cross_vals,
            out_path=args.out,
            suptitle=args.suptitle,
            style=args.style,
            grid_b=grid_b,
        )
        print(f"Wrote {path}")
        return

    _run_toy_demo(args.out, args.n, args.k, args.style)


if __name__ == "__main__":
    main()
