#!/usr/bin/env python3
"""Draft plotting utility for two diagnostic panels.

  results/figures/funnel.pdf        — verification funnel, both corpus branches
  results/figures/dissociation.pdf  — decision vs propagation by condition

These are working figures, not the final ones: the typeset figures in
the paper are produced by a separate set of plotting scripts held with the
paper sources, and this file is included to show how the underlying numbers
are assembled.

Inputs: the canary-branch counts come from data/canaries/survival_analysis.json,
a withheld corpus artifact not part of this release, so fig_funnel() cannot run
here. The natural-branch and E1 values are transcribed constants (provenance:
data/router_validation/router_validation_report.md and results/e1_analysis.md);
the four canary-funnel counts are checked by an assert, the E1 panel values are
not, so treat them as a snapshot rather than a live read.

Run:  python scripts/analysis/make_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "figures"

GRAY = "#9a9a9a"
DARK = "#3b3b3b"
ACCENT = "#b2182b"   # used only to pair C2 / C2-shuf

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def canary_funnel_counts() -> tuple[int, int, int, int]:
    data = json.loads(
        (REPO / "data" / "canaries" / "survival_analysis.json").read_text())
    n = len(data)
    survived = sum(1 for c in data.values() if c["survived"])
    routed = sum(1 for c in data.values() if "verifiable" in c["routings"])
    caught = sum(1 for c in data.values() if "disbelief" in c["verdicts"])
    return n, survived, routed, caught


def fig_funnel() -> None:
    n, survived, routed, caught = canary_funnel_counts()
    assert (n, survived, routed, caught) == (115, 69, 10, 1), \
        f"canary funnel drifted from frozen values: {(n, survived, routed, caught)}"

    # Natural branch (frozen): 4,575 unique claims; 1,141 routed verifiable;
    # belief 8.1% of those = 92 claims positively confirmed end-to-end.
    nat_stages = ["unique claims", "routed verifiable", "Belief verdicts"]
    nat_vals = [4575, 1141, 92]
    can_stages = ["planted", "survive extraction", "routed verifiable",
                  "Disbelief verdict"]
    can_vals = [n, survived, routed, caught]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.3, 2.35))
    for ax, stages, vals, title in (
            (ax1, nat_stages, nat_vals, "(a) Natural corpus (72 decks)"),
            (ax2, can_stages, can_vals, "(b) Canary cohort (40 decks)")):
        y = list(range(len(stages)))[::-1]
        ax.barh(y, vals, height=0.62, color=DARK)
        ax.set_yticks(y, stages)
        ax.tick_params(axis="y", length=0)
        for yi, v in zip(y, vals):
            ax.text(v + vals[0] * 0.025, yi, f"{v:,}", va="center",
                    ha="left", fontsize=7.5, color=DARK)
        ax.set_title(title, loc="left", fontsize=8, pad=4)
        ax.set_xlim(0, vals[0] * 1.18)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
    fig.tight_layout(h_pad=1.2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "funnel.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_dissociation() -> None:
    # Frozen E1 values (results/e1_analysis.md, seed 0, 2026-06-12).
    conds = ["C0", "C1", "C2", "C2-shuf", "C3"]
    pinvest = [0.964, 0.934, 0.170, 0.142, 0.452]
    prop = [9.6, 11.3, 9.6, 8.7, 8.7]
    ci_lo = [3.6, 5.8, 4.3, 3.4, 3.5]
    ci_hi = [16.4, 17.8, 15.7, 14.9, 14.8]
    human = [7.0, 11.3, 9.6, 7.8, 7.0]
    colors = [GRAY, GRAY, ACCENT, ACCENT, GRAY]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.3, 2.3))

    x = range(len(conds))
    ax1.bar(x, pinvest, width=0.62, color=colors)
    for xi, v in zip(x, pinvest):
        ax1.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
    ax1.set_xticks(list(x), conds)
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("mean $P$(invest)")
    ax1.set_title("(a) Decisions collapse under labels — real or shuffled",
                  loc="left")

    yerr = [[p - lo for p, lo in zip(prop, ci_lo)],
            [hi - p for p, hi in zip(prop, ci_hi)]]
    ax2.errorbar(list(x), prop, yerr=yerr, fmt="o", color=DARK,
                 capsize=2.5, markersize=4, lw=1,
                 label="raw (95 % CI)")
    ax2.scatter(list(x), human, marker="D", s=14, facecolors="none",
                edgecolors=ACCENT, lw=1, label="human-verified",
                zorder=3)
    ax2.set_xticks(list(x), conds)
    ax2.set_ylim(0, 20)
    ax2.set_ylabel("propagation (%)")
    ax2.set_title("(b) Falsehood propagation is flat (all contrasts null)",
                  loc="left")
    ax2.legend(frameon=False, fontsize=7, loc="upper right",
               handletextpad=0.4)

    fig.tight_layout(h_pad=1.4)
    fig.savefig(OUT / "dissociation.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_funnel()
    fig_dissociation()
    print(f"wrote {OUT / 'funnel.pdf'} and {OUT / 'dissociation.pdf'}")
