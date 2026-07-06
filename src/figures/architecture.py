"""Model architecture diagram: full stack + one expanded transformer block."""
from __future__ import annotations

from pathlib import Path

from ._style import (AQUA, BLUE, GREEN, INK, INK2, MUTED, ORANGE, PANEL, RED,
                     VIOLET, YELLOW, apply_base)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"


def vbox(ax, cx, cy, w, h, title, color, sub=None, fill=None, tsize=10):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 linewidth=1.8, edgecolor=color, facecolor=fill or (color + "1c")))
    dy = 0.12 if sub else 0
    ax.text(cx, cy + dy, title, ha="center", va="center", fontsize=tsize,
            fontweight="bold", color=INK)
    if sub:
        ax.text(cx, cy - 0.26, sub, ha="center", va="center", fontsize=7.8,
                color=INK2, family="monospace")


def uarrow(ax, x, y1, y2, color=INK2, lw=1.8):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>",
                 mutation_scale=13, lw=lw, color=color, shrinkA=0, shrinkB=0))


def main():
    apply_base()
    fig, ax = plt.subplots(figsize=(12.5, 9.2))
    ax.set_xlim(0, 17); ax.set_ylim(0, 15.2); ax.axis("off")
    ax.text(0.2, 14.8, "nanoGPT-Seis model — 113M decoder (Llama-style)",
            fontsize=15, fontweight="bold", color=INK)
    ax.text(0.2, 14.15, "d_model 768 · 16 layers · 12 query / 4 KV heads (GQA) · "
            "head_dim 64 · SwiGLU 2048 · vocab 16384 · ctx 4096",
            fontsize=9.5, color=INK2)

    # ---------------- left: full stack ----------------
    cx, w = 3.3, 4.6
    ys = {"in": 0.9, "emb": 2.6, "block": 6.2, "norm": 9.6, "head": 11.0, "out": 12.5}
    vbox(ax, cx, ys["in"], w, 0.9, "input token ids", MUTED, "(B, T)", fill=PANEL)
    vbox(ax, cx, ys["emb"], w, 1.0, "Token Embedding", BLUE, "16384 × 768")
    # repeated block (shadow for ×16)
    for off in (0.22, 0.11, 0.0):
        ax.add_patch(FancyBboxPatch((cx - w / 2 + off, 4.9 + off), w, 2.6,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     linewidth=1.8, edgecolor=VIOLET, facecolor=VIOLET + "1c"))
    ax.text(cx, 7.0, "Transformer Block", ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(cx, 6.55, "RMSNorm → GQA(+RoPE) → +", ha="center", fontsize=8, color=INK2)
    ax.text(cx, 6.2, "RMSNorm → SwiGLU → +", ha="center", fontsize=8, color=INK2)
    ax.text(cx, 5.55, "× 16 layers", ha="center", fontsize=9.5, fontweight="bold", color=VIOLET)
    vbox(ax, cx, ys["norm"], w, 0.95, "Final RMSNorm", AQUA)
    vbox(ax, cx, ys["head"], w, 1.0, "LM Head", BLUE, "768 × 16384")
    vbox(ax, cx, ys["out"], w, 0.9, "logits", MUTED, "(B, T, 16384)", fill=PANEL)

    uarrow(ax, cx, ys["in"] + 0.45, ys["emb"] - 0.5)
    uarrow(ax, cx, ys["emb"] + 0.5, 4.9)
    uarrow(ax, cx, 7.5 + 0.22, ys["norm"] - 0.48)
    uarrow(ax, cx, ys["norm"] + 0.48, ys["head"] - 0.5)
    uarrow(ax, cx, ys["head"] + 0.5, ys["out"] - 0.45)

    # weight tying annotation (embed <-> head)
    ax.add_patch(FancyArrowPatch((cx + w / 2, ys["emb"]), (cx + w / 2 + 1.0, ys["emb"]),
                 arrowstyle="-", lw=1.3, color=ORANGE))
    ax.add_patch(FancyArrowPatch((cx + w / 2 + 1.0, ys["emb"]), (cx + w / 2 + 1.0, ys["head"]),
                 arrowstyle="-", lw=1.3, color=ORANGE, linestyle=(0, (4, 3))))
    ax.add_patch(FancyArrowPatch((cx + w / 2 + 1.0, ys["head"]), (cx + w / 2, ys["head"]),
                 arrowstyle="-|>", mutation_scale=12, lw=1.3, color=ORANGE))
    ax.text(cx + w / 2 + 1.15, (ys["emb"] + ys["head"]) / 2, "weight\ntying",
            fontsize=8, color=ORANGE, va="center")

    # ---------------- right: expanded block ----------------
    bx, bw = 12.4, 4.9
    ax.text(bx, 13.0, "one Transformer Block (pre-norm, residual)",
            fontsize=11, fontweight="bold", color=VIOLET, ha="center")
    # dashed connector from the ×16 stack to the detail
    ax.add_patch(FancyArrowPatch((cx + w / 2 + 0.1, 7.2), (bx - bw / 2 - 0.1, 7.0),
                 arrowstyle="-|>", mutation_scale=12, lw=1.4, color=VIOLET,
                 linestyle=(0, (5, 3)), connectionstyle="arc3,rad=-0.15"))

    yb = {"in": 1.3, "n1": 2.9, "attn": 4.4, "a1": 5.9, "n2": 7.6, "mlp": 9.1, "a2": 10.6, "out": 12.1}
    vbox(ax, bx, yb["in"], bw, 0.8, "x", MUTED, "(B, T, 768)", fill=PANEL)
    vbox(ax, bx, yb["n1"], bw, 0.8, "RMSNorm", AQUA)
    vbox(ax, bx, yb["attn"], bw, 1.0, "Grouped-Query Attention", VIOLET, "RoPE on Q,K · Flash SDPA")
    vbox(ax, bx, yb["a1"], bw, 0.7, "⊕  add residual", ORANGE, fill=ORANGE + "10")
    vbox(ax, bx, yb["n2"], bw, 0.8, "RMSNorm", AQUA)
    vbox(ax, bx, yb["mlp"], bw, 1.0, "SwiGLU MLP", GREEN, "768 → 2048 → 768")
    vbox(ax, bx, yb["a2"], bw, 0.7, "⊕  add residual", ORANGE, fill=ORANGE + "10")
    vbox(ax, bx, yb["out"], bw, 0.8, "output", MUTED, "(B, T, 768)", fill=PANEL)
    order = ["in", "n1", "attn", "a1", "n2", "mlp", "a2", "out"]
    for a, b in zip(order[:-1], order[1:]):
        uarrow(ax, bx, yb[a] + 0.4, yb[b] - 0.4)
    # residual skip arcs (from input of sublayer around to the ⊕)
    for src, dst in [(yb["in"], yb["a1"]), (yb["a1"], yb["a2"])]:
        ax.add_patch(FancyArrowPatch((bx - bw / 2, src + 0.05), (bx - bw / 2, dst),
                     arrowstyle="-|>", mutation_scale=12, lw=1.6, color=ORANGE,
                     connectionstyle="arc3,rad=-0.45", shrinkA=0, shrinkB=0))
    ax.text(bx - bw / 2 - 1.35, (yb["in"] + yb["a2"]) / 2, "residual\nskips",
            fontsize=8, color=ORANGE, va="center", ha="center", rotation=90)

    fig.savefig(ASSETS / "architecture.png", dpi=150, bbox_inches="tight")
    print("saved assets/architecture.png")


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    main()
