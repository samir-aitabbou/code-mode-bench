"""Publication-style figures (matplotlib, serif, Okabe-Ito palette).

Reads results/raw/*.jsonl and writes PNGs to results/figures/paper/.
Usage: uv run python scripts/paper_figures.py
"""
import json, re, math, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "paper"; OUT.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for f in sorted((ROOT / "results" / "raw").glob("*.jsonl")) for l in open(f) if l.strip()]
df = pd.DataFrame(rows)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    "hatch.linewidth": 0.6,
})
C_CL, C_CM = "#9a9a9a", "#0072B2"          # classic: grey, Code Mode: Okabe-Ito blue
MODELS = [("gpt-oss-20b", "gpt-oss-20b"), ("qwen3.5-9b", "Qwen3.5-9B")]
TASKS = ["T1", "T2", "T3", "T4", "T5", "T6"]

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0, 0)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, max(0, c-h), min(1, c+h)

def save(fig, name):
    fig.savefig(OUT / f"{name}.png"); plt.close(fig); print("ok", name)

main = df[(df.tool_count == 12) & (df.task_id.isin(TASKS))]

# ---------------------------------------------------------------- Fig. 2: success rates
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for ax, (mid, mname) in zip(axes, MODELS):
    x = np.arange(len(TASKS) + 1); w = 0.36
    for j, (mode, col, lab) in enumerate([("classic", C_CL, "Classic MCP"), ("codemode", C_CM, "Code Mode")]):
        ps, lo, hi = [], [], []
        for t in TASKS:
            s = main[(main.model_id == mid) & (main["mode"] == mode) & (main.task_id == t)].success
            p, l, h = wilson(int(s.sum()), len(s)); ps.append(p*100); lo.append((p-l)*100); hi.append((h-p)*100)
        if mode == "classic":
            s = df[(df.model_id == mid) & (df.task_id == "T4p")].success
            p, l, h = wilson(int(s.sum()), len(s)); ps.append(p*100); lo.append((p-l)*100); hi.append((h-p)*100)
            xs = x
        else:
            xs = x[:-1]
        xs = xs + (j - 0.5) * w
        ax.bar(xs, ps, w, color=col, edgecolor="black", linewidth=0.5, label=lab, zorder=2)
        ax.errorbar(xs, ps, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.8, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(TASKS + ["T4p"])
    ax.set_title(mname, loc="left", fontweight="bold")
    ax.set_ylim(0, 105); ax.yaxis.grid(True, linewidth=0.4, color="#dddddd", zorder=0); ax.set_axisbelow(True)
    ax.axvline(len(TASKS) - 0.5, color="black", linewidth=0.5, linestyle=":")
axes[0].set_ylabel("Success rate (%)")
axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, handlelength=1.2, columnspacing=1.2)
save(fig, "fig2-success")

# ---------------------------------------------------------------- Fig. 3: tokens and latency (medians, IQR)
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
for r, (mid, mname) in enumerate(MODELS):
    for c, (col_name, ylab, scale) in enumerate([("total_tokens", "Total tokens per trial (×1,000)", 1e3), ("latency_total_s", "Wall-clock time per trial (s)", 1)]):
        ax = axes[r, c]; x = np.arange(len(TASKS)); w = 0.36
        for j, (mode, colr, lab) in enumerate([("classic", C_CL, "Classic MCP"), ("codemode", C_CM, "Code Mode")]):
            med, lo, hi, failed = [], [], [], []
            for t in TASKS:
                s = main[(main.model_id == mid) & (main["mode"] == mode) & (main.task_id == t)]
                v = s[col_name] / scale; m = v.median(); med.append(m)
                lo.append(m - v.quantile(.25)); hi.append(v.quantile(.75) - m); failed.append(s.success.mean() < 0.5)
            xs = x + (j - 0.5) * w
            for xi, m, f in zip(xs, med, failed):
                ax.bar(xi, m, w, color="white" if f else colr, edgecolor=colr if f else "black",
                       hatch="////" if f else None, linewidth=0.6, zorder=2)
            ax.errorbar(xs, med, yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=0.6, capsize=1.8, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(TASKS); ax.set_ylabel(ylab)
        ax.yaxis.grid(True, linewidth=0.4, color="#dddddd", zorder=0); ax.set_axisbelow(True)
        ax.set_title(f"({'abcd'[r*2+c]}) {mname}", loc="left", fontweight="bold")
handles = [Rectangle((0,0),1,1,fc=C_CL,ec="black",lw=0.5), Rectangle((0,0),1,1,fc=C_CM,ec="black",lw=0.5),
           Rectangle((0,0),1,1,fc="white",ec="#555555",hatch="////",lw=0.6)]
fig.legend(handles, ["Classic MCP", "Code Mode", "Cell success rate < 50% (not comparable)"], loc="upper center",
           ncol=3, bbox_to_anchor=(0.5, 1.03), handlelength=1.4)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "fig3-efficiency")

# ---------------------------------------------------------------- Fig. 4: context limit on T4 / T4p
def req_size(detail):
    m = re.search(r"Input length \((\d+)\)", detail or "")
    return int(m.group(1)) if m else None
fig, ax = plt.subplots(figsize=(5.6, 2.9))
rng = np.random.default_rng(0)
cats = [("T4", "classic", "T4, classic\n(unpaginated tool)"), ("T4p", "classic", "T4p, classic\n(paginated tool)"), ("T4", "codemode", "T4, Code Mode")]
for i, (task, mode, lab) in enumerate(cats):
    for mid, mk, off in [("gpt-oss-20b", "o", -0.12), ("qwen3.5-9b", "s", 0.12)]:
        s = df[(df.model_id == mid) & (df.task_id == task) & (df["mode"] == mode) & (df.tool_count == 12)]
        colr = C_CM if mode == "codemode" else "#444444"
        if mode == "codemode":
            ys = (s.prompt_tokens / s.num_requests).values; fc = "white"
        else:
            o = s[s.failure_type == "context_overflow"]
            ys = np.array([req_size(d) or 32769 for d in o.detail]); fc = colr
            if mid == "qwen3.5-9b": fc = "white"
        xs = i + off + rng.uniform(-0.05, 0.05, len(ys))
        ax.scatter(xs, ys / 1e3, marker=mk, s=16, facecolors=fc, edgecolors=colr, linewidths=0.8, zorder=3)
ax.axhline(32.768, color="#D55E00", linestyle="--", linewidth=0.9)
ax.text(2.45, 33.6, "context limit (32,768)", color="#D55E00", ha="right", va="bottom", fontsize=8.5)
ax.set_xticks(range(3)); ax.set_xticklabels([c[2] for c in cats]); ax.set_xlim(-0.5, 2.5)
ax.set_ylabel("Tokens per request (×1,000)"); ax.set_ylim(0, 56)
ax.yaxis.grid(True, linewidth=0.4, color="#dddddd", zorder=0); ax.set_axisbelow(True)
leg = [Line2D([], [], marker="o", ls="", mfc="#444444", mec="#444444", ms=4.5, label="gpt-oss-20b: rejected request (exact size)"),
       Line2D([], [], marker="s", ls="", mfc="white", mec="#444444", ms=4.5, label="Qwen3.5-9B: rejected request (≥ 32,769, lower bound)"),
       Line2D([], [], marker="o", ls="", mfc="white", mec=C_CM, ms=4.5, label="Code Mode, gpt-oss-20b: mean tokens per request (all trials)"),
       Line2D([], [], marker="s", ls="", mfc="white", mec=C_CM, ms=4.5, label="Code Mode, Qwen3.5-9B: mean tokens per request (all trials)")]
ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(0, -0.22), ncol=1, handletextpad=0.3)
save(fig, "fig4-context")

# ---------------------------------------------------------------- Fig. 5: GPU time
gpu = df[df.task_id.isin(TASKS)].groupby(["model_id", "mode"]).gpu_seconds.sum() / 3600
fig, ax = plt.subplots(figsize=(4.2, 2.3))
y = np.arange(2); h = 0.34
for j, (mode, colr, lab) in enumerate([("classic", C_CL, "Classic MCP"), ("codemode", C_CM, "Code Mode")]):
    vals = [gpu[(m, mode)] for m, _ in MODELS]
    ax.barh(y + (j - 0.5) * h, vals, h, color=colr, edgecolor="black", linewidth=0.5, label=lab)
    for yi, v in zip(y + (j - 0.5) * h, vals): ax.text(v + 0.03, yi, f"{v:.2f} h", va="center", fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels([n for _, n in MODELS]); ax.invert_yaxis()
ax.set_xlabel("Total GPU time (hours), 70 trials per model and mode"); ax.set_xlim(0, 2.3)
ax.xaxis.grid(True, linewidth=0.4, color="#dddddd"); ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
save(fig, "fig5-gpu-time")

# ---------------------------------------------------------------- Fig. 6: tool bloat (slope chart)
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)
for ax, t in zip(axes, ["T3", "T5"]):
    labels_pos = []
    for mid, mname in MODELS:
        ls = "-" if mid == "gpt-oss-20b" else "--"
        for mode, colr, mk in [("classic", "#555555", "o"), ("codemode", C_CM, "s")]:
            v = [df[(df.model_id == mid) & (df["mode"] == mode) & (df.task_id == t) & (df.tool_count == n)].prompt_tokens.median() / 1e3 for n in (12, 60)]
            ax.plot([0, 1], v, ls=ls, color=colr, marker=mk, ms=4, mfc="white" if mid == "qwen3.5-9b" else colr, lw=0.9)
            labels_pos.append([v[1], f"{(v[1]/v[0]-1)*100:+.0f}% ({'gpt-oss' if mid=='gpt-oss-20b' else 'Qwen'})", colr])
    labels_pos.sort(key=lambda z: z[0]); last = -1e9
    for yv, txt, colr in labels_pos:
        yv = max(yv, last + 10); last = yv
        ax.text(1.06, yv, txt, va="center", fontsize=7.8, color=colr)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["12 tools", "60 tools"]); ax.set_xlim(-0.2, 1.7)
    ax.set_title(t, loc="left", fontweight="bold"); ax.yaxis.grid(True, linewidth=0.4, color="#dddddd"); ax.set_axisbelow(True)
axes[0].set_ylabel("Median input tokens (×1,000)")
leg = [Line2D([], [], color="#555555", marker="o", ms=4, label="Classic MCP"), Line2D([], [], color=C_CM, marker="s", ms=4, label="Code Mode"),
       Line2D([], [], color="black", ls="-", label="gpt-oss-20b (filled)"), Line2D([], [], color="black", ls="--", label="Qwen3.5-9B (open)")]
fig.legend(handles=leg, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06), handlelength=1.8)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "fig6-tool-bloat")

# ---------------------------------------------------------------- Fig. 7: failure breakdown
ft_order = [("wrong_answer", "Wrong answer", "#E69F00"), ("context_overflow", "Context overflow", "#D55E00"),
            ("tool_error", "Tool error", "#CC79A7"), ("max_turns", "Turn limit", "#F0E442"),
            ("provider_error", "Server error (HTTP 500)", "#56B4E9"), ("timeout", "Timeout (900 s)", "#999999")]
mainall = df[df.task_id.isin(TASKS)]
confs = [(m, mo) for m, _ in MODELS for mo in ("classic", "codemode")]
fig, ax = plt.subplots(figsize=(6.4, 2.2))
for i, (m, mo) in enumerate(confs):
    left = 0; s = mainall[(mainall.model_id == m) & (mainall["mode"] == mo)]
    for ft, lab, colr in ft_order:
        n = int((s.failure_type == ft).sum())
        if n:
            ax.barh(i, n, left=left, color=colr, edgecolor="black", linewidth=0.5)
            if n >= 3: ax.text(left + n/2, i, str(n), ha="center", va="center", fontsize=8)
            left += n
    ax.text(left + 0.6, i, f"{left}/{len(s)}" + (" (no failures)" if left == 0 else ""), va="center", fontsize=8.5)
names = {"gpt-oss-20b": "gpt-oss-20b", "qwen3.5-9b": "Qwen3.5-9B"}
ax.set_yticks(range(4)); ax.set_yticklabels([f"{names[m]}, {'classic' if mo=='classic' else 'Code Mode'}" for m, mo in confs])
ax.set_xlabel("Failed trials (out of 70 per configuration)"); ax.set_xlim(0, 45); ax.set_ylim(3.6, -0.6)
ax.xaxis.grid(True, linewidth=0.4, color="#dddddd"); ax.set_axisbelow(True)
ax.legend(handles=[Rectangle((0,0),1,1,fc=c,ec="black",lw=0.5) for _,_,c in ft_order], labels=[l for _,l,_ in ft_order],
          loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=3, handlelength=1.2)
save(fig, "fig7-failures")

# ---------------------------------------------------------------- Fig. 1: setup diagram (line art)
def B(ax, x, y, w, h, title, sub=None, ls="-", lw=0.8, fc="white", tsize=8.6, ssize=7.6, ec="black"):
    ax.add_patch(Rectangle((x, y), w, h, fill=True, fc=fc, ec=ec, lw=lw, ls=ls))
    if sub:
        ax.text(x + w/2, y + h*0.66, title, ha="center", va="center", fontsize=tsize, fontweight="bold")
        ax.text(x + w/2, y + h*0.30, sub, ha="center", va="center", fontsize=ssize, linespacing=1.25)
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=tsize, fontweight="bold")
def A(ax, x1, y1, x2, y2, both=False, ls="-", txt=None, tx=0, ty=0, ha="center"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", lw=0.8, color="black", ls=ls, mutation_scale=7, shrinkA=0, shrinkB=0))
    if txt: ax.text((x1+x2)/2 + tx, (y1+y2)/2 + ty, txt, fontsize=7.3, ha=ha, va="center", style="italic")
fig, ax = plt.subplots(figsize=(10.0, 4.7)); ax.set_xlim(0, 100); ax.set_ylim(0, 47); ax.axis("off")
B(ax, 1, 25, 15, 9, "Task prompt", "T1–T6, identical\nin both modes")
B(ax, 1, 6, 15, 9, "Grader", "SQL ground truth,\nDB state for T4")
B(ax, 31, 37.5, 34, 9.3, "Model server (vLLM 0.19)", "gpt-oss-20b or Qwen3.5-9B\n1× L40S-24Q vGPU · 32K-token context")
ax.add_patch(Rectangle((21, 1.5), 54, 32, fill=False, ec="black", lw=0.8, ls=(0, (4, 2))))
ax.text(22.2, 31.6, "Agent harness", fontsize=8.6, fontweight="bold", va="center")
B(ax, 24, 19.5, 48, 9, "(a) Classic mode", "12 tool schemas in the prompt · one tool call per turn\nfull tool results appended to the conversation (≤ 15 turns)", fc="#f2f2f2")
B(ax, 24, 4, 48, 13, "", None, fc="#e8f1f8")
ax.text(48, 14.8, "(b) Code Mode", ha="center", va="center", fontsize=8.6, fontweight="bold")
B(ax, 25.5, 5.3, 13, 7, "execute_python", "single tool,\ntyped tools.py", tsize=7.6, ssize=6.9)
B(ax, 42, 5.3, 13, 7, "Sandbox", "Docker, no network,\n120 s, stdout ≤ 8k chars", tsize=7.6, ssize=6.9)
B(ax, 57.5, 5.3, 13.5, 7, "JSON-RPC bridge", "tool calls from\nthe script", tsize=7.6, ssize=6.9)
A(ax, 38.5, 8.8, 42, 8.8); A(ax, 55, 8.8, 57.5, 8.8)
B(ax, 81, 22, 18, 9, "MCP server", "12 tools (or 60)")
B(ax, 81, 6, 18, 9, "SQLite CRM", "6 tables, seed 42,\nreset per trial")
A(ax, 16, 29.5, 21, 29.5); A(ax, 21, 10.5, 16, 10.5, txt="answer", ty=1.4)
A(ax, 48, 33.5, 48, 37.5, both=True, txt="requests / responses (usage = tokens)", tx=1.2, ha="left")
A(ax, 72, 24, 81, 26.5, both=True); A(ax, 70.5, 9.5, 81, 24, both=True)
A(ax, 90, 22, 90, 15, both=True)
save(fig, "fig1-setup")
