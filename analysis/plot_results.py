"""Summary tables, speedup and efficiency, LaTeX tables and figures of the report.

Usage (from any directory):  python analysis/plot_results.py [results_dir]

Reads   raw_measurements.csv, process_memory.csv and run_status.csv of results_dir
Writes  summary.csv      median, quartiles, IQR, GFLOP/s and peak memory per configuration
        speedup.csv      S(p), E(p), bootstrap interval of S(p) and the Karp-Flatt metric
        plot_*.png       figures
        tables/*.tex     tables included by the report

Definitions (Assignment 3 brief): T_seq(n) is the median time of the sequential kernel
(ikj or blocked, same build, same n, same experiment), T_p(n) the median time of its
OpenMP version with p threads, S(p) = T_seq / T_p and E(p) = S(p) / p. All the medians
pool the measured runs of every process of a configuration. The 95 % interval of S(p) is
a percentile bootstrap: the measured runs of the two configurations are resampled
independently (BOOTSTRAP resamples, fixed seed) and the ratio of the two medians is
recomputed. The Karp-Flatt metric e(p) = (1/S - 1/p) / (1 - 1/p) is the serial fraction
that would explain the observed speedup under Amdahl's law.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullLocator

RESULTS_DIR = (sys.argv[1] if len(sys.argv) > 1 else
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
BOOTSTRAP = 2000
SEED = 2026
MIB = 2 ** 20
PHYSICAL_P_CORES = 8

COLORS = {100: "#9e9e9e", 250: "#7b5ea7", 500: "#4c8eda", 1000: "#5baa5b",
          2000: "#e07b39", 4000: "#c0504d"}
KERNEL_LABEL = {"ikj": "ikj (row-wise, no blocking)", "blocked": "blocked ikj (cache tiles)"}
BUILD_STYLE = {"scalar": "-", "avx2": "--"}
BUILD_MARKER = {"scalar": "o", "avx2": "s"}
BUILD_LABEL = {"scalar": "scalar build", "avx2": "AVX2 build"}

plt.rcParams.update({"xtick.labelsize": 10, "ytick.labelsize": 10})
KEYS = ["experiment", "method", "build", "n", "block", "threads", "placement"]


def path(*parts):
    return os.path.join(RESULTS_DIR, *parts)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(path(name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def grid(ax):
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#dddddd", linewidth=0.7)


def thread_axis(ax, ps):
    ax.xaxis.set_major_locator(FixedLocator(ps))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("threads p")


# --------------------------------------------------------------------------- data

def load():
    raw = pd.read_csv(path("raw_measurements.csv"), dtype={"checksum": str})
    raw["block"] = raw["block"].fillna(0).astype(int)
    peaks = pd.read_csv(path("process_memory.csv"))
    peaks["block"] = peaks["block"].fillna(0).astype(int)
    status = pd.read_csv(path("run_status.csv"))
    status["block"] = status["block"].fillna(0).astype(int)
    return raw, peaks, status


def kernel_of(method):
    return method.replace("_omp", "")


def summarise(raw, peaks):
    m = raw[raw["run_type"] == "measured"]
    g = m.groupby(KEYS)
    s = g["time_seconds"].agg(
        runs="size", median_time="median",
        q1_time=lambda x: x.quantile(0.25), q3_time=lambda x: x.quantile(0.75),
        min_time="min", max_time="max").reset_index()
    s["processes"] = g["process"].nunique().values
    s["batch_size"] = g["batch_size"].max().values
    s["iqr_time"] = s["q3_time"] - s["q1_time"]
    s["iqr_over_median_pct"] = 100 * s["iqr_time"] / s["median_time"]
    s["gflops"] = 2 * s["n"].astype(float) ** 3 / s["median_time"] / 1e9
    s["storage_bytes"] = 24 * s["n"].astype(float) ** 2
    p = peaks.groupby(KEYS).agg(peak_bytes=("peak_bytes", "median"),
                                peak_private_bytes=("peak_private_bytes", "median")).reset_index()
    s = s.merge(p, on=KEYS, how="left")
    s["kernel"] = s["method"].map(kernel_of)
    s["parallel"] = s["method"].str.endswith("_omp")
    return s


def bootstrap_ratio(seq, par, rng):
    i = rng.integers(0, len(seq), (BOOTSTRAP, len(seq)))
    j = rng.integers(0, len(par), (BOOTSTRAP, len(par)))
    ratios = np.median(seq[i], axis=1) / np.median(par[j], axis=1)
    return np.percentile(ratios, [2.5, 97.5])


def speedups(raw, summary):
    """S(p) and E(p) of every parallel configuration against its sequential kernel. The
    placement experiment has no sequential runs of its own: it uses those of the threads
    experiment (same build and n, thread 0 on the same processor)."""
    m = raw[raw["run_type"] == "measured"]
    times = {k: v["time_seconds"].to_numpy() for k, v in m.groupby(KEYS)}
    rng = np.random.default_rng(SEED)
    rows = []
    for _, r in summary[summary["parallel"]].iterrows():
        seq_exp = "threads" if r["experiment"] == "placement" else r["experiment"]
        seq_place = "spread" if r["experiment"] == "placement" else r["placement"]
        seq_key = (seq_exp, r["kernel"], r["build"], r["n"], r["block"], 1, seq_place)
        if seq_key not in times:
            continue
        seq, par = times[seq_key], times[tuple(r[k] for k in KEYS)]
        t_seq = np.median(seq)
        s = t_seq / r["median_time"]
        p = r["threads"]
        lo, hi = bootstrap_ratio(seq, par, rng)
        rows.append({**{k: r[k] for k in KEYS}, "kernel": r["kernel"], "t_seq": t_seq,
                     "t_p": r["median_time"], "speedup": s, "speedup_lo": lo,
                     "speedup_hi": hi, "efficiency": s / p,
                     "karp_flatt": (1 / s - 1 / p) / (1 - 1 / p) if p > 1 else np.nan,
                     "gflops": r["gflops"]})
    return pd.DataFrame(rows)


def pick(d, **kw):
    for k, v in kw.items():
        d = d[d[k] == v]
    return d.sort_values("threads")


# --------------------------------------------------------------------------- figures

def plot_speedup(sp, build="scalar"):
    """S(p) and E(p) of both kernels, one line per n (threads experiment)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True)
    for col, kernel in enumerate(["ikj", "blocked"]):
        d = pick(sp, experiment="threads", build=build, kernel=kernel)
        ps = sorted(d["threads"].unique())
        for n in sorted(d["n"].unique()):
            dn = pick(d, n=n)
            axes[0, col].errorbar(dn["threads"], dn["speedup"],
                                  yerr=[dn["speedup"] - dn["speedup_lo"],
                                        dn["speedup_hi"] - dn["speedup"]],
                                  marker="o", ms=4, capsize=2, color=COLORS.get(n),
                                  label=f"n = {n}")
            axes[1, col].plot(dn["threads"], dn["efficiency"], marker="o", ms=4,
                              color=COLORS.get(n), label=f"n = {n}")
        axes[0, col].plot(ps, ps, color="black", linewidth=0.8, linestyle=":",
                          label="ideal S(p) = p")
        axes[1, col].axhline(1, color="black", linewidth=0.8, linestyle=":")
        for ax in axes[:, col]:
            ax.axvline(PHYSICAL_P_CORES, color="#888888", linewidth=0.8, linestyle="--")
            thread_axis(ax, ps)
            grid(ax)
        axes[0, col].set_title(KERNEL_LABEL[kernel], fontsize=11)
        axes[0, col].set_ylabel("speedup S(p) = T_seq / T_p")
        axes[1, col].set_ylabel("efficiency E(p) = S(p) / p")
        axes[1, col].set_ylim(0, 1.3)
        axes[0, col].set_xlabel("")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.text(0.5, -0.01, "Dashed vertical line: 8 physical P-cores (p > 8 uses the second "
             "logical processor of a core). Bars: 95 % bootstrap interval of S(p).",
             ha="center", fontsize=9)
    save(fig, f"plot_speedup_{build}.png")


def plot_gflops(sp, summary):
    """Throughput of both builds against p at the two largest sizes: separates the SIMD
    effect (vertical distance between builds) from the thread effect (slope)."""
    sizes = sorted(summary["n"].unique())[-2:]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, kernel in zip(axes, ["ikj", "blocked"]):
        for build in ["scalar", "avx2"]:
            for n in sizes:
                d = pick(summary, experiment="threads", build=build, kernel=kernel,
                         parallel=True, n=n)
                seq = pick(summary, experiment="threads", build=build, kernel=kernel,
                           parallel=False, n=n)
                ax.plot(d["threads"], d["gflops"], linestyle=BUILD_STYLE[build],
                        marker=BUILD_MARKER[build], ms=4, color=COLORS.get(n),
                        label=f"{BUILD_LABEL[build]}, n = {n}")
                if len(seq):
                    ax.scatter([1], seq["gflops"], marker="x", color=COLORS.get(n), s=40,
                               zorder=3)
        ax.axvline(PHYSICAL_P_CORES, color="#888888", linewidth=0.8, linestyle="--")
        thread_axis(ax, sorted(d["threads"].unique()))
        ax.set_title(KERNEL_LABEL[kernel], fontsize=11)
        grid(ax)
    axes[0].set_ylabel("GFLOP/s  (2n$^3$ / median time)")
    axes[0].legend(fontsize=8)
    fig.text(0.5, -0.02, "x: sequential kernel. Solid: scalar build; dashed: AVX2 build.",
             ha="center", fontsize=9)
    save(fig, "plot_gflops_builds.png")


def plot_overhead(summary):
    """T_p(1) / T_seq against n: the cost of the OpenMP region and of the row split with
    one thread."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for build in ["scalar", "avx2"]:
        for kernel, color in [("ikj", "#4c8eda"), ("blocked", "#c0504d")]:
            seq = pick(summary, experiment="threads", build=build, kernel=kernel,
                       parallel=False).set_index("n")
            one = pick(summary, experiment="threads", build=build, kernel=kernel,
                       parallel=True, threads=1).set_index("n")
            r = (one["median_time"] / seq["median_time"]).dropna()
            ax.plot(r.index, r.values, linestyle=BUILD_STYLE[build],
                    marker=BUILD_MARKER[build], color=color, ms=4,
                    label=f"{kernel}, {BUILD_LABEL[build]}")
    ax.axhline(1, color="black", linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ns = sorted(summary["n"].unique())
    ax.xaxis.set_major_locator(FixedLocator(ns))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("matrix size n")
    ax.set_ylabel("T(OpenMP, p = 1) / T_seq")
    ax.legend(fontsize=8)
    grid(ax)
    save(fig, "plot_overhead_p1.png")


def plot_time(summary, build="scalar"):
    """Median time against p for every n (log scale), both kernels."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, kernel in zip(axes, ["ikj", "blocked"]):
        for n in sorted(summary["n"].unique()):
            d = pick(summary, experiment="threads", build=build, kernel=kernel, parallel=True,
                     n=n)
            ax.errorbar(d["threads"], d["median_time"],
                        yerr=[d["median_time"] - d["q1_time"], d["q3_time"] - d["median_time"]],
                        marker="o", ms=4, capsize=2, color=COLORS.get(n), label=f"n = {n}")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.axvline(PHYSICAL_P_CORES, color="#888888", linewidth=0.8, linestyle="--")
        thread_axis(ax, sorted(d["threads"].unique()))
        ax.set_title(KERNEL_LABEL[kernel], fontsize=11)
        grid(ax)
    axes[0].set_ylabel("median time per multiplication (s)")
    axes[1].legend(fontsize=8)
    save(fig, f"plot_time_{build}.png")


def plot_placement(sp):
    """Spread (one thread per physical core first) against compact placement."""
    if not len(sp[sp["experiment"] == "placement"]):
        return
    sizes = sorted(sp[sp["experiment"] == "placement"]["n"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, kernel in zip(axes, ["ikj", "blocked"]):
        for n in sizes:
            for exp, place, style in [("threads", "spread", "-"), ("placement", "compact", "--")]:
                d = pick(sp, experiment=exp, build="scalar", kernel=kernel, n=n,
                         placement=place)
                d = d[d["threads"].isin(sp[sp["experiment"] == "placement"]["threads"])]
                ax.plot(d["threads"], d["speedup"], linestyle=style, marker="o", ms=4,
                        color=COLORS.get(n), label=f"{place}, n = {n}")
        ps = sorted(sp[sp["experiment"] == "placement"]["threads"].unique())
        ax.plot(ps, ps, color="black", linewidth=0.8, linestyle=":")
        thread_axis(ax, ps)
        ax.set_title(KERNEL_LABEL[kernel], fontsize=11)
        grid(ax)
    axes[0].set_ylabel("speedup S(p)")
    axes[0].legend(fontsize=8)
    save(fig, "plot_placement.png")


def plot_hybrid(sp):
    """All 24 logical processors, threads placed by Windows, including oversubscription."""
    if not len(sp[sp["experiment"] == "hybrid"]):
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for kernel, color in [("ikj", "#4c8eda"), ("blocked", "#c0504d")]:
        for n, style in zip(sorted(sp[sp["experiment"] == "hybrid"]["n"].unique()), ["--", "-"]):
            d = pick(sp, experiment="hybrid", kernel=kernel, n=n)
            ax.errorbar(d["threads"], d["speedup"],
                        yerr=[d["speedup"] - d["speedup_lo"], d["speedup_hi"] - d["speedup"]],
                        linestyle=style, marker="o", ms=4, capsize=2, color=color,
                        label=f"{kernel}, n = {n}")
    ps = sorted(sp[sp["experiment"] == "hybrid"]["threads"].unique())
    ax.axvspan(24, max(ps) + 2, color="#f2f2f2", zorder=0)
    ax.text(25, 1.2, "oversubscribed\n(p > 24 logical processors)", fontsize=8)
    ax.axvline(16, color="#888888", linewidth=0.8, linestyle="--")
    ax.text(15.5, 1.2, "16 P-core logical\nprocessors", fontsize=8, ha="right")
    thread_axis(ax, ps)
    ax.set_ylabel("speedup S(p)")
    ax.legend(fontsize=8, loc="upper left")
    grid(ax)
    save(fig, "plot_hybrid.png")


def plot_memory(summary):
    """Peak working set minus the 24n^2 bytes of A, B and C, against p."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for kernel, color in [("ikj", "#4c8eda"), ("blocked", "#c0504d")]:
        for n, style in [(1000, "--"), (4000, "-")]:
            d = pick(summary, experiment="threads", build="scalar", kernel=kernel,
                     parallel=True, n=n)
            seq = pick(summary, experiment="threads", build="scalar", kernel=kernel,
                       parallel=False, n=n)
            if not len(d):
                continue
            ax.plot(d["threads"], (d["peak_bytes"] - d["storage_bytes"]) / MIB,
                    linestyle=style, marker="o", ms=4, color=color,
                    label=f"{kernel}_omp, n = {n}")
            if len(seq):
                ax.scatter([1], (seq["peak_bytes"] - seq["storage_bytes"]) / MIB,
                           marker="x", color=color, s=40, zorder=3)
    thread_axis(ax, sorted(summary[summary["experiment"] == "threads"]["threads"].unique()))
    ax.set_ylabel("peak working set - 24n$^2$ bytes (MiB)")
    ax.legend(fontsize=8)
    grid(ax)
    save(fig, "plot_memory.png")


# --------------------------------------------------------------------------- tables

def fmt_time(t):
    if t >= 1:
        return f"{t:.3g} s" if t < 100 else f"{t:.0f} s"
    if t >= 1e-3:
        return f"{t * 1e3:.3g} ms"
    return f"{t * 1e6:.3g} $\\mu$s"


def write_table(name, lines):
    os.makedirs(path("tables"), exist_ok=True)
    with open(path("tables", name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote tables/" + name)


def table_runtime(summary, build):
    """Median time and IQR (as % of the median) per n and p, both kernels."""
    ps = sorted(summary[(summary["experiment"] == "threads") & summary["parallel"]]["threads"].unique())
    lines = []
    for kernel in ["ikj", "blocked"]:
        lines.append(f"\\multicolumn{{{len(ps) + 2}}}{{l}}{{\\textbf{{{kernel}}}}} \\\\")
        for n in sorted(summary["n"].unique()):
            seq = pick(summary, experiment="threads", build=build, kernel=kernel,
                       parallel=False, n=n)
            cells = [str(n)]
            cells.append(fmt_time(seq["median_time"].iloc[0]) if len(seq) else "--")
            for p in ps:
                d = pick(summary, experiment="threads", build=build, kernel=kernel,
                         parallel=True, n=n, threads=p)
                cells.append(f"{fmt_time(d['median_time'].iloc[0])}" if len(d) else "--")
            lines.append(" & ".join(cells) + " \\\\")
    write_table(f"runtime_{build}.tex", lines)


def table_speedup(sp, build):
    """S(p) [95 % interval] and E(p) per n and p."""
    d0 = sp[(sp["experiment"] == "threads") & (sp["build"] == build)]
    ps = sorted(d0["threads"].unique())
    lines = []
    for kernel in ["ikj", "blocked"]:
        lines.append(f"\\multicolumn{{{len(ps) + 1}}}{{l}}{{\\textbf{{{kernel}}}: "
                     f"S(p) and, below, E(p)}} \\\\")
        for n in sorted(d0["n"].unique()):
            s_cells, e_cells = [str(n)], [""]
            for p in ps:
                d = pick(d0, kernel=kernel, n=n, threads=p)
                if len(d):
                    s_cells.append(f"{d['speedup'].iloc[0]:.2f}")
                    e_cells.append(f"\\footnotesize {d['efficiency'].iloc[0]:.2f}")
                else:
                    s_cells.append("--")
                    e_cells.append("")
            lines.append(" & ".join(s_cells) + " \\\\")
            lines.append(" & ".join(e_cells) + " \\\\[1pt]")
    write_table(f"speedup_{build}.tex", lines)


def table_variability(summary):
    """Largest IQR / median and processes per size (threads experiment)."""
    lines = []
    d0 = summary[summary["experiment"] == "threads"]
    for n in sorted(d0["n"].unique()):
        d = d0[d0["n"] == n]
        worst = d.loc[d["iqr_over_median_pct"].idxmax()]
        lines.append(f"{n} & {int(d['runs'].min())}--{int(d['runs'].max())} & "
                     f"{int(d['batch_size'].min())}--{int(d['batch_size'].max())} & "
                     f"{d['iqr_over_median_pct'].median():.2f} & "
                     f"{d['iqr_over_median_pct'].max():.2f} "
                     f"(\\texttt{{{worst['method'].replace('_', chr(92) + '_')}}}, "
                     f"{worst['build']}, $p = {int(worst['threads'])}$) \\\\")
    write_table("variability.tex", lines)


def table_vectorisation(summary):
    """T_scalar / T_avx2 at the same kernel, n and p: the SIMD effect at fixed threads."""
    d0 = summary[summary["experiment"] == "threads"]
    ps = [1] + sorted(d0[d0["parallel"]]["threads"].unique())
    lines = []
    for kernel in ["ikj", "blocked"]:
        for n in sorted(d0["n"].unique())[-3:]:
            cells = [kernel, str(n)]
            for i, p in enumerate(ps):
                par = i > 0
                s = pick(d0, build="scalar", kernel=kernel, n=n, parallel=par, threads=p)
                v = pick(d0, build="avx2", kernel=kernel, n=n, parallel=par, threads=p)
                cells.append(f"{s['median_time'].iloc[0] / v['median_time'].iloc[0]:.2f}"
                             if len(s) and len(v) else "--")
            lines.append(" & ".join(cells) + " \\\\")
    write_table("vectorisation.tex", lines)


def table_memory(summary):
    """Peak working set per p at n = 4000 (MiB) and the computed storage."""
    d0 = summary[(summary["experiment"] == "threads") & (summary["build"] == "scalar")]
    n = d0["n"].max()
    ps = sorted(d0[d0["parallel"]]["threads"].unique())
    lines = []
    for kernel in ["ikj", "blocked"]:
        seq = pick(d0, kernel=kernel, n=n, parallel=False)
        cells = [kernel, f"{24 * n * n / MIB:.1f}", f"{seq['peak_bytes'].iloc[0] / MIB:.1f}"]
        for p in ps:
            d = pick(d0, kernel=kernel, n=n, parallel=True, threads=p)
            cells.append(f"{d['peak_bytes'].iloc[0] / MIB:.1f}" if len(d) else "--")
        lines.append(" & ".join(cells) + " \\\\")
    write_table("memory.tex", lines)


def table_status(status):
    counts = status.groupby(["experiment", "status"]).size().unstack(fill_value=0)
    lines = [f"{exp} & " + " & ".join(str(counts.loc[exp].get(s, 0))
                                      for s in ["OK", "TIMEOUT", "MEMORY_LIMIT", "ERROR",
                                                "SKIPPED"]) + " \\\\"
             for exp in counts.index]
    write_table("status.tex", lines)


def main():
    raw, peaks, status = load()
    summary = summarise(raw, peaks)
    summary.to_csv(path("summary.csv"), index=False)
    sp = speedups(raw, summary)
    sp.to_csv(path("speedup.csv"), index=False)
    print("wrote summary.csv and speedup.csv")

    for build in ["scalar", "avx2"]:
        plot_speedup(sp, build)
        plot_time(summary, build)
        table_runtime(summary, build)
        table_speedup(sp, build)
    plot_gflops(sp, summary)
    plot_overhead(summary)
    plot_placement(sp)
    plot_hybrid(sp)
    plot_memory(summary)
    table_variability(summary)
    table_vectorisation(summary)
    table_memory(summary)
    table_status(status)


if __name__ == "__main__":
    main()
