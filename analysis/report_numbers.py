"""Every number quoted in the text of the report, computed from the results.

Usage (from any directory, after plot_results.py):  python analysis/report_numbers.py [results_dir]

Reads   summary.csv, speedup.csv, placement.csv, run_status.csv, correctness.csv,
        checksums.csv
Writes  report_numbers.txt (also printed)
"""
import os
import sys

import pandas as pd

RESULTS_DIR = (sys.argv[1] if len(sys.argv) > 1 else
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
MIB = 2 ** 20
OUT = []


def path(name):
    return os.path.join(RESULTS_DIR, name)


def say(text=""):
    OUT.append(text)
    print(text)


def one(d, column, **kw):
    for k, v in kw.items():
        d = d[d[k] == v]
    if len(d) != 1:
        raise ValueError(f"{len(d)} rows for {kw}")
    return d[column].iloc[0]


def main():
    s = pd.read_csv(path("summary.csv"))
    sp = pd.read_csv(path("speedup.csv"))
    status = pd.read_csv(path("run_status.csv"))
    correct = pd.read_csv(path("correctness.csv"))
    checks = pd.read_csv(path("checksums.csv"))
    placement = pd.read_csv(path("placement.csv"))
    th = sp[sp["experiment"] == "threads"]

    say("== Runs")
    say(f"processes: {len(status)}; status counts: {status['status'].value_counts().to_dict()}")
    say(f"benchmark time: {status['wall_seconds'].sum() / 60:.1f} min")
    say(f"configurations: {len(s)}; runs per configuration: {s['runs'].min()}-{s['runs'].max()}")

    say("\n== Correctness")
    say(f"element-wise checks: {len(correct)}, passed {correct['passed'].sum()}, "
        f"bitwise equal {correct['bitwise_equal_exact_reference'].sum()}, max rel diff vs "
        f"numpy {correct['max_rel_diff_vs_numpy_matmul'].max():.2e}")
    for _, r in checks.iterrows():
        say(f"n = {r['n']}: {r['distinct_checksums']} checksum over {r['configurations']} "
            f"configurations; equal to Assignment 2: {r['equal_to_assignment2']}")
    say(f"placement: every pinned thread on its requested processor: "
        f"{(placement[placement['placement'] != 'os']['cpu'] == placement[placement['placement'] != 'os']['requested_cpu']).all()}")

    say("\n== Variability (threads experiment)")
    d = s[s["experiment"] == "threads"]
    for n in sorted(d["n"].unique()):
        x = d[d["n"] == n]["iqr_over_median_pct"]
        say(f"n = {n}: IQR/median median {x.median():.2f} %, max {x.max():.2f} %, "
            f"share <= 3 %: {100 * (x <= 3).mean():.0f} %")
    say(f"all n >= 1000: share of configurations with IQR <= 3 %: "
        f"{100 * (d[d['n'] >= 1000]['iqr_over_median_pct'] <= 3).mean():.1f} %")

    say("\n== Sequential baselines and best parallel time (threads experiment)")
    for build in ["scalar", "avx2"]:
        for kernel in ["ikj", "blocked"]:
            for n in [1000, 2000, 4000]:
                t = one(s, "median_time", experiment="threads", build=build, method=kernel, n=n)
                g = th[(th["build"] == build) & (th["kernel"] == kernel) & (th["n"] == n)]
                best = g.loc[g["speedup"].idxmax()]
                say(f"{build} {kernel} n = {n}: T_seq {t:.4g} s ({one(s, 'gflops', experiment='threads', build=build, method=kernel, n=n):.1f} GFLOP/s); "
                    f"best p = {best['threads']} S = {best['speedup']:.2f} "
                    f"[{best['speedup_lo']:.2f}, {best['speedup_hi']:.2f}], "
                    f"T = {best['t_p']:.4g} s, {best['gflops']:.1f} GFLOP/s")

    say("\n== H1 blocked, n >= 1000: E(p) >= 0.8 up to p = 8; S(16)/S(8) <= 1.2")
    for build in ["scalar", "avx2"]:
        g = th[(th["build"] == build) & (th["kernel"] == "blocked") & (th["n"] >= 1000)]
        e = g[g["threads"] <= 8]["efficiency"]
        say(f"{build}: E(p <= 8) min {e.min():.3f}, max {e.max():.3f}")
        for n in [1000, 2000, 4000]:
            s8 = one(g, "speedup", n=n, threads=8)
            s16 = one(g, "speedup", n=n, threads=16)
            s12 = one(g, "speedup", n=n, threads=12)
            say(f"  n = {n}: S(8) {s8:.2f}, S(12) {s12:.2f}, S(16) {s16:.2f}, "
                f"S(16)/S(8) {s16 / s8:.2f}")

    say("\n== H2 ikj, n >= 2000: S(8) < 8 and below blocked; S(16) > S(8)")
    for build in ["scalar", "avx2"]:
        for n in [1000, 2000, 4000]:
            g = th[(th["build"] == build) & (th["n"] == n)]
            say(f"{build} n = {n}: ikj S(8) {one(g, 'speedup', kernel='ikj', threads=8):.2f} "
                f"[{one(g, 'speedup_lo', kernel='ikj', threads=8):.2f}, "
                f"{one(g, 'speedup_hi', kernel='ikj', threads=8):.2f}], "
                f"S(12) {one(g, 'speedup', kernel='ikj', threads=12):.2f}, "
                f"S(16) {one(g, 'speedup', kernel='ikj', threads=16):.2f}; "
                f"blocked S(8) {one(g, 'speedup', kernel='blocked', threads=8):.2f}")
    for build in ["scalar", "avx2"]:
        for n in [1000, 2000, 4000]:
            g = th[(th["build"] == build) & (th["kernel"] == "ikj") & (th["n"] == n)]
            say(f"{build} ikj n = {n}: GFLOP/s at p = 8 {one(g, 'gflops', threads=8):.1f}, "
                f"p = 16 {one(g, 'gflops', threads=16):.1f}")

    say("\n== Superlinear points: S(p) > p with the lower bootstrap bound above p")
    sup = th[(th["speedup_lo"] > th["threads"]) & (th["threads"] > 1)]
    for _, r in sup.iterrows():
        say(f"{r['build']} {r['kernel']} n = {r['n']} p = {r['threads']}: S {r['speedup']:.3f} "
            f"[{r['speedup_lo']:.3f}, {r['speedup_hi']:.3f}], S(p)/S(1) "
            f"{r['speedup'] / one(th, 'speedup', build=r['build'], kernel=r['kernel'], n=r['n'], threads=1):.3f}")
    say("(relative speedup S(p)/S(1) uses the OpenMP kernel with one thread as the baseline)")

    say("\n== H3 n = 100: S(p) < 8 for every p")
    for build in ["scalar", "avx2"]:
        g = th[(th["build"] == build) & (th["n"] == 100)]
        for kernel in ["ikj", "blocked"]:
            x = g[g["kernel"] == kernel]
            best = x.loc[x["speedup"].idxmax()]
            say(f"{build} {kernel}: max S {best['speedup']:.2f} at p = {best['threads']}; "
                f"T_seq {best['t_seq'] * 1e6:.0f} us, T_p {best['t_p'] * 1e6:.0f} us")

    say("\n== H4 T(OpenMP, p = 1) / T_seq")
    for build in ["scalar", "avx2"]:
        for kernel in ["ikj", "blocked"]:
            g = th[(th["build"] == build) & (th["kernel"] == kernel) & (th["threads"] == 1)]
            say(f"{build} {kernel}: " + ", ".join(
                f"n={r['n']} {1 / r['speedup']:.3f} [{1 / r['speedup_hi']:.3f}, {1 / r['speedup_lo']:.3f}]"
                for _, r in g.sort_values("n").iterrows()))

    say("\n== H5 E(12) < E(8); load-imbalance model for blocked: S(12) ~ 12 * S(16) / 16")
    for build in ["scalar", "avx2"]:
        for kernel in ["ikj", "blocked"]:
            for n in [1000, 2000, 4000]:
                g = th[(th["build"] == build) & (th["kernel"] == kernel) & (th["n"] == n)]
                e8, e12 = one(g, "efficiency", threads=8), one(g, "efficiency", threads=12)
                s12, s16 = one(g, "speedup", threads=12), one(g, "speedup", threads=16)
                say(f"{build} {kernel} n = {n}: E(8) {e8:.2f}, E(12) {e12:.2f}; S(12) "
                    f"{s12:.2f}, model 12*S(16)/16 = {12 * s16 / 16:.2f}")

    say("\n== H6 vectorisation: T_scalar / T_avx2 at the same kernel, n, p")
    d = s[s["experiment"] == "threads"]
    for kernel in ["ikj", "blocked"]:
        for n in [1000, 2000, 4000]:
            cells = []
            for method, ps in [(kernel, [1]), (kernel + "_omp", [1, 2, 4, 6, 8, 12, 16])]:
                for p in ps:
                    r = (one(d, "median_time", build="scalar", method=method, n=n, threads=p)
                         / one(d, "median_time", build="avx2", method=method, n=n, threads=p))
                    cells.append(f"{'seq' if method == kernel else p}:{r:.2f}")
            say(f"{kernel} n = {n}: " + " ".join(cells))

    say("\n== H7 memory: peak working set - 24 n^2 bytes and peak private bytes (MiB)")
    for n in [100, 1000, 4000]:
        g = d[(d["build"] == "scalar") & (d["n"] == n)]
        extra = (g["peak_bytes"] - g["storage_bytes"]) / MIB
        priv = g["peak_private_bytes"] / MIB
        say(f"n = {n}: storage {24 * n * n / MIB:.2f} MiB; peak - storage "
            f"{extra.min():.2f}-{extra.max():.2f} MiB; peak private {priv.min():.2f}-"
            f"{priv.max():.2f} MiB")
    for kernel in ["ikj", "blocked"]:
        g = d[(d["build"] == "scalar") & (d["n"] == 4000) & (d["method"] == kernel + "_omp")]
        a = one(g, "peak_private_bytes", threads=2)
        b = one(g, "peak_private_bytes", threads=16)
        say(f"{kernel}_omp n = 4000: private bytes p = 2 -> 16: +{(b - a) / 1024:.0f} KiB "
            f"({(b - a) / 14 / 1024:.0f} KiB per extra thread)")

    say("\n== Placement: compact (two threads per core) against spread")
    pl = sp[sp["experiment"] == "placement"]
    for kernel in ["ikj", "blocked"]:
        for n in [1000, 4000]:
            cells = []
            for p in [2, 4, 8, 16]:
                c = one(pl, "speedup", kernel=kernel, n=n, threads=p)
                sp_ = one(th, "speedup", build="scalar", kernel=kernel, n=n, threads=p)
                cells.append(f"p={p}: compact {c:.2f} spread {sp_:.2f}")
            say(f"{kernel} n = {n}: " + "; ".join(cells))
    say("one core, two threads (compact p = 2) against one thread: per-core throughput ratio")
    for kernel in ["ikj", "blocked"]:
        for n in [1000, 4000]:
            say(f"  {kernel} n = {n}: {one(pl, 'speedup', kernel=kernel, n=n, threads=2):.2f}")

    say("\n== H8 hybrid (24 logical processors, threads placed by Windows)")
    hy = sp[sp["experiment"] == "hybrid"]
    for kernel in ["ikj", "blocked"]:
        for n in [1000, 4000]:
            g = hy[(hy["kernel"] == kernel) & (hy["n"] == n)]
            cells = " ".join(f"S({p})={one(g, 'speedup', threads=p):.2f}"
                             for p in [8, 16, 24, 32, 48])
            say(f"{kernel} n = {n}: {cells}; S(24)/S(16) "
                f"{one(g, 'speedup', threads=24) / one(g, 'speedup', threads=16):.2f}; pinned "
                f"spread S(8) {one(th, 'speedup', build='scalar', kernel=kernel, n=n, threads=8):.2f}"
                f", S(16) {one(th, 'speedup', build='scalar', kernel=kernel, n=n, threads=16):.2f}")
    hp = placement[placement["experiment"] == "hybrid"]
    for p, g in hp.groupby("threads"):
        cpus = g["cpu"].tolist()
        on_p = [c for c in cpus if c < 16]
        say(f"observed placement p = {p}: {len(on_p)} threads on P-core logical processors "
            f"({len({c // 2 for c in on_p})} distinct P-cores), {len(cpus) - len(on_p)} on "
            f"E-cores, {len(set(cpus))} distinct logical processors")

    with open(path("report_numbers.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
