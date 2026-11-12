"""Correctness of the sequential and parallel kernels.

Usage (from the repository root, after building with run_experiment.py):
    python analysis/check_correctness.py              # verify runs + checksum checks
    python analysis/check_correctness.py --checksums  # checksum checks only

1. Element-wise check (results/correctness.csv). For every build, kernel, thread count
   and n in VERIFY_SIZES, the benchmark writes its full result matrix (verify mode). The
   inputs are regenerated independently in NumPy with the SplitMix64 formula, and the
   result is compared with two references:
   - an exact emulation of the C kernels: C = sum over k = 0, 1, ..., n-1 of the outer
     product A[:, k] B[k, :], accumulated in float64 in increasing k. Every element gets
     the same IEEE 754 multiplications and additions in the same order as in the C
     kernels (compiled without fused multiply-add), so the result must be bit-identical;
   - numpy.matmul (OpenBLAS, another summation order, possibly with FMA): the relative
     difference must stay below the tolerance RTOL(n) = n * u, with u = 2^-53 the unit
     roundoff. All inputs are non-negative, so |A||B| = AB and the classical bound
     |fl(AB) - AB| <= gamma_n |A||B| (Higham, 2002) gives a relative error of at most
     about n * u for each of the two results.
2. Checksum check (results/checksums.csv). The sum of all the elements of C of every
   non-batched timed run in results/raw_measurements.csv must take a single value per n
   across methods, builds, thread counts and placements, and equal the checksum of the
   same matrices in Assignment 2 (Cafabri/Assigment2, results/checksums.csv) where that
   assignment measured the same n.
"""
import csv
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from run_experiment import BLOCK, BUILDS, child_env, cpu_list, exe  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
TMP = os.path.join(RESULTS, "verify_tmp")
VERIFY_SIZES = [1, 7, 100, 333, 500]
VERIFY_THREADS = [1, 2, 3, 8, 16, 24]
U = 2.0 ** -53
SEED_A, SEED_B = 42, 43

# Checksums of the dense matrices in Assignment 2 (commit c40444e, results/checksums.csv,
# every method and build had a single checksum per n)
A2_CHECKSUMS = {100: "25018534.16978135", 1000: "25011985768.228165",
                2000: "199919091921.61813", 4000: "1599806487478.5974"}


def splitmix_matrix(seed, n):
    """Values in [0, 10) of the SplitMix64 counter generator, position p = i*n + j."""
    with np.errstate(over="ignore"):
        p = np.arange(1, n * n + 1, dtype=np.uint64)
        z = np.uint64(seed) + p * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return ((z >> np.uint64(11)).astype(np.float64) * 2.0 ** -53 * 10.0).reshape(n, n)


def exact_reference(A, B):
    """IEEE emulation of the C kernels: products added in increasing k."""
    C = np.zeros_like(A)
    for k in range(A.shape[0]):
        C += np.multiply.outer(A[:, k], B[k, :])
    return C


def verify_configs():
    for build in BUILDS:
        yield build, "ikj", 1
        yield build, "blocked", 1
        for p in VERIFY_THREADS:
            yield build, "ikj_omp", p
            yield build, "blocked_omp", p


def check_elements():
    os.makedirs(TMP, exist_ok=True)
    rows, ok = [], True
    for n in VERIFY_SIZES:
        A, B = splitmix_matrix(SEED_A, n), splitmix_matrix(SEED_B, n)
        exact, blas = exact_reference(A, B), A @ B
        rtol = n * U
        for build, method, p in verify_configs():
            block = BLOCK[build] if method.startswith("blocked") else 0
            placement = "spread" if p <= 16 else "os"
            path = os.path.join(TMP, f"{build}_{method}_{p}_{n}.txt")
            subprocess.run([exe(build), "verify", method, str(n), str(block), str(p),
                            cpu_list(placement), path], check=True, env=child_env())
            C = np.loadtxt(path).reshape(n, n)
            os.remove(path)
            bitwise = bool(np.array_equal(C, exact))
            rel = float(np.max(np.abs(C - blas) / np.abs(blas))) if n > 0 else 0.0
            passed = bitwise and rel <= rtol
            ok &= passed
            rows.append([n, build, method, block or "", p, placement, bitwise,
                         f"{rel:.3e}", f"{rtol:.3e}", passed])
            print(f"n={n:4d} {build:6s} {method:11s} p={p:2d} bitwise={bitwise} "
                  f"max_rel_vs_numpy={rel:.2e} (tol {rtol:.2e}) {'OK' if passed else 'FAIL'}")
    os.rmdir(TMP)
    with open(os.path.join(RESULTS, "correctness.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "build", "method", "block", "threads", "placement",
                    "bitwise_equal_exact_reference", "max_rel_diff_vs_numpy_matmul",
                    "tolerance_n_u", "passed"])
        w.writerows(rows)
    return ok


def check_checksums():
    path = os.path.join(RESULTS, "raw_measurements.csv")
    if not os.path.exists(path):
        print("no raw_measurements.csv yet: checksum check skipped")
        return True
    values = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["checksum"]:
                values.setdefault(int(r["n"]), {}).setdefault(r["checksum"], set()).add(
                    (r["experiment"], r["method"], r["build"], r["threads"], r["placement"]))
    rows, ok = [], True
    for n in sorted(values):
        sums = values[n]
        a2 = A2_CHECKSUMS.get(n, "")
        A, B = splitmix_matrix(SEED_A, n), splitmix_matrix(SEED_B, n)
        ref = float((A @ B).sum())
        configs = sum(len(c) for c in sums.values())
        for checksum in sums:
            rel = abs(float(checksum) - ref) / abs(ref)
            passed = len(sums) == 1 and (not a2 or checksum == a2) and rel <= 2 * n * n * U
            ok &= passed
            rows.append([n, len(sums), configs, checksum, a2, a2 == checksum if a2 else "",
                         repr(ref), f"{rel:.3e}", passed])
            print(f"n={n:5d} distinct={len(sums)} configurations={configs} checksum={checksum} "
                  f"A2={a2 or '-'} rel_vs_numpy={rel:.2e} {'OK' if passed else 'FAIL'}")
    with open(os.path.join(RESULTS, "checksums.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "distinct_checksums", "configurations", "checksum",
                    "assignment2_checksum", "equal_to_assignment2", "numpy_checksum",
                    "rel_diff_vs_numpy", "passed"])
        w.writerows(rows)
    return ok


if __name__ == "__main__":
    good = True
    if "--checksums" not in sys.argv:
        good &= check_elements()
    good &= check_checksums()
    print("ALL CHECKS PASSED" if good else "SOME CHECKS FAILED")
    sys.exit(0 if good else 1)
