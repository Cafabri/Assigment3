# Predictions (stated before the main measurement run)

Committed while `run_experiment.py` was running and before any of its results were
analysed. H1–H4 and H6–H8 were derived only from the background (cache sizes, core
layout, Assignment 2 results). H5 was formulated after the smoke test of the runner
(`--smoke`, n = 100 and 250 only), which showed a drop of the speedup from p = 8 to
p = 12, so it is a prediction informed by a pilot, not a blind one.

Notation: T_seq = median time of the sequential kernel, T_p = median time of its OpenMP
version with p threads, S(p) = T_seq / T_p, E(p) = S(p) / p. Platform: 8 P-cores with
two logical processors each (16 logical), threads placed one per physical core first.

| Id | Prediction | Reason |
|----|------------|--------|
| H1 | blocked, n ≥ 1000: E(p) ≥ 0.8 up to p = 8; S(16) / S(8) ≤ 1.2 | The tiled kernel is limited by arithmetic in each core; the two logical processors of a core share its floating-point units |
| H2 | ikj, n ≥ 2000: S(8) clearly below 8 and lower than that of blocked; S(16) > S(8) | For n ≥ 2000 B (32 MB) does not fit in the 30 MB L3 and is streamed from memory for every row of C, so the shared memory bandwidth limits the speedup; a second thread per core can hide part of the memory latency |
| H3 | n = 100: S(p) stays below 8 for every p | Each call does only 2·10^6 flops (about 0.2–1 ms); the fork/join of the parallel region and the imbalance of 100 rows among p threads are not negligible |
| H4 | T(OpenMP, p = 1) / T_seq is within 2 % of 1 for n ≥ 250 | With one thread the parallel kernel runs the same loops; only the region entry and the row-range computation are added |
| H5 | (pilot-informed) E(12) < E(8) for both kernels | With 12 threads four cores run two threads and four run one; the rows are divided equally, so the threads sharing a core finish last |
| H6 | AVX2 / scalar time ratio of blocked is about the same at every p (2–3×); that of ikj decreases as p grows at n ≥ 2000 | SIMD speeds up the arithmetic of each core, which keeps limiting blocked; ikj becomes limited by the shared memory bandwidth, which SIMD does not increase |
| H7 | Peak working set ≈ 24n² bytes + a constant; growth with p below 1 MiB per thread | The threads share A, B and C; each thread adds only its stack and runtime data |
| H8 | All 24 logical processors, threads placed by Windows: S(24) not more than 10 % above S(16); S(32) and S(48) below S(24) for blocked | The E-cores are slower and the rows are divided equally, so their threads finish last; more threads than logical processors add time-slicing and cache interference |
