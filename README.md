# Assignment 3 — Parallel Matrix Multiplication with OpenMP

**BIG DATA · Academic year 2026-2027**  
Grado en Ciencia e Ingeniería de Datos — Universidad de Las Palmas de Gran Canaria  
**Author:** Carlos Falcón Brito  
**Report:** [`report/Assignment03_FalconBrito_Report.pdf`](report/Assignment03_FalconBrito_Report.pdf)

## Research question

How does the number of threads affect the time, speedup, efficiency and memory of an
OpenMP matrix multiplication compared with its equivalent sequential kernel, and under
which conditions (matrix size, kernel, core sharing, vectorisation) does parallel
execution help?

The predictions were committed before the results were analysed:
[`report/hypotheses.md`](report/hypotheses.md).

## Reused material

The sequential baselines are the `ikj` and cache-blocked kernels of Assignment 2
([Cafabri/Assigment2](https://github.com/Cafabri/Assigment2), commit `c40444e`,
`c/matmul.c`), copied unchanged into `c/matmul.c` (`ijk`, from Assignment 1, is kept only
for the self-tests). The SplitMix64 generator and seeds (42 and 43), the timer, the
batching rule, the budget, the peak-memory measurement and the block sizes (b = 32 scalar,
b = 256 AVX2, `results/block_choice.txt` of Assignment 2) are also those of Assignment 2.
The inputs are therefore the same matrices, and `analysis/check_correctness.py` checks
that the checksums equal those of Assignment 2.

## Platform

**The study runs on Windows only.** The C timer (`QueryPerformanceCounter`), the thread
pinning (`SetThreadAffinityMask`), the process affinity, the topology query
(`GetLogicalProcessorInformationEx`) and the peak-memory measurement
(`GetProcessMemoryInfo`) use Windows APIs. Running the study on another Windows machine
reproduces the procedure, not the numbers; the placement lists are derived from the
topology of the machine.

## Repository structure

```
├── run_experiment.py          # Builds both binaries, records the environment, runs the experiments
├── c/
│   ├── matmul.h / matmul.c    # Sequential kernels of Assignment 2 (ijk, ikj, blocked), unchanged
│   ├── parallel.h / parallel.c  # OpenMP versions (row split per thread) and thread pinning
│   └── benchmark.c            # Inputs, timing, batching, team-size check, self-tests, verify mode
├── analysis/
│   ├── check_correctness.py   # Bitwise and NumPy comparison of every kernel; checksum checks
│   ├── plot_results.py        # summary.csv, speedup.csv, figures and LaTeX tables
│   └── report_numbers.py      # Every number quoted in the report, and the test of each prediction
├── results/
│   ├── raw_measurements.csv   # Every timed run (warm-up, measured and check runs)
│   ├── run_status.csv         # OK / TIMEOUT / MEMORY_LIMIT / ERROR / SKIPPED per process
│   ├── process_memory.csv     # Peak working set and peak private bytes of every process
│   ├── placement.csv          # Logical processor observed for every OpenMP thread
│   ├── vectorisation_report.txt  # gcc -fopt-info-vec of both builds (vectorised and missed loops)
│   ├── environment.txt        # Hardware, OS, toolchain, OpenMP settings, placements, protocol
│   ├── correctness.csv        # 140 element-wise checks
│   ├── checksums.csv          # One checksum per n across all configurations, vs Assignment 2
│   ├── summary.csv            # Median, quartiles, IQR, GFLOP/s, memory per configuration
│   ├── speedup.csv            # S(p), E(p), 95 % bootstrap interval, Karp-Flatt metric
│   ├── report_numbers.txt     # Numbers quoted in the report
│   ├── tables/*.tex           # Tables included by the report
│   └── plot_*.png             # Figures
└── report/
    ├── hypotheses.md          # Predictions, committed before the analysis
    ├── Assignment3_Report.tex
    └── Assignment03_FalconBrito_Report.pdf
```

## Environment

| Item | Value |
|------|-------|
| OS | Windows 11 Pro |
| CPU | Intel Core i7-13700KF: 8 P-cores with SMT (logical processors 0-1, 2-3, …, 14-15) + 8 E-cores (16–23) |
| Cache / RAM | 48 KB L1D and 2 MB L2 per P-core, 30 MB shared L3 / 31.8 GiB |
| Power plan | Balanced (Windows default), frequency not fixed |
| C compiler | gcc 16.1.0 (MSYS2/MinGW64) with its OpenMP runtime (libgomp) |
| Build `scalar` | `-O2 -ffp-contract=off -fopenmp` (no kernel loop vectorised) |
| Build `avx2` | `-O2 -ffp-contract=off -fopenmp -fvect-cost-model=dynamic -mavx2` (inner `j` loops vectorised, 32-byte vectors) |
| OpenMP settings | `OMP_DYNAMIC=false`, `omp_set_dynamic(0)`, `num_threads(p)`; `OMP_PROC_BIND`, `OMP_PLACES`, `OMP_WAIT_POLICY` unset |
| Python | CPython 3.11.9 (runner and analysis only) |
| Analysis | numpy 2.2, pandas 2.2, matplotlib 3.10 |

## Reproducing the study

Requirements: MSYS2 with `mingw-w64-x86_64-gcc` (which includes libgomp), Python 3.11 and
`pip install numpy pandas matplotlib`. The path to `gcc.exe` is set at the top of
`run_experiment.py` (`GCC`); change it if gcc is installed elsewhere. Close other
programs while the benchmarks run.

```bash
python run_experiment.py                  # build + all three experiments (157 min here)
python analysis/check_correctness.py      # element-wise checks + checksum checks
python analysis/plot_results.py           # summary.csv, speedup.csv, figures and tables
python analysis/report_numbers.py         # numbers quoted in the report
```

`run_experiment.py --only threads|placement|hybrid` runs one experiment; a run that stops
can be resumed (processes that already have a status are not repeated).
`--smoke` (n ≤ 250, written to `results/smoke/`) only tests the runner.

To compile the report (run twice so that references resolve):

```bash
cd report
pdflatex -jobname=Assignment03_FalconBrito_Report Assignment3_Report.tex
```

The benchmark can also be built and run on its own:

```bash
cd c
gcc -O2 -ffp-contract=off -fopenmp -DBUILD_NAME='"scalar"' -o benchmark_scalar.exe matmul.c parallel.c benchmark.c
gcc -O2 -ffp-contract=off -fopenmp -fvect-cost-model=dynamic -mavx2 -DBUILD_NAME='"avx2"' -o benchmark_avx2.exe matmul.c parallel.c benchmark.c
# run <method> <n> <block> <threads> <cpus> <warmups> <reps>
benchmark_scalar.exe run blocked 2000 32 1 0 2 5                                     # sequential, pinned to 0
benchmark_scalar.exe run blocked_omp 2000 32 8 0,2,4,6,8,10,12,14 2 5                # 8 threads, spread
benchmark_scalar.exe run ikj_omp 2000 0 16 0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15 2 5
benchmark_scalar.exe run ikj_omp 2000 0 24 none 2 5                                  # unpinned
benchmark_scalar.exe verify blocked_omp 100 32 8 none out.txt                        # full result matrix
benchmark_scalar.exe placement 16 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15              # thread -> processor
```

Methods: `ikj`, `blocked` (sequential, 1 thread), `ikj_omp`, `blocked_omp`. `cpus` is
the placement list (thread t runs on entry t) or `none`. gcc's `bin` directory must be on
`PATH` so that `libgomp-1.dll` is found.

## Experimental settings

- **Inputs:** dense n × n float64, SplitMix64 counter generator of Assignment 1 (seed 42
  for A, 43 for B, uniform in [0, 10), position p = i·n + j). `check_correctness.py`
  regenerates them independently with NumPy.
- **Parallel kernels:** one `omp parallel` region per call with exactly p threads; thread
  t computes a contiguous range of rows (sizes differ by at most one) with the loops of
  the sequential kernel. No shared write, lock or reduction; results bit-identical to the
  sequential kernel.
- **Experiments:**
  - `threads` (main study and vectorisation extension): both builds, both kernels,
    sequential and p ∈ {1, 2, 4, 6, 8, 12, 16}, n ∈ {100, 250, 500, 1000, 2000, 4000},
    process restricted to the P-cores, *spread* placement
    (0, 2, …, 14, then 1, 3, …, 15).
  - `placement`: *compact* placement (0, 1, 2, …), p ∈ {2, 4, 8, 16}, scalar,
    n ∈ {1000, 4000}.
  - `hybrid`: all 24 logical processors, threads not pinned (Windows places them),
    p ∈ {1, 8, 16, 24, 32, 48} and the sequential kernels, scalar, n ∈ {1000, 4000}.
- **Timer:** `QueryPerformanceCounter` around the kernel call only; inputs generated and C
  allocated and zeroed outside it. The parallel region (fork/join) is inside the timed
  call; thread creation is not (an untimed region first checks the team size).
- **Warm-up and repetitions:** 2 warm-up and 5 measured runs per process, 3 independent
  interleaved processes per configuration: 15 measured runs. Calls shorter than 10 ms are
  timed in batches of at least 10 ms (time per call reported).
- **Metrics:** medians of the pooled measured runs; S(p) = T_seq / T_p, E(p) = S(p) / p,
  95 % percentile bootstrap interval of S(p) (2000 resamples), Karp-Flatt metric.
- **Budget:** 15 minutes per run (TIMEOUT) and 16 GiB of matrix storage per process
  (MEMORY_LIMIT). No configuration reached either limit.
- **Memory:** peak working set and peak private bytes of every process
  (`GetProcessMemoryInfo`), and the computed storage 24n² bytes.

## Raw data format (`results/raw_measurements.csv`)

| Column | Description |
|--------|-------------|
| `method` | `ikj`, `blocked` (sequential), `ikj_omp` or `blocked_omp` |
| `build` | `scalar` or `avx2` |
| `n` | Matrix side length |
| `block` | Block size b (blocked methods only) |
| `threads` | Threads of the parallel region (1 for the sequential methods) |
| `dtype` | Numeric type (`float64`) |
| `run_type` | `warmup`, `measured` or `check` (separate first call of a batched configuration) |
| `repetition` | Run index within its type |
| `batch_size` | Calls per timed batch |
| `time_seconds` | Wall-clock time per call (batch time / batch size) |
| `memory_bytes` | Computed storage of A, B and C (24n²) |
| `checksum` | Sum of all elements of C (empty for batched runs) |
| `placement` | `spread`, `compact` or `os` (not pinned) |
| `experiment` | `threads`, `placement` or `hybrid` |
| `process` | Index of the independent process (1–3) |

## Use of generative AI

Claude Opus 5.5 (Anthropic), used through Claude Code, served as a support tool during this
assignment: to help plan the study, write and debug the parallel kernels, the benchmark
program, the experiment runner and the analysis scripts, and to draft and revise the
wording of the report. All the measurements come from running the code on the machine
described above, and every number in the report is produced by
`analysis/report_numbers.py` from the raw data. The author reviewed the code and the
AI-generated text, checked the numbers against the raw data and the references, and is
responsible for the conclusions.
