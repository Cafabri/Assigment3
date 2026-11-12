"""Builds the C benchmark (scalar and AVX2 builds) and runs every configuration of the
study under the declared budget.

Usage (from the repository root):  python run_experiment.py
                                   python run_experiment.py --only threads
Options for testing the runner only: --smoke (the smallest sizes, written to
results/smoke/) and --out DIR.

Windows only: the C timer, the processor affinity, the peak-memory measurement and the
hardware queries use Windows APIs. The toolchain path below must point to the local
installation.

Experiments, run in this order:
  threads    Main study and vectorisation extension. Both builds, both kernels (ikj and
             blocked): the sequential kernel and its OpenMP version with p in THREADS,
             n in SIZES. The process runs on the 16 logical processors of the P-cores and
             the threads use the "spread" placement: first one thread per physical P-core
             (0, 2, ..., 14), then the second logical processor of each core (1, 3, ..., 15).
  placement  Extension: "compact" placement (0, 1, 2, ...: two threads per physical core
             before using a new core) against "spread", p in PLACEMENT_THREADS, scalar
             build, n in PLACEMENT_SIZES.
  hybrid     Extension: the process may run on all 24 logical processors (P- and E-cores)
             and the threads are not pinned (placement "os": the Windows scheduler places
             them); p in HYBRID_THREADS, which includes oversubscription (p > 24). Scalar
             build, n in HYBRID_SIZES, with the sequential kernels as their baseline.

Every experiment repeats its configurations in PROCESSES independent processes when the
first one is cheap (interleaved, as in Assignment 2). A run that stops can be resumed:
the (configuration, process) pairs that already have a status are not repeated.

Outputs (results/):
  raw_measurements.csv    every timed run (warm-up, measured and check runs)
  run_status.csv          OK / TIMEOUT / MEMORY_LIMIT / ERROR / SKIPPED per process
  process_memory.csv      peak working set of every benchmark process
  placement.csv           logical processor observed for every OpenMP thread of every
                          (threads, placement) configuration
  vectorisation_report.txt  loops vectorised by gcc in each build (-fopt-info-vec)
  environment.txt         hardware, OS, toolchain, OpenMP settings and protocol
"""
import ctypes
import ctypes.wintypes as wt
import os
import platform
import queue
import statistics
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.join(ROOT, "c")

# Toolchain (adjust this path on another machine)
GCC = r"C:\msys64\mingw64\bin\gcc.exe"
CFLAGS = ["-O2", "-ffp-contract=off", "-fopenmp"]          # Assignment 2 flags + OpenMP
BUILDS = {"scalar": CFLAGS,
          "avx2": CFLAGS + ["-fvect-cost-model=dynamic", "-mavx2"]}
SOURCES = ["matmul.c", "parallel.c", "benchmark.c"]

# gcc needs its own bin directory on PATH to find the assembler, the linker and, at run
# time, libgomp-1.dll
os.environ["PATH"] = os.path.dirname(GCC) + os.pathsep + os.environ["PATH"]

# OpenMP runtime settings of every benchmark process. The thread count and the placement
# are set by the program itself (num_threads clause, SetThreadAffinityMask); OMP_DYNAMIC
# is also disabled in the program. The wait policy is left at the libgomp default.
OMP_ENV = {"OMP_DYNAMIC": "false", "OMP_NUM_THREADS": "1"}
OMP_UNSET = ["OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT",
             "GOMP_CPU_AFFINITY", "OMP_SCHEDULE"]

# Declared before running: sizes, thread counts, repetitions and budget
SIZES = [100, 250, 500, 1000, 2000, 4000]
THREADS = [1, 2, 4, 6, 8, 12, 16]
KERNELS = ["ikj", "blocked"]
# Block sizes chosen in Assignment 2 for each build (results/block_choice.txt, n = 2000)
BLOCK = {"scalar": 32, "avx2": 256}
PLACEMENT_THREADS = [2, 4, 8, 16]
PLACEMENT_SIZES = [1000, 4000]
HYBRID_THREADS = [1, 8, 16, 24, 32, 48]
HYBRID_SIZES = [1000, 4000]

WARMUPS = 2
REPETITIONS = 5                       # measured runs per process
PROCESSES = 3                         # independent processes for cheap configurations
CHEAP_SECONDS = 60                    # cheap = median run of the first process below this
RUN_BUDGET_SECONDS = 15 * 60          # maximum duration of a single run
MEMORY_BUDGET_BYTES = 16 * 2 ** 30    # per process, checked by the program before allocating

HEADER = ("method,build,n,block,threads,dtype,run_type,repetition,batch_size,time_seconds,"
          "memory_bytes,checksum,placement,experiment,process")


def exe(build):
    return os.path.join(C_DIR, f"benchmark_{build}.exe")


# ---------------------------------------------------------------- processor topology

def topology():
    """Physical cores as (efficiency class, [logical processors]), from
    GetLogicalProcessorInformationEx (RelationProcessorCore)."""
    k = ctypes.WinDLL("kernel32")
    size = ctypes.c_ulong(0)
    k.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
    buf = (ctypes.c_ubyte * size.value)()
    if not k.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
        sys.exit("GetLogicalProcessorInformationEx failed")
    cores, off = [], 0
    while off < size.value:
        length = int.from_bytes(bytes(buf[off + 4:off + 8]), "little")
        efficiency = buf[off + 9]                              # PROCESSOR_RELATIONSHIP
        mask = int.from_bytes(bytes(buf[off + 32:off + 40]), "little")  # GroupMask[0].Mask
        cores.append((efficiency, [c for c in range(64) if mask >> c & 1]))
        off += length
    return cores


CORES = topology()
P_CLASS = max(e for e, _ in CORES)
P_CORES = [lps for e, lps in CORES if e == P_CLASS]
E_CORES = [lps for e, lps in CORES if e != P_CLASS]
P_LOGICAL = sorted(c for lps in P_CORES for c in lps)
ALL_LOGICAL = sorted(c for _, lps in CORES for c in lps)

# Thread t runs on entry t of the list
PLACEMENTS = {
    "spread": [lps[0] for lps in P_CORES] + [c for lps in P_CORES for c in lps[1:]],
    "compact": P_LOGICAL,
    "os": None,
}
AFFINITY = {"threads": P_LOGICAL, "placement": P_LOGICAL, "hybrid": ALL_LOGICAL}


def set_runner_affinity(cpus):
    """Restricts this process (and therefore every child it starts) to the given logical
    processors."""
    k = ctypes.WinDLL("kernel32")
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not k.SetProcessAffinityMask(k.GetCurrentProcess(), sum(1 << c for c in cpus)):
        sys.exit("SetProcessAffinityMask failed")


def cpu_list(placement):
    cpus = PLACEMENTS[placement]
    return ",".join(map(str, cpus)) if cpus else "none"


# ------------------------------------------------------------------- peak memory

class _Counters(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD)] + [
        (f, ctypes.c_size_t) for f in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
            "PagefileUsage", "PeakPagefileUsage")]


_psapi = ctypes.WinDLL("psapi")
_psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_Counters), wt.DWORD]
PEAK_METRIC = "peak working set (GetProcessMemoryInfo)"


def wait_with_peak(proc):
    """Waits for the process and returns its peak working set and peak private bytes,
    read through the process handle after the process has exited."""
    proc.wait()
    c = _Counters()
    c.cb = ctypes.sizeof(c)
    ok = _psapi.GetProcessMemoryInfo(int(proc._handle), ctypes.byref(c), c.cb)
    return (c.PeakWorkingSetSize, c.PeakPagefileUsage) if ok else (None, None)


# ------------------------------------------------------------------- environment

def run_text(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True)
    except OSError:
        return ""
    return (out.stdout.decode("oem", "replace") + out.stderr.decode("oem", "replace")).strip()


def first_line(cmd):
    text = run_text(cmd)
    return text.splitlines()[0] if text else "not found"


def hardware():
    ps = ("$p = Get-CimInstance Win32_Processor; $m = Get-CimInstance Win32_ComputerSystem;"
          "\"$($p.Name)|$($p.NumberOfCores)|$($p.NumberOfLogicalProcessors)|"
          "$($p.L2CacheSize)|$($p.L3CacheSize)|$($p.MaxClockSpeed)|$($m.TotalPhysicalMemory)\"")
    name, cores, logical, l2, l3, clock, ram = run_text(
        ["powershell", "-NoProfile", "-Command", ps]).split("|")
    return [f"cpu: {name.strip()}",
            f"cores: {cores} physical, {logical} logical",
            f"P-cores: {len(P_CORES)} physical with SMT, logical processors "
            + " ".join("-".join(map(str, lps)) for lps in P_CORES),
            f"E-cores: {len(E_CORES)} physical without SMT, logical processors "
            + " ".join(str(lps[0]) for lps in E_CORES),
            f"cache: L2 {int(l2) // 1024} MB total, L3 {int(l3) // 1024} MB",
            f"base clock reported by Win32_Processor: {clock} MHz",
            f"ram: {int(ram) / 2**30:.1f} GiB"]


def power_plan():
    text = run_text(["powercfg", "/getactivescheme"])
    return "power plan: " + (text.split(":", 1)[-1].strip() if text else "unknown")


def library_versions():
    versions = []
    for mod in ("numpy", "pandas", "matplotlib"):
        try:
            versions.append(f"{mod} {__import__(mod).__version__}")
        except ImportError:
            versions.append(f"{mod} not installed")
    return ", ".join(versions)


def save_environment(out, sizes):
    lines = [f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"os: {platform.platform()}"] + hardware() + [
        power_plan(),
        f"c compiler: {first_line([GCC, '--version'])} (OpenMP runtime: libgomp of the same "
        f"gcc)",
        *[f"build {name}: {' '.join(flags)}" for name, flags in BUILDS.items()],
        f"block size: {BLOCK} (chosen in Assignment 2)",
        f"python: {platform.python_implementation()} {platform.python_version()} (runner and "
        f"analysis only)",
        f"analysis libraries: {library_versions()}",
        f"openmp environment: {OMP_ENV}; unset: {OMP_UNSET}; omp_set_dynamic(0) in the "
        f"program; team size checked before timing",
        f"process affinity: threads/placement experiments {P_LOGICAL[0]}-{P_LOGICAL[-1]} "
        f"(P-cores), hybrid experiment {ALL_LOGICAL[0]}-{ALL_LOGICAL[-1]} (all)",
        *[f"placement {name}: {cpu_list(name)}" for name in PLACEMENTS],
        f"peak memory metric: {PEAK_METRIC}; also peak private bytes (PeakPagefileUsage)",
        f"threads experiment: n = {sizes['threads']}, p = {THREADS}, kernels {KERNELS}, "
        f"builds {list(BUILDS)}, placement spread",
        f"placement experiment: n = {sizes['placement']}, p = {PLACEMENT_THREADS}, scalar, "
        f"compact (spread from the threads experiment)",
        f"hybrid experiment: n = {sizes['hybrid']}, p = {HYBRID_THREADS}, scalar, placement os",
        f"warm-ups: {WARMUPS}; measured runs per process: {REPETITIONS}",
        f"processes: {PROCESSES} interleaved when the first process has a median below "
        f"{CHEAP_SECONDS} s, otherwise 1",
        "batching: a first call shorter than 10 ms is a separate check call, and the runs "
        "are timed in batches lasting >= 10 ms (time per call reported)",
        f"budget: {RUN_BUDGET_SECONDS} s per run, {MEMORY_BUDGET_BYTES / 2**30:.0f} GiB of "
        f"matrix storage per process",
    ]
    with open(os.path.join(out, "environment.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


# ------------------------------------------------------------------- building

def build(out):
    """Compiles both builds and records which loops of the kernels gcc vectorised in each
    of them, and the loops of the kernels it reports it could not vectorise."""
    report = []
    for name, flags in BUILDS.items():
        cmd = [GCC, *flags, f'-DBUILD_NAME="{name}"', "-fopt-info-vec-optimized",
               "-o", exe(name), *SOURCES]
        res = subprocess.run(cmd, cwd=C_DIR, capture_output=True, text=True)
        if res.returncode != 0:
            sys.exit(res.stderr)
        kernels = [l for l in res.stderr.splitlines() if l.startswith(("matmul.c", "parallel.c"))]
        report += [f"== build {name}: gcc {' '.join(flags)}",
                   "-fopt-info-vec-optimized (loops vectorised):",
                   *(kernels or ["(no loop of matmul.c or parallel.c vectorised)"])]
        missed = []
        for src in ("matmul.c", "parallel.c"):
            res = subprocess.run([GCC, *flags, "-fopt-info-vec-missed", "-c", "-o", os.devnull,
                                  src], cwd=C_DIR, capture_output=True, text=True)
            missed += [l for l in res.stderr.splitlines() if "couldn't vectorize loop" in l]
        report += ["-fopt-info-vec-missed (loops not vectorised):", *missed, ""]
    with open(os.path.join(out, "vectorisation_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("\n".join(report), flush=True)


# ------------------------------------------------------------------- running

def child_env():
    env = dict(os.environ, MEMORY_LIMIT_BYTES=str(MEMORY_BUDGET_BYTES), **OMP_ENV)
    for var in OMP_UNSET:
        env.pop(var, None)
    return env


def run_process(cmd):
    """Runs one benchmark process, killing it when a single run exceeds the budget.

    Returns (status, detail, wall seconds, output lines, peak working set, peak private)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd=ROOT, env=child_env())
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in proc.stdout] + [lines.put(None)],
                     daemon=True).start()

    out = []
    start = last = time.monotonic()
    while True:
        try:
            line = lines.get(timeout=max(0.0, RUN_BUDGET_SECONDS - (time.monotonic() - last)))
        except queue.Empty:
            proc.kill()
            proc.wait()
            return ("TIMEOUT", f"a run exceeded {RUN_BUDGET_SECONDS} s",
                    time.monotonic() - start, out, None, None)
        if line is None:
            break
        out.append(line.strip())
        last = time.monotonic()
        print("   ", line.strip(), flush=True)

    peak, private = wait_with_peak(proc)
    wall = time.monotonic() - start
    detail = proc.stderr.read().strip().replace("\n", " ").replace(",", ";")
    if proc.returncode == 3:
        return "MEMORY_LIMIT", detail, wall, out, peak, private
    if proc.returncode != 0:
        return "ERROR", f"exit code {proc.returncode}: {detail}", wall, out, peak, private
    return "OK", "", wall, out, peak, private


class Config:
    def __init__(self, experiment, method, build, n, threads, placement):
        self.experiment, self.method, self.build = experiment, method, build
        self.n, self.threads, self.placement = n, threads, placement
        self.block = BLOCK[build] if method.startswith("blocked") else 0

    def series(self):
        """Configurations that differ only in n; a failure stops the larger sizes."""
        return (self.experiment, self.method, self.build, self.threads, self.placement)

    def label(self):
        return (f"{self.experiment},{self.method},{self.build},{self.n},"
                f"{self.block or ''},{self.threads},{self.placement}")

    def command(self):
        return [exe(self.build), "run", self.method, str(self.n), str(self.block),
                str(self.threads), cpu_list(self.placement), str(WARMUPS), str(REPETITIONS)]


STATUS_HEADER = "experiment,method,build,n,block,threads,placement,process,status,detail,wall_seconds"
MEMORY_HEADER = ("experiment,method,build,n,block,threads,placement,process,peak_bytes,"
                 "peak_private_bytes,metric")


def raw_key(line):
    """(configuration label, process) of a line of raw_measurements.csv"""
    c = line.split(",")
    return f"{c[13]},{c[0]},{c[1]},{c[2]},{c[3]},{c[4]},{c[12]}", c[14]


def status_key(line):
    """(configuration label, process) of a line of run_status.csv or process_memory.csv"""
    c = line.split(",")
    return ",".join(c[:7]), c[7]


class Runner:
    def __init__(self, out):
        self.out = out
        self.done = {}      # (label, process) -> status of the processes already run
        self.times = {}     # (label, process) -> their measured times
        files = {"raw_measurements.csv": HEADER, "run_status.csv": STATUS_HEADER,
                 "process_memory.csv": MEMORY_HEADER}
        resume = os.path.exists(os.path.join(out, "run_status.csv"))
        if resume:
            self.resume()
        self.raw, self.status, self.memory = (
            open(os.path.join(out, name), "a" if resume else "w", newline="") for name in files)
        if not resume:
            for f, header in zip((self.raw, self.status, self.memory), files.values()):
                f.write(header + "\n")

    def resume(self):
        """Reads the processes that already have a status, and removes the raw and memory
        lines of a process that was interrupted before its status was written."""
        with open(os.path.join(self.out, "run_status.csv")) as f:
            for line in f.read().splitlines()[1:]:
                self.done[status_key(line)] = line.split(",")[8]
        dropped = 0
        for name, header, key in [("raw_measurements.csv", HEADER, raw_key),
                                  ("process_memory.csv", MEMORY_HEADER, status_key)]:
            lines = []
            if os.path.exists(os.path.join(self.out, name)):
                with open(os.path.join(self.out, name)) as f:
                    lines = [l for l in f.read().splitlines()[1:] if l]
            kept = [l for l in lines if key(l) in self.done]
            dropped += len(lines) - len(kept)
            with open(os.path.join(self.out, name), "w", newline="") as f:
                f.write("\n".join([header] + kept) + "\n")
            if name == "raw_measurements.csv":
                for l in kept:
                    if l.split(",")[6] == "measured":
                        self.times.setdefault(raw_key(l), []).append(float(l.split(",")[9]))
        print(f"resuming: {len(self.done)} processes already have a status; {dropped} lines "
              f"of interrupted processes removed", flush=True)

    def benchmark(self, cfg, process):
        """Runs one process; returns its status and the median of its measured runs."""
        key = (cfg.label(), str(process))
        if key in self.done:
            times = self.times.get(key)
            return self.done[key], statistics.median(times) if times else None
        print(f"{cfg.label()} process={process}", flush=True)
        status, detail, wall, out, peak, private = run_process(cfg.command())
        for line in out:
            self.raw.write(f"{line},{cfg.placement},{cfg.experiment},{process}\n")
        self.status.write(f"{cfg.label()},{process},{status},{detail},{wall:.1f}\n")
        if peak is not None and status == "OK":
            self.memory.write(f"{cfg.label()},{process},{peak},{private},{PEAK_METRIC}\n")
        for f in (self.raw, self.status, self.memory):
            f.flush()
        times = [float(l.split(",")[9]) for l in out if l.split(",")[6] == "measured"]
        return status, statistics.median(times) if times else None

    def skip(self, cfg):
        if (cfg.label(), "1") in self.done:
            return
        self.status.write(f"{cfg.label()},1,SKIPPED,larger than a size over budget,,\n")
        self.status.flush()

    def run_all(self, configs):
        """First process of every configuration in order (a failure skips the larger sizes
        of its series), then the further processes of the cheap ones, interleaved."""
        failed, cheap = set(), []
        for cfg in configs:
            if cfg.series() in failed:
                self.skip(cfg)
                continue
            status, median = self.benchmark(cfg, 1)
            if status != "OK":
                failed.add(cfg.series())
            elif median is not None and median < CHEAP_SECONDS:
                cheap.append(cfg)
        for process in range(2, PROCESSES + 1):
            for cfg in cheap:
                self.benchmark(cfg, process)


def save_placement(out, experiment, pairs):
    """Logical processor observed for every OpenMP thread of each (threads, placement)."""
    path = os.path.join(out, "placement.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        if new:
            f.write("experiment,placement,threads,thread,cpu,requested_cpu\n")
        for p, placement in pairs:
            res = subprocess.run([exe("scalar"), "placement", str(p), cpu_list(placement)],
                                 capture_output=True, text=True, check=True, env=child_env())
            cpus = PLACEMENTS[placement]
            for line in res.stdout.split():
                t, cpu = map(int, line.split(","))
                f.write(f"{experiment},{placement},{p},{t},{cpu},"
                        f"{cpus[t % len(cpus)] if cpus else ''}\n")


# ------------------------------------------------------------------- experiments

def threads_configs(sizes):
    configs = []
    for build in BUILDS:
        for n in sizes:
            for kernel in KERNELS:
                configs.append(Config("threads", kernel, build, n, 1, "spread"))
                configs += [Config("threads", f"{kernel}_omp", build, n, p, "spread")
                            for p in THREADS]
    return configs


def placement_configs(sizes):
    return [Config("placement", f"{kernel}_omp", "scalar", n, p, "compact")
            for n in sizes for kernel in KERNELS for p in PLACEMENT_THREADS]


def hybrid_configs(sizes):
    configs = []
    for n in sizes:
        for kernel in KERNELS:
            configs.append(Config("hybrid", kernel, "scalar", n, 1, "os"))
            configs += [Config("hybrid", f"{kernel}_omp", "scalar", n, p, "os")
                        for p in HYBRID_THREADS]
    return configs


ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def keep_awake(on):
    """Stops Windows from sleeping while the study runs (on) or restores the default."""
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))


def main(args):
    smoke = "--smoke" in args
    out = os.path.join(ROOT, "results", "smoke") if smoke else os.path.join(ROOT, "results")
    if "--out" in args:
        out = os.path.abspath(args[args.index("--out") + 1])
    only = args[args.index("--only") + 1] if "--only" in args else None
    sizes = ({"threads": [100, 250], "placement": [250], "hybrid": [250]} if smoke else
             {"threads": SIZES, "placement": PLACEMENT_SIZES, "hybrid": HYBRID_SIZES})
    experiments = [("threads", threads_configs, [(p, "spread") for p in THREADS]),
                   ("placement", placement_configs, [(p, "compact") for p in PLACEMENT_THREADS]),
                   ("hybrid", hybrid_configs, [(p, "os") for p in HYBRID_THREADS])]

    os.makedirs(out, exist_ok=True)
    keep_awake(True)
    try:
        start = time.monotonic()
        build(out)
        save_environment(out, sizes)
        runner = Runner(out)
        for name, make, pairs in experiments:
            if only and name != only:
                continue
            set_runner_affinity(AFFINITY[name])
            if not any(k[0].startswith(name + ",") for k in runner.done):
                save_placement(out, name, pairs)
            runner.run_all(make(sizes[name]))
        print(f"finished in {(time.monotonic() - start) / 60:.1f} min", flush=True)
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main(sys.argv)
