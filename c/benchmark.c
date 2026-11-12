#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <omp.h>
#include <windows.h>
#include "matmul.h"
#include "parallel.h"

#ifndef BUILD_NAME
#define BUILD_NAME "scalar"
#endif

/* Inputs, timer and batching rule of Assignments 1 and 2 (Cafabri/Assigment2, commit
   c40444e, c/benchmark.c): the same SplitMix64 generator and seeds give the same dense
   matrices, so the checksums can be compared with those of Assignment 2. */
#define SEED_A 42
#define SEED_B 43
/* A first call shorter than BATCH_MIN_SECONDS makes the runs batched: the batch size is
   doubled until a batch lasts at least BATCH_MIN_SECONDS, and the time per call is
   reported. */
#define BATCH_MIN_SECONDS 0.010
#define DEFAULT_MEMORY_LIMIT (16LL << 30)
#define MAX_THREADS 256

/* Monotonic clock in seconds (Windows QueryPerformanceCounter) */
static double now(void) {
    static LARGE_INTEGER freq;
    LARGE_INTEGER t;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}

/* ------------------------------------------------------------------ input matrices */

/* SplitMix64 of Assignment 1 as a counter: the p-th output is computed directly, so
   position p = i*n + j of a matrix always gets the same value. */
static uint64_t splitmix64_at(uint64_t seed, int64_t p) {
    uint64_t z = seed + (uint64_t)(p + 1) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Values in [0, 10): the top 53 bits scaled to [0, 1), times 10 (Assignment 1) */
static double value_at(uint64_t seed, int64_t p) {
    return (double)(splitmix64_at(seed, p) >> 11) * 0x1.0p-53 * 10.0;
}

static void fill_matrix(double *M, int n, uint64_t seed) {
    for (int64_t p = 0; p < (int64_t)n * n; p++)
        M[p] = value_at(seed, p);
}

static double *alloc_matrix(int n) {
    double *M = calloc((size_t)n * n, sizeof(double));
    if (!M) { fprintf(stderr, "Out of memory\n"); exit(2); }
    return M;
}

static double sum_matrix(const double *M, int n) {
    double s = 0.0;
    for (int64_t p = 0; p < (int64_t)n * n; p++)
        s += M[p];
    return s;
}

/* ------------------------------------------------------------------ methods */

typedef enum { M_IKJ, M_BLOCKED, M_IKJ_OMP, M_BLOCKED_OMP } method_t;
static const char *METHOD_NAMES[] = {"ikj", "blocked", "ikj_omp", "blocked_omp"};

static method_t parse_method(const char *s) {
    for (int m = 0; m <= M_BLOCKED_OMP; m++)
        if (strcmp(s, METHOD_NAMES[m]) == 0) return (method_t)m;
    fprintf(stderr, "Unknown method %s\n", s);
    exit(2);
}

static int is_parallel(method_t m) { return m == M_IKJ_OMP || m == M_BLOCKED_OMP; }

typedef struct {
    method_t method;
    int n, bs, threads;
    double *A, *B, *C;
} bench_t;

static void kernel(bench_t *b) {
    switch (b->method) {
    case M_IKJ:         multiply_ikj(b->A, b->B, b->C, b->n); break;
    case M_BLOCKED:     multiply_blocked(b->A, b->B, b->C, b->n, b->bs); break;
    case M_IKJ_OMP:     multiply_ikj_omp(b->A, b->B, b->C, b->n, b->threads); break;
    case M_BLOCKED_OMP: multiply_blocked_omp(b->A, b->B, b->C, b->n, b->bs, b->threads); break;
    }
}

/* Times `batch` consecutive calls. C is zeroed before the batch (outside the timer) and
   accumulates inside it, so every call does the same n^3 multiply-adds (as in
   Assignments 1 and 2). The parallel region is created and joined inside every call. */
static double time_batch(bench_t *b, long batch) {
    memset(b->C, 0, (size_t)b->n * b->n * sizeof(double));
    double t0 = now();
    for (long r = 0; r < batch; r++) kernel(b);
    return now() - t0;
}

/* "0,2,4" -> {0, 2, 4}; "none" -> no per-thread placement */
static int parse_cpus(const char *s, int *cpus) {
    if (strcmp(s, "none") == 0) return 0;
    int count = 0;
    const char *p = s;
    while (*p && count < MAX_THREADS) {
        cpus[count++] = (int)strtol(p, (char **)&p, 10);
        if (*p == ',') p++;
    }
    return count;
}

static int64_t memory_limit(void) {
    const char *s = getenv("MEMORY_LIMIT_BYTES");
    return s ? strtoll(s, NULL, 10) : DEFAULT_MEMORY_LIMIT;
}

/* Creates the inputs outside every timed region, applies the placement and checks (also
   outside the timer) that a parallel region really has the requested number of threads.
   This first region also starts the threads of the OpenMP runtime, so their creation is
   not part of any timed run. A sequential method runs on the main thread, which is OpenMP
   thread 0 of that region, so it is pinned to the same processor as thread 0 of a
   parallel method. */
static void setup(bench_t *b, const char *method, int n, int bs, int threads,
                  const char *cpus_arg) {
    memset(b, 0, sizeof(*b));
    b->method = parse_method(method);
    b->n = n;
    b->bs = bs;
    b->threads = threads;
    if (!is_parallel(b->method) && threads != 1) {
        fprintf(stderr, "A sequential method runs with 1 thread\n");
        exit(2);
    }
    if ((b->method == M_BLOCKED || b->method == M_BLOCKED_OMP) && bs <= 0) {
        fprintf(stderr, "The blocked methods need a block size\n");
        exit(2);
    }
    int64_t need = 3 * (int64_t)n * n * (int64_t)sizeof(double);
    if (need > memory_limit()) {
        fprintf(stderr, "MEMORY_LIMIT: needs %" PRId64 " bytes, limit %" PRId64 "\n", need,
                memory_limit());
        exit(3);
    }
    int cpus[MAX_THREADS], seen[MAX_THREADS];
    set_thread_placement(cpus, parse_cpus(cpus_arg, cpus));
    int team = observe_placement(threads, seen);
    if (team != threads) {
        fprintf(stderr, "Team of %d threads instead of %d\n", team, threads);
        exit(1);
    }
    b->A = alloc_matrix(n);
    b->B = alloc_matrix(n);
    b->C = alloc_matrix(n);
    fill_matrix(b->A, n, SEED_A);
    fill_matrix(b->B, n, SEED_B);
}

static void teardown(bench_t *b) {
    free(b->A); free(b->B); free(b->C);
}

/* benchmark run <method> <n> <block> <threads> <cpus> <warmups> <reps>
   One CSV line per run on stdout:
   method,build,n,block,threads,dtype,run_type,repetition,batch_size,time_seconds,
   memory_bytes,checksum */
static void run(const char *method, int n, int bs, int threads, const char *cpus,
                int warmups, int reps) {
    bench_t b;
    setup(&b, method, n, bs, threads, cpus);
    int64_t mem = 3 * (int64_t)n * n * (int64_t)sizeof(double);
    char prefix[160];
    if (b.method == M_BLOCKED || b.method == M_BLOCKED_OMP)
        snprintf(prefix, sizeof(prefix), "%s,%s,%d,%d,%d,float64", method, BUILD_NAME, n, bs,
                 threads);
    else
        snprintf(prefix, sizeof(prefix), "%s,%s,%d,,%d,float64", method, BUILD_NAME, n,
                 threads);

    /* first call: a check call if it is short (the runs are then batched), otherwise the
       first warm-up run */
    long batch = 1;
    double first = time_batch(&b, 1);
    int r0 = 1;
    if (first < BATCH_MIN_SECONDS) {
        printf("%s,check,1,1,%.9e,%" PRId64 ",%.17g\n", prefix, first, mem,
               sum_matrix(b.C, n));
        /* calibration (not recorded) */
        while (time_batch(&b, batch) < BATCH_MIN_SECONDS)
            batch *= 2;
    } else {
        printf("%s,%s,1,1,%.9e,%" PRId64 ",%.17g\n", prefix,
               warmups > 0 ? "warmup" : "measured", first, mem, sum_matrix(b.C, n));
        r0 = 2;
    }
    fflush(stdout);

    for (int r = r0; r <= warmups + reps; r++) {
        double per_call = time_batch(&b, batch) / batch;
        int warm = r <= warmups;
        int rep = warm ? r : r - warmups;
        if (batch == 1)
            printf("%s,%s,%d,1,%.9e,%" PRId64 ",%.17g\n", prefix,
                   warm ? "warmup" : "measured", rep, per_call, mem, sum_matrix(b.C, n));
        else    /* C holds the sum of `batch` products: no checksum */
            printf("%s,%s,%d,%ld,%.9e,%" PRId64 ",\n", prefix, warm ? "warmup" : "measured",
                   rep, batch, per_call, mem);
        fflush(stdout);
    }
    teardown(&b);
}

/* benchmark verify <method> <n> <block> <threads> <cpus> <file>
   Writes the full result matrix (row-major, one value per line) for the comparison with
   NumPy. */
static void verify(const char *method, int n, int bs, int threads, const char *cpus,
                   const char *path) {
    bench_t b;
    setup(&b, method, n, bs, threads, cpus);
    time_batch(&b, 1);
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(2); }
    for (int64_t p = 0; p < (int64_t)n * n; p++)
        fprintf(f, "%.17g\n", b.C[p]);
    fclose(f);
    teardown(&b);
}

/* benchmark placement <threads> <cpus>
   One line per OpenMP thread: thread number and the logical processor it runs on. */
static void placement_info(int threads, const char *cpus_arg) {
    int cpus[MAX_THREADS], seen[MAX_THREADS];
    set_thread_placement(cpus, parse_cpus(cpus_arg, cpus));
    int team = observe_placement(threads, seen);
    for (int t = 0; t < team; t++)
        printf("%d,%d\n", t, seen[t]);
}

/* ------------------------------------------------------------------ correctness tests */

static void fail(const char *test, const char *method, int threads) {
    fprintf(stderr, "%s failed (%s, %d threads)\n", test, method, threads);
    exit(1);
}

static int same(const double *X, const double *Y, int n) {
    return memcmp(X, Y, (size_t)n * n * sizeof(double)) == 0;
}

/* The sequential kernels and the parallel kernels with several thread counts (including
   more threads than rows, so that some threads get no row) must all give exactly the
   result of ijk. */
static void check_kernels(const double *A, const double *B, int n, int bs,
                          const double *expected, const char *test) {
    static const int THREADS[] = {1, 2, 3, 4, 7, 8, 16, 24, 40};
    double *C = alloc_matrix(n);
    memset(C, 0, (size_t)n * n * sizeof(double));
    multiply_ikj(A, B, C, n);
    if (!same(C, expected, n)) fail(test, "ikj", 1);
    memset(C, 0, (size_t)n * n * sizeof(double));
    multiply_blocked(A, B, C, n, bs);
    if (!same(C, expected, n)) fail(test, "blocked", 1);
    for (size_t t = 0; t < sizeof(THREADS) / sizeof(THREADS[0]); t++) {
        memset(C, 0, (size_t)n * n * sizeof(double));
        multiply_ikj_omp(A, B, C, n, THREADS[t]);
        if (!same(C, expected, n)) fail(test, "ikj_omp", THREADS[t]);
        memset(C, 0, (size_t)n * n * sizeof(double));
        multiply_blocked_omp(A, B, C, n, bs, THREADS[t]);
        if (!same(C, expected, n)) fail(test, "blocked_omp", THREADS[t]);
    }
    free(C);
}

static void check_correctness(void) {
    double A[] = {1,2, 3,4};
    double I[] = {1,0, 0,1};
    double Z[] = {0,0, 0,0};
    double B[] = {5,6, 7,8};
    double AB[] = {19,22, 43,50};
    check_kernels(A, I, 2, 1, A, "Identity test");
    check_kernels(A, Z, 2, 1, Z, "Zero matrix test");
    check_kernels(A, B, 2, 1, AB, "Known result test");

    /* 37 x 37 inputs of the generator; block size 5 leaves partial tiles and row ranges
       that are not multiples of the block size */
    int n = 37;
    double *X = alloc_matrix(n), *Y = alloc_matrix(n), *ref = alloc_matrix(n);
    fill_matrix(X, n, SEED_A);
    fill_matrix(Y, n, SEED_B);
    multiply_ijk(X, Y, ref, n);
    check_kernels(X, Y, n, 5, ref, "Generated 37x37 test");
    free(X); free(Y); free(ref);
}

int main(int argc, char **argv) {
    omp_set_dynamic(0);     /* the runtime may not give a region fewer threads */
    check_correctness();

    if (argc == 9 && strcmp(argv[1], "run") == 0) {
        run(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), argv[6], atoi(argv[7]),
            atoi(argv[8]));
    } else if (argc == 8 && strcmp(argv[1], "verify") == 0) {
        verify(argv[2], atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), argv[6], argv[7]);
    } else if (argc == 4 && strcmp(argv[1], "placement") == 0) {
        placement_info(atoi(argv[2]), argv[3]);
    } else {
        fprintf(stderr,
                "usage: benchmark run <method> <n> <block> <threads> <cpus> <warmups> <reps>\n"
                "       benchmark verify <method> <n> <block> <threads> <cpus> <file>\n"
                "       benchmark placement <threads> <cpus>\n"
                "methods: ikj blocked (sequential, 1 thread) ikj_omp blocked_omp\n"
                "cpus: comma-separated logical processors (thread t -> entry t), or none\n");
        return 2;
    }
    return 0;
}
