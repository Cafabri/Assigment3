#include <omp.h>
#include <windows.h>
#include "parallel.h"

/*
 * Parallelisation of the two sequential kernels of matmul.c.
 *
 * Work division: the n rows of C are split into `threads` contiguous ranges whose sizes
 * differ by at most one row (row_range). Thread t only writes the rows of its range, and
 * A and B are only read, so no two threads write the same element: there is no data
 * race, no lock and no reduction. Inside its range a thread runs exactly the loops of the
 * sequential kernel, so every C[i][j] still receives its n products in increasing k and
 * the result is bit-identical to the sequential kernel for every thread count.
 *
 * Every parallel region requests exactly `threads` threads (num_threads clause) and the
 * caller disables dynamic adjustment, so the team size is the reported thread count.
 */

#define MAX_PLACEMENT 256
static int placement[MAX_PLACEMENT];
static int placement_count = 0;
static __thread int pinned_cpu = -1;    /* processor this OS thread is pinned to */

void set_thread_placement(const int *cpus, int count) {
    if (count > MAX_PLACEMENT) count = MAX_PLACEMENT;
    for (int t = 0; t < count; t++) placement[t] = cpus[t];
    placement_count = count;
}

/* Pins the calling thread (OpenMP thread number t) to its processor. The OpenMP runtime
   reuses its threads between regions, so the system call is only made when the thread
   is not already pinned to the requested processor. */
static void pin_current_thread(int t) {
    if (placement_count == 0) return;
    int cpu = placement[t % placement_count];
    if (pinned_cpu == cpu) return;
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu);
    pinned_cpu = cpu;
}

/* Rows [*i0, *i1) of thread t out of p: the first n % p threads get one extra row. */
static void row_range(int n, int p, int t, int *i0, int *i1) {
    int base = n / p, extra = n % p;
    *i0 = t * base + (t < extra ? t : extra);
    *i1 = *i0 + base + (t < extra);
}

static int min_int(int a, int b) { return a < b ? a : b; }

/* The loops of multiply_ikj restricted to the rows [i0, i1). */
static void ikj_rows(const double *A, const double *B, double *C, int n, int i0, int i1) {
    for (int i = i0; i < i1; i++)
        for (int k = 0; k < n; k++) {
            double a = A[i*n + k];
            for (int j = 0; j < n; j++)
                C[i*n + j] += a * B[k*n + j];
        }
}

/* The tile loops of multiply_blocked restricted to the rows [i0, i1): the row tiles start
   at i0, so a range whose length is not a multiple of bs ends with a shorter tile. */
static void blocked_rows(const double *A, const double *B, double *C, int n, int bs,
                         int i0, int i1) {
    for (int ii = i0; ii < i1; ii += bs) {
        int i_end = min_int(ii + bs, i1);
        for (int kk = 0; kk < n; kk += bs) {
            int k_end = min_int(kk + bs, n);
            for (int jj = 0; jj < n; jj += bs) {
                int j_end = min_int(jj + bs, n);
                for (int i = ii; i < i_end; i++)
                    for (int k = kk; k < k_end; k++) {
                        double a = A[i*n + k];
                        for (int j = jj; j < j_end; j++)
                            C[i*n + j] += a * B[k*n + j];
                    }
            }
        }
    }
}

void multiply_ikj_omp(const double *A, const double *B, double *C, int n, int threads) {
    #pragma omp parallel num_threads(threads)
    {
        int t = omp_get_thread_num(), p = omp_get_num_threads(), i0, i1;
        pin_current_thread(t);
        row_range(n, p, t, &i0, &i1);
        ikj_rows(A, B, C, n, i0, i1);
    }
}

void multiply_blocked_omp(const double *A, const double *B, double *C, int n, int bs,
                          int threads) {
    #pragma omp parallel num_threads(threads)
    {
        int t = omp_get_thread_num(), p = omp_get_num_threads(), i0, i1;
        pin_current_thread(t);
        row_range(n, p, t, &i0, &i1);
        blocked_rows(A, B, C, n, bs, i0, i1);
    }
}

int observe_placement(int threads, int *cpu_of_thread) {
    int team = 0;
    #pragma omp parallel num_threads(threads)
    {
        int t = omp_get_thread_num();
        pin_current_thread(t);
        /* a short busy loop so that the scheduler has placed every thread */
        volatile double x = 0.0;
        for (int r = 0; r < 2000000; r++) x += r;
        (void)x;
        cpu_of_thread[t] = (int)GetCurrentProcessorNumber();
        #pragma omp single
        team = omp_get_num_threads();
    }
    return team;
}
