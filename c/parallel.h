#ifndef PARALLEL_H
#define PARALLEL_H

/* OpenMP versions of the sequential kernels of matmul.c. They run the same loops on the
   same row-major float64 matrices; the rows of C are divided among `threads` threads.
   C must be zeroed by the caller and is accumulated into (C += A * B). */
void multiply_ikj_omp(const double *A, const double *B, double *C, int n, int threads);
void multiply_blocked_omp(const double *A, const double *B, double *C, int n, int bs,
                          int threads);

/* Thread placement: OpenMP thread t of every parallel region runs on logical processor
   cpus[t % count]. count = 0 disables the per-thread placement (the threads may then run
   on any processor of the process affinity mask). */
void set_thread_placement(const int *cpus, int count);

/* Logical processor on which each of the `threads` threads of a parallel region runs
   (written to cpu_of_thread[0 .. threads-1]); returns the team size. */
int observe_placement(int threads, int *cpu_of_thread);

#endif
