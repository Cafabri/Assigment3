#include "matmul.h"

/*
 * Sequential baselines, copied unchanged from Assignment 2 (c/matmul.c, commit c40444e).
 * Every kernel adds the n products of C[i][j] in increasing k, starting from C[i][j] = 0,
 * with the same IEEE 754 operations (no fused multiply-add: -ffp-contract=off). They only
 * change the order in which the elements of C are visited, so their results are
 * bit-identical.
 */

/* The triple-loop kernel of Assignment 1; used here only by the correctness tests. */
void multiply_ijk(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i*n + j] += A[i*n + k] * B[k*n + j];
}

/* Loop reordering: j innermost, so B and C are read row by row with unit stride. */
void multiply_ikj(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double a = A[i*n + k];
            for (int j = 0; j < n; j++)
                C[i*n + j] += a * B[k*n + j];
        }
}

static int min_int(int a, int b) { return a < b ? a : b; }

/* Cache blocking of the ikj kernel: bs x bs tiles. The kk loop runs outside the jj loop
   and in increasing order, so the k order of every C[i][j] is still 0, 1, ..., n-1. */
void multiply_blocked(const double *A, const double *B, double *C, int n, int bs) {
    for (int ii = 0; ii < n; ii += bs) {
        int i_end = min_int(ii + bs, n);
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
