#ifndef MATMUL_H
#define MATMUL_H

/* Sequential dense kernels reused from Assignment 2 (Cafabri/Assigment2, commit c40444e,
   c/matmul.c): row-major n x n float64 matrices, C must be zeroed by the caller and is
   accumulated into (C += A * B). */
void multiply_ijk(const double *A, const double *B, double *C, int n);
void multiply_ikj(const double *A, const double *B, double *C, int n);
void multiply_blocked(const double *A, const double *B, double *C, int n, int bs);

#endif
