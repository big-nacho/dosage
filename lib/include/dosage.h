#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef enum Method {
    FastMBD,
    Dosage,
    Hybrid
} Method;

void dosage(
    size_t w,
    size_t h,
    const uint8_t *colors,
    double *work_color,
    double *work_image,
    double *work_histogram,
    Method method,
    size_t n_iter,
    double sigma,
    size_t boundary_size,
    long n_threads,
    int *exit
);