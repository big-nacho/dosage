#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

typedef enum Method {
    FastMBD,
    Dosage,
    Hybrid
} Method;

typedef enum Boundary {
    BoundaryTop,
    BoundaryRight,
    BoundaryBottom,
    BoundaryLeft,
    BoundaryNone
} Boundary;

void dosage(
    size_t w,
    size_t h,
    const uint8_t *colors,
    double *work_color,
    double *work_image,
    double *work_histogram,
    Method method,
    double sigma,
    size_t boundary_thickness,
    Boundary foreground_boundary,
    size_t n_passes,
    int *exit
);