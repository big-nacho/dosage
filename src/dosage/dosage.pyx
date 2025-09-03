import numpy as np

cimport cython
cimport numpy as cnp
from libc.stdint cimport uint8_t

cdef extern from 'dosage.h':
    cpdef enum Method:
        FastMBD,
        Dosage,
        Hybrid

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
    )

def detect(
    size_t width,
    size_t height,
    cnp.ndarray[uint8_t, ndim = 2] colors,
    cnp.ndarray[double, ndim = 1] work_color,
    cnp.ndarray[cython.double, ndim = 1] work_image,
    cnp.ndarray[cython.double, ndim = 1] work_histogram,
    Method method,
    size_t n_iter,
    double sigma,
    size_t boundary_size,
    long n_threads
):
    cdef uint8_t[:, :] colors_memoryview = colors
    cdef uint8_t *colors_pointer = &colors_memoryview[0, 0]

    cdef double[:] work_color_memoryview = work_color
    cdef double *work_color_pointer = &work_color_memoryview[0]

    cdef double[:] work_image_memoryview = work_image
    cdef double *work_image_pointer = &work_image_memoryview[0]

    cdef double[:] work_histogram_memoryview = work_histogram
    cdef double *work_histogram_pointer = &work_histogram_memoryview[0]

    cdef int exit

    dosage(
        width,
        height,
        colors_pointer,
        work_color_pointer,
        work_image_pointer,
        work_histogram_pointer,
        method,
        n_iter,
        sigma,
        boundary_size,
        n_threads,
        &exit
    )

    return exit == 0

method_fast_mbd = FastMBD
method_dosage = Dosage
method_hybrid = Hybrid

__all__ = [
    "detect",
    "method_fast_mbd",
    "method_dosage",
    "method_hybrid"
]