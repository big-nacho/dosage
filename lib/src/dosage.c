#include <printf.h>
#include <pthread.h>
#include <unistd.h>

#include "dosage.h"

// ========== Constants ==========
static const size_t bin_count = 32; // Number of bins for histograms
static const long threads_max = 32; // Max number of threads to use
// ===============================

// ========== Globals ==========
static size_t width; // Image width
static size_t height; // Image height
static size_t color_count; // Color count (width * height)
static double kernel[63]; // Convolution kernel buffer
static size_t kernel_radius; // Convolution kernel radius
// ===============================

// ========== Indexing macros ==========
// Index [x, cols] matrix.
#define index_matrix(array, cols, row, col) ((array)[((row) * (cols)) + (col)])

// Index [height * width, 3] color list.
#define index_colors(array, row, c) (index_matrix(array, 3, row, c))

// Index [height, width, 3] image.
#define index_image(array, y, x, c) (index_matrix(array, 3, y * width + x, c))

// Index [height, width] 2D map.
#define index_map(array, y, x) (index_matrix(array, width, y, x)) // Read a [height, width] map

// Index 3D histogram
#define index_histogram(histogram, x, y, z) (histogram[x * bin_count * bin_count + y * bin_count + z])
// ===============================

// Scale map values to be in the [0, 1] range
static void normalize_map(double *map) {
    double min = map[0];
    double max = map[0];
    for (size_t i = 0; i < color_count; i++) {
        min = fmin(min, map[i]);
        max = fmax(max, map[i]);
    }

    double d = max - min;

    if (d < 1e-8) {
        // Don't think we care about this case
        for (size_t i = 0; i < color_count; i++) {
            map[i] = 0;
        }
        return;
    }

    for (size_t i = 0; i < color_count; i++) {
        map[i] = (map[i] - min) / d;
    }
}

long get_thread_count(long requested) {
    long count = requested;

    if (requested <= 0) {
        // Auto detect.
        long n_threads = sysconf(_SC_NPROCESSORS_ONLN);
        count = n_threads;
    }

    if (count < 7) {
        // We need a minimum of 7 threads even if not physical.
        // This also catches sysconf failures (-1).
        count = 7;
    }

    if (count > threads_max) {
        return threads_max;
    }

    return count;
}

// Fast MBD raster scan
static void raster_scan(
    const double *image,
    size_t channel, // Channel to read while scanning (0 -> R, 1 -> G, 2 -> B)
    double *D,
    double *U,
    double *L
) {
    for (size_t y = 1; y < height - 1; y++) {
        for (size_t x = 1; x < width - 1; x++) {
            double v = index_image(image, y, x, channel);
            double d = index_map(D, y, x);
            double u1 = index_map(U, y - 1, x);
            double l1 = index_map(L, y - 1, x);
            double u2 = index_map(U, y, x - 1);
            double l2 = index_map(L, y, x - 1);
            double b1 = fmax(u1, v) - fmin(l1, v);
            double b2 = fmax(u2, v) - fmin(l2, v);

            if (d <= b1 && d <= b2) {
                continue;
            }

            else if (b1 < d && b1 <= b2) {
                index_map(D, y, x) = b1;
                index_map(U, y, x) = fmax(u1, v);
                index_map(L, y, x) = fmin(l1, v);
            }

            else {
                index_map(D, y, x) = b2;
                index_map(U, y, x) = fmax(u2, v);
                index_map(L, y, x) = fmin(l2, v);
            }
        }
    }
}

// Fast MBD raster scan (inverse)
static void raster_scan_inverse(
    const double *image,
    size_t channel, // Channel to read while scanning (0 -> R, 1 -> G, 2 -> B)
    double *D,
    double *U,
    double *L
) {
    for (size_t y = height - 2; y > 1; y--) {
        for (size_t x = width - 2; x > 1; x--) {
            double v = index_image(image, y, x, channel);
            double d = index_map(D, y, x);
            double u1 = index_map(U, y + 1, x);
            double l1 = index_map(L, y + 1, x);
            double u2 = index_map(U, y, x + 1);
            double l2 = index_map(L, y, x + 1);
            double b1 = fmax(u1, v) - fmin(l1, v);
            double b2 = fmax(u2, v) - fmin(l2, v);

            if (d <= b1 && d <= b2) {
                continue;
            }

            else if (b1 < d && b1 <= b2) {
                index_map(D, y, x) = b1;
                index_map(U, y, x) = fmax(u1, v);
                index_map(L, y, x) = fmin(l1, v);
            }

            else {
                index_map(D, y, x) = b2;
                index_map(U, y, x) = fmax(u2, v);
                index_map(L, y, x) = fmin(l2, v);
            }
        }
    }
}

static double *fast_mbd(
    const double *image,
    size_t channel, // Channel to read while scanning (0 -> R, 1 -> G, 2 -> B)
    size_t iter,
    double *D,
    double *U,
    double *L
) {
    for (size_t i = 0; i < color_count; i++) {
        U[i] = L[i] = index_colors(image, i, channel);
    }

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            index_map(D, y, x) = INFINITY;
        }
    }

    for (size_t x = 0; x < width; x++) {
        index_matrix(D, width, 0, x) = 0;
        index_matrix(D, width, height - 1, x) = 0;
    }

    for (size_t y = 0; y < height; y++) {
        index_matrix(D, width, y, 0) = 0;
        index_matrix(D, width, y, width - 1) = 0;
    }

    for (size_t i = 0; i < iter; i++) {
        if (i % 2 == 0) {
            raster_scan(
                image,
                channel,
                D,
                U,
                L
            );
        }
        else {
            raster_scan_inverse(
                image,
                channel,
                D,
                U,
                L
            );
        }
    }

    return D;
}

typedef struct FastMBDPayload {
    const double *image;
    size_t channel;
    size_t iter;
    double *D;
    double *U;
    double *L;
} FastMBDPayload;

void* process_fast_mbd(void* data) {
    FastMBDPayload *payload = (FastMBDPayload *)data;
    fast_mbd(
        payload->image,
        payload->channel,
        payload->iter,
        payload->D,
        payload->U,
        payload->L
    );
    return NULL;
}

// Convert RGB color to histogram space.
// Scales values from [0, 255] to [0, bin_count = 32].
void RGB_to_histogram_space(
    double R,
    double G,
    double B,
    double *hist_R,
    double *hist_G,
    double *hist_B
) {
    double histogram_dx = 255.0 / (double)bin_count;
    *hist_R = R / histogram_dx;
    *hist_G = G / histogram_dx;
    *hist_B = B / histogram_dx;
}

// Gets the histogram index of a color in histogram space.
void histogram_space_to_histogram_bin(
    double hist_R,
    double hist_G,
    double hist_B,
    size_t *bin_R,
    size_t *bin_G,
    size_t *bin_B
) {
    *bin_R = (size_t)hist_R;
    *bin_G = (size_t)hist_G;
    *bin_B = (size_t)hist_B;
    *bin_R = *bin_R < bin_count ? *bin_R : bin_count - 1;
    *bin_G = *bin_G < bin_count ? *bin_G : bin_count - 1;
    *bin_B = *bin_B < bin_count ? *bin_B : bin_count - 1;
}

// Gets the histogram bin of an RGB color.
void RGB_to_histogram_bin(
    double R,
    double G,
    double B,
    size_t *bin_R,
    size_t *bin_G,
    size_t *bin_B
) {
    double hist_R, hist_G, hist_B;
    RGB_to_histogram_space(R, G, B, &hist_R, &hist_G, &hist_B);
    histogram_space_to_histogram_bin(hist_R, hist_G, hist_B, bin_R, bin_G, bin_B);
}

// Populates a histogram based on a sub-rectangle of the image.
void populate_histogram(
    const double *image,
    size_t low_x,
    size_t high_x, // Non-inclusive
    size_t low_y,
    size_t high_y, // Non-inclusive
    double *histogram
) {
    size_t volume = bin_count * bin_count * bin_count;

    // Zero-out histogram.
    memset(
        histogram,
        0,
        volume * sizeof(double)
    );

    for (size_t y = low_y; y < high_y; y++) {
        for (size_t x = low_x; x < high_x; x++) {
            double R = index_image(image, y, x, 0);
            double G = index_image(image, y, x, 1);
            double B = index_image(image, y, x, 2);

            size_t bin_R, bin_G, bin_B;
            RGB_to_histogram_bin(R, G, B, &bin_R, &bin_G, &bin_B);
            index_histogram(histogram, bin_R, bin_G, bin_B)++;
        }
    }

    // Convert histogram values to probabilities
    double sum = 0;
    for (size_t i = 0; i < volume; i++) {
        sum += histogram[i];
    }

    for (size_t i = 0; i < volume; i++) {
        histogram[i] /= sum;
    }
}

void convolve(
    const double *histogram,
    size_t axis,
    double *convolved // On exit, contains the result
) {
    ssize_t radius = (ssize_t)kernel_radius;
    for (ssize_t x = 0; x < bin_count; x++) {
        for (ssize_t y = 0; y < bin_count; y++) {
            for (ssize_t z = 0; z < bin_count; z++) {

                double sum = 0;
                for (ssize_t o = -radius; o <= radius; o++) {
                    double v;

                    if (axis == 0) {
                        ssize_t xx = x + o;
                        if (xx < 0 || xx >= bin_count) {
                            continue;
                        }
                        v = index_histogram(histogram, xx, y, z);
                    }

                    if (axis == 1) {
                        ssize_t yy = y + o;
                        if (yy < 0 || yy >= bin_count) {
                            continue;
                        }
                        v = index_histogram(histogram, x, yy, z);
                    }

                    if (axis == 2) {
                        ssize_t zz = z + o;
                        if (zz < 0 || zz >= bin_count) {
                            continue;
                        }
                        v = index_histogram(histogram, x, y, zz);
                    }

                    double weight = kernel[o + radius];
                    sum += (double)v * weight;
                }

                index_histogram(convolved, x, y, z) = sum;
            }
        }
    }
}

double trilinear_interpolate(
    const double *histogram,
    double x,
    double y,
    double z
) {
    double fx, fy, fz;
    size_t i0, j0, k0;
    RGB_to_histogram_space(x, y, z, &fx, &fy, &fz);
    histogram_space_to_histogram_bin(fx, fy, fz, &i0, &j0, &k0);

    double tx = fx - (double)i0;
    double ty = fy - (double)j0;
    double tz = fz - (double)k0;

    size_t i1 = i0 < bin_count - 1 ? i0 + 1 : i0;
    size_t j1 = j0 < bin_count - 1 ? j0 + 1 : j0;
    size_t k1 = k0 < bin_count - 1 ? k0 + 1 : k0;

    double c000 = index_histogram(histogram, i0, j0, k0);
    double c010 = index_histogram(histogram, i0, j1, k0);
    double c001 = index_histogram(histogram, i0, j0, k1);
    double c011 = index_histogram(histogram, i0, j1, k1);
    double c100 = index_histogram(histogram, i1, j0, k0);
    double c110 = index_histogram(histogram, i1, j1, k0);
    double c101 = index_histogram(histogram, i1, j0, k1);
    double c111 = index_histogram(histogram, i1, j1, k1);

    double c00 = c000 * (1 - tx) + c100 * tx;
    double c01 = c001 * (1 - tx) + c101 * tx;
    double c10 = c010 * (1 - tx) + c110 * tx;
    double c11 = c011 * (1 - tx) + c111 * tx;

    double c0 = c00 * (1 - ty) + c10 * ty;
    double c1 = c01 * (1 - ty) + c11 * ty;

    return c0 * (1 - tz) + c1 * tz;
}

void histogram_to_saliency(
    const double *image,
    const double *histogram,
    double *saliency
) {
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            double cx = index_image(image, y, x, 0);
            double cy = index_image(image, y, x, 1);
            double cz = index_image(image, y, x, 2);
            size_t binLt, binUt, binVt;
            RGB_to_histogram_bin(cx, cy, cz, &binLt, &binUt, &binVt);
            index_map(saliency, y, x) = trilinear_interpolate(histogram, cx, cy, cz);
        }
    }

    for (size_t i = 0; i < color_count; i++) {
        saliency[i] = -log(saliency[i] + 1e-8);
    }

    normalize_map(saliency);
}

typedef struct DosagePayload {
    double *aux;
    double *histogram;
    double *image;
    size_t low_x;
    size_t high_x;
    size_t low_y;
    size_t high_y;
    double *result;
} DosagePayload;

void *process_dosage(void *data) {
    DosagePayload *payload = (DosagePayload *)data;
    double *aux = payload->aux;
    double *histogram = payload->histogram;
    double *image = payload->image;
    size_t low_x = payload->low_x;
    size_t high_x = payload->high_x;
    size_t low_y = payload->low_y;
    size_t high_y = payload->high_y;
    double *result = payload->result;

    size_t volume = bin_count * bin_count * bin_count;
    populate_histogram(image, low_x, high_x, low_y, high_y, histogram);
    memcpy(aux, histogram, sizeof(double) * volume);
    convolve(aux, 0, histogram);
    memcpy(aux, histogram, sizeof(double) * volume);
    convolve(aux, 1, histogram);
    memcpy(aux, histogram, sizeof(double) * volume);
    convolve(aux, 2, histogram);
    histogram_to_saliency(image, histogram, result);
    return NULL;
}

void create_convolution_kernel(double sigma) {
    kernel_radius = (size_t)(0.5 + 4.0 * sigma); // Kernel covers ~8 sigma

    if (kernel_radius > 31) {
        // Cap kernel radius at 31.
        // This means after a sigma of ~8 tails start getting lost.
        kernel_radius = 31;
    }

    double sum = 0;
    size_t length = 2 * kernel_radius + 1;
    for (size_t i = 0; i < length; i++) {
        double x = ((double)i - (double)kernel_radius);
        kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }

    for (size_t i = 0; i < length; i++) {
        kernel[i] /= sum;
    }
}

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
) {
    *exit = 0;

    width = w;
    height = h;
    color_count = w * h;

    size_t thread_count = get_thread_count(n_threads);
    pthread_t threads[thread_count];

    for (size_t i = 0; i < color_count * 3; i++) {
        work_color[i] = (double)colors[i];
    }

    pthread_t t1 = threads[0];
    pthread_t t2 = threads[1];
    pthread_t t3 = threads[2];
    pthread_t t4 = threads[3];
    pthread_t t5 = threads[4];
    pthread_t t6 = threads[5];
    pthread_t t7 = threads[6];

    FastMBDPayload L_payload;
    FastMBDPayload U_payload;
    FastMBDPayload V_payload;

    bool perform_fast_mbd = method != Dosage;
    bool perform_dosage = method != FastMBD;

    if (perform_fast_mbd) {
        L_payload.image = work_color;
        L_payload.D = work_image;
        L_payload.U = L_payload.D + color_count;
        L_payload.L = L_payload.U + color_count;
        L_payload.channel = 0;
        L_payload.iter = n_iter;

        U_payload.image = work_color;
        U_payload.D = L_payload.L + color_count;
        U_payload.U = U_payload.D + color_count;
        U_payload.L = U_payload.U + color_count;
        U_payload.channel = 1;
        U_payload.iter = n_iter;

        V_payload.image = work_color;
        V_payload.D = U_payload.L + color_count;
        V_payload.U = V_payload.D + color_count;
        V_payload.L = V_payload.U + color_count;
        V_payload.channel = 2;
        V_payload.iter = n_iter;

        // Process FastMBD in parallel
        if (pthread_create(&t1, NULL, process_fast_mbd, &L_payload) != 0) {
            *exit = 1;
            return;
        }

        if (pthread_create(&t2, NULL, process_fast_mbd, &U_payload) != 0) {
            *exit = 1;
            return;
        }

        if (pthread_create(&t3, NULL, process_fast_mbd, &V_payload) != 0) {
            *exit = 1;
            return;
        }
    }

    DosagePayload top_payload;
    DosagePayload right_payload;
    DosagePayload bottom_payload;
    DosagePayload left_payload;

    if (perform_dosage) {
        create_convolution_kernel(sigma);

        size_t volume = bin_count * bin_count * bin_count;

        top_payload.aux = work_histogram;
        top_payload.histogram = top_payload.aux + volume;
        top_payload.image = work_color;
        top_payload.low_x = 0;
        top_payload.high_x = width;
        top_payload.low_y = 0;
        top_payload.high_y = boundary_size;
        top_payload.result = method == Hybrid ? V_payload.L + color_count : work_image;

        right_payload.aux = top_payload.histogram + volume;
        right_payload.histogram = right_payload.aux + volume;
        right_payload.image = work_color;
        right_payload.low_x = width - boundary_size;
        right_payload.high_x = width;
        right_payload.low_y = 0;
        right_payload.high_y = height;
        right_payload.result = top_payload.result + color_count;

        bottom_payload.aux = right_payload.histogram + volume;
        bottom_payload.histogram = bottom_payload.aux + volume;
        bottom_payload.image = work_color;
        bottom_payload.low_x = 0;
        bottom_payload.high_x = width;
        bottom_payload.low_y = height - boundary_size;
        bottom_payload.high_y = height;
        bottom_payload.result = right_payload.result + color_count;

        left_payload.aux = bottom_payload.histogram + volume;
        left_payload.histogram = left_payload.aux + volume;
        left_payload.image = work_color;
        left_payload.low_x = 0;
        left_payload.high_x = boundary_size;
        left_payload.low_y = 0;
        left_payload.high_y = height;
        left_payload.result = bottom_payload.result + color_count;

        if (pthread_create(&t4, NULL, process_dosage, &top_payload) != 0) {
            *exit = 1;
            return;
        }

        if (pthread_create(&t5, NULL, process_dosage, &right_payload) != 0) {
            *exit = 1;
            return;
        }

        if (pthread_create(&t6, NULL, process_dosage, &bottom_payload) != 0) {
            *exit = 1;
            return;
        }

        if (pthread_create(&t7, NULL, process_dosage, &left_payload) != 0) {
            *exit = 1;
            return;
        }
    }

    if (perform_fast_mbd) {
        pthread_join(t1, NULL);
        pthread_join(t2, NULL);
        pthread_join(t3, NULL);

        // Combine all three results
        for (size_t i = 0; i < color_count; i++) {
            L_payload.D[i] += U_payload.D[i] + V_payload.D[i];
        }
        normalize_map(L_payload.D);
    }

    if (perform_dosage) {
        pthread_join(t4, NULL);
        pthread_join(t5, NULL);
        pthread_join(t6, NULL);
        pthread_join(t7, NULL);

        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                double t = index_map(top_payload.result, y, x);
                double r = index_map(right_payload.result, y, x);
                double b = index_map(bottom_payload.result, y, x);
                double l = index_map(left_payload.result, y, x);
                double max = fmax(t, fmax(r, fmax(b, l)));
                index_map(top_payload.result, y, x) = (t + r + b + l) - max;
            }
        }
        normalize_map(top_payload.result);
    }

    if (perform_fast_mbd && perform_dosage) {
        for (size_t i = 0; i < width * height; i++) {
            work_image[i] += (top_payload.result)[i];
        }
        normalize_map(work_image);
    }

    else if (perform_dosage) {
        for (size_t i = 0; i < width * height; i++) {
            work_image[i] = (top_payload.result)[i];
        }
    }
}