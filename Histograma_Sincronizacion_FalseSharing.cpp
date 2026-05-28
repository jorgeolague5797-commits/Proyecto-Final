#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <string>
#include <cstdint>
#include <limits>
#include <iomanip>
#include <omp.h>

struct Pixel {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

static constexpr int COLOR_BINS = 1 << 24;

struct HistogramStats {
    double timeSeconds;
    unsigned long long totalPixels;
    unsigned long long uniqueColors;
};

struct FalseSharingStats {
    double timeSeconds;
    unsigned long long checksum;
};

struct CompactCounter {
    volatile unsigned long long value;
};

struct alignas(64) PaddedCounter {
    volatile unsigned long long value;
    char padding[64 - sizeof(unsigned long long)];
};

unsigned char clampToByte(double value) {
    if (value < 0.0) return 0;
    if (value > 255.0) return 255;
    return static_cast<unsigned char>(std::round(value));
}

uint32_t colorKey(const Pixel& p) {
    return (static_cast<uint32_t>(p.r) << 16) |
           (static_cast<uint32_t>(p.g) << 8) |
           static_cast<uint32_t>(p.b);
}

Pixel mandelbrotColor(int iterations, int maxIterations) {
    if (iterations == maxIterations) {
        return {0, 0, 0};
    }

    double t = static_cast<double>(iterations) / static_cast<double>(maxIterations);

    double r = 9.0 * (1.0 - t) * t * t * t * 255.0;
    double g = 15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0;
    double b = 8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0;

    return {
        clampToByte(r),
        clampToByte(g),
        clampToByte(b)
    };
}

void generateMandelbrot(std::vector<Pixel>& image, int width, int height, int maxIterations) {
    const double minReal = -2.5;
    const double maxReal = 1.0;
    const double minImag = -1.0;
    const double maxImag = 1.0;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        double cImag = minImag + static_cast<double>(y) * (maxImag - minImag) / static_cast<double>(height - 1);

        for (int x = 0; x < width; ++x) {
            double cReal = minReal + static_cast<double>(x) * (maxReal - minReal) / static_cast<double>(width - 1);

            double zReal = 0.0;
            double zImag = 0.0;
            int iteration = 0;

            while (zReal * zReal + zImag * zImag <= 4.0 && iteration < maxIterations) {
                double tempReal = zReal * zReal - zImag * zImag + cReal;
                zImag = 2.0 * zReal * zImag + cImag;
                zReal = tempReal;
                ++iteration;
            }

            image[y * width + x] = mandelbrotColor(iteration, maxIterations);
        }
    }
}

std::vector<double> createGaussianKernel(int radius, double sigma) {
    int size = 2 * radius + 1;
    std::vector<double> kernel(size * size);

    double sum = 0.0;

    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            double exponent = -static_cast<double>(kx * kx + ky * ky) / (2.0 * sigma * sigma);
            double value = std::exp(exponent);

            int index = (ky + radius) * size + (kx + radius);
            kernel[index] = value;
            sum += value;
        }
    }

    for (double& value : kernel) {
        value /= sum;
    }

    return kernel;
}

void applyConvolution2D(
    const std::vector<Pixel>& input,
    std::vector<Pixel>& output,
    int width,
    int height,
    const std::vector<double>& kernel,
    int radius
) {
    int kernelSize = 2 * radius + 1;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sumR = 0.0;
            double sumG = 0.0;
            double sumB = 0.0;

            for (int ky = -radius; ky <= radius; ++ky) {
                int imageY = std::clamp(y + ky, 0, height - 1);

                for (int kx = -radius; kx <= radius; ++kx) {
                    int imageX = std::clamp(x + kx, 0, width - 1);

                    int imageIndex = imageY * width + imageX;
                    int kernelIndex = (ky + radius) * kernelSize + (kx + radius);

                    double weight = kernel[kernelIndex];

                    sumR += input[imageIndex].r * weight;
                    sumG += input[imageIndex].g * weight;
                    sumB += input[imageIndex].b * weight;
                }
            }

            output[y * width + x] = {
                clampToByte(sumR),
                clampToByte(sumG),
                clampToByte(sumB)
            };
        }
    }
}

HistogramStats histogramAtomic(const std::vector<Pixel>& image) {
    std::vector<uint32_t> histogram(COLOR_BINS, 0);

    double start = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(image.size()); ++i) {
        uint32_t key = colorKey(image[i]);

        #pragma omp atomic update
        histogram[key]++;
    }

    double end = omp_get_wtime();

    unsigned long long totalPixels = 0;
    unsigned long long uniqueColors = 0;

    for (uint32_t count : histogram) {
        if (count > 0) {
            uniqueColors++;
            totalPixels += count;
        }
    }

    return {end - start, totalPixels, uniqueColors};
}

HistogramStats histogramLocalPerThread(const std::vector<Pixel>& image) {
    int numThreads = omp_get_max_threads();

    std::vector<uint32_t> localHistograms(
        static_cast<size_t>(numThreads) * COLOR_BINS,
        0
    );

    std::vector<uint32_t> globalHistogram(COLOR_BINS, 0);

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint32_t* localHistogram = localHistograms.data() + static_cast<size_t>(tid) * COLOR_BINS;

        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(image.size()); ++i) {
            uint32_t key = colorKey(image[i]);
            localHistogram[key]++;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int color = 0; color < COLOR_BINS; ++color) {
        uint32_t sum = 0;

        for (int tid = 0; tid < numThreads; ++tid) {
            sum += localHistograms[static_cast<size_t>(tid) * COLOR_BINS + color];
        }

        globalHistogram[color] = sum;
    }

    double end = omp_get_wtime();

    unsigned long long totalPixels = 0;
    unsigned long long uniqueColors = 0;

    for (uint32_t count : globalHistogram) {
        if (count > 0) {
            uniqueColors++;
            totalPixels += count;
        }
    }

    return {end - start, totalPixels, uniqueColors};
}

template <typename CounterType>
FalseSharingStats falseSharingExperiment(int numThreads, long long incrementsPerThread) {
    std::vector<CounterType> counters(numThreads);

    for (int i = 0; i < numThreads; ++i) {
        counters[i].value = 0;
    }

    double start = omp_get_wtime();

    #pragma omp parallel num_threads(numThreads)
    {
        int tid = omp_get_thread_num();

        for (long long i = 0; i < incrementsPerThread; ++i) {
            counters[tid].value++;
        }
    }

    double end = omp_get_wtime();

    unsigned long long checksum = 0;

    for (int i = 0; i < numThreads; ++i) {
        checksum += counters[i].value;
    }

    return {end - start, checksum};
}

int main(int argc, char* argv[]) {
    int width = 1920;
    int height = 1080;
    int maxIterations = 500;
    int radius = 7;
    double sigma = 3.0;
    int repetitions = 3;
    long long falseSharingIncrements = 10000000;

    if (argc >= 2) width = std::stoi(argv[1]);
    if (argc >= 3) height = std::stoi(argv[2]);
    if (argc >= 4) maxIterations = std::stoi(argv[3]);
    if (argc >= 5) radius = std::stoi(argv[4]);
    if (argc >= 6) sigma = std::stod(argv[5]);
    if (argc >= 7) repetitions = std::stoi(argv[6]);
    if (argc >= 8) falseSharingIncrements = std::stoll(argv[7]);

    int numThreads = omp_get_max_threads();

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Benchmark de histograma y false sharing\n";
    std::cout << "Resolucion: " << width << " x " << height << "\n";
    std::cout << "Iteraciones Mandelbrot: " << maxIterations << "\n";
    std::cout << "Radio convolucion: " << radius << "\n";
    std::cout << "Sigma: " << sigma << "\n";
    std::cout << "Hilos OpenMP: " << numThreads << "\n";
    std::cout << "Repeticiones: " << repetitions << "\n\n";

    double memoryMB = static_cast<double>(numThreads) * COLOR_BINS * sizeof(uint32_t) / (1024.0 * 1024.0);
    std::cout << "Memoria aproximada para histogramas locales: " << memoryMB << " MB\n\n";

    std::vector<Pixel> image(static_cast<size_t>(width) * height);
    std::vector<Pixel> filteredImage(static_cast<size_t>(width) * height);

    auto kernel = createGaussianKernel(radius, sigma);

    double startMandelbrot = omp_get_wtime();
    generateMandelbrot(image, width, height, maxIterations);
    double endMandelbrot = omp_get_wtime();

    double startConvolution = omp_get_wtime();
    applyConvolution2D(image, filteredImage, width, height, kernel, radius);
    double endConvolution = omp_get_wtime();

    std::cout << "Tiempo generacion Mandelbrot: " << endMandelbrot - startMandelbrot << " segundos\n";
    std::cout << "Tiempo convolucion 2D: " << endConvolution - startConvolution << " segundos\n\n";

    std::cout << "metodo,promedio_segundos,mejor_segundos,total_pixeles,colores_unicos\n";

    double atomicSum = 0.0;
    double atomicBest = std::numeric_limits<double>::max();
    HistogramStats atomicLast{0.0, 0, 0};

    for (int rep = 0; rep < repetitions; ++rep) {
        HistogramStats stats = histogramAtomic(filteredImage);
        atomicLast = stats;
        atomicSum += stats.timeSeconds;
        atomicBest = std::min(atomicBest, stats.timeSeconds);
    }

    std::cout << "atomic,"
              << atomicSum / repetitions << ","
              << atomicBest << ","
              << atomicLast.totalPixels << ","
              << atomicLast.uniqueColors << "\n";

    double localSum = 0.0;
    double localBest = std::numeric_limits<double>::max();
    HistogramStats localLast{0.0, 0, 0};

    for (int rep = 0; rep < repetitions; ++rep) {
        HistogramStats stats = histogramLocalPerThread(filteredImage);
        localLast = stats;
        localSum += stats.timeSeconds;
        localBest = std::min(localBest, stats.timeSeconds);
    }

    std::cout << "local_per_thread,"
              << localSum / repetitions << ","
              << localBest << ","
              << localLast.totalPixels << ","
              << localLast.uniqueColors << "\n\n";

    std::cout << "False sharing test\n";
    std::cout << "tipo,promedio_segundos,mejor_segundos,checksum\n";

    double compactSum = 0.0;
    double compactBest = std::numeric_limits<double>::max();
    FalseSharingStats compactLast{0.0, 0};

    for (int rep = 0; rep < repetitions; ++rep) {
        FalseSharingStats stats = falseSharingExperiment<CompactCounter>(
            numThreads,
            falseSharingIncrements
        );

        compactLast = stats;
        compactSum += stats.timeSeconds;
        compactBest = std::min(compactBest, stats.timeSeconds);
    }

    std::cout << "compact_shared_array,"
              << compactSum / repetitions << ","
              << compactBest << ","
              << compactLast.checksum << "\n";

    double paddedSum = 0.0;
    double paddedBest = std::numeric_limits<double>::max();
    FalseSharingStats paddedLast{0.0, 0};

    for (int rep = 0; rep < repetitions; ++rep) {
        FalseSharingStats stats = falseSharingExperiment<PaddedCounter>(
            numThreads,
            falseSharingIncrements
        );

        paddedLast = stats;
        paddedSum += stats.timeSeconds;
        paddedBest = std::min(paddedBest, stats.timeSeconds);
    }

    std::cout << "padded_array,"
              << paddedSum / repetitions << ","
              << paddedBest << ","
              << paddedLast.checksum << "\n";

    std::cout << "\nInterpretacion rapida:\n";
    std::cout << "- atomic usa un histograma compartido y sincronizacion por actualizacion.\n";
    std::cout << "- local_per_thread usa histogramas privados por hilo y luego combina resultados.\n";
    std::cout << "- compact_shared_array puede mostrar false sharing porque los contadores quedan juntos en cache.\n";
    std::cout << "- padded_array separa contadores por linea de cache para reducir false sharing.\n";

    return 0;
}