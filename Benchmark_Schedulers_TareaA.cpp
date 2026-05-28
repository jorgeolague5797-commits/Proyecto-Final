#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <string>
#include <limits>
#include <omp.h>

struct Pixel {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

unsigned char clampToByte(double value) {
    if (value < 0.0) return 0;
    if (value > 255.0) return 255;
    return static_cast<unsigned char>(std::round(value));
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

// Version con planificador por defecto de OpenMP.
// Normalmente equivale a static si no se especifica otra cosa.
void generateMandelbrotDefaultStatic(
    std::vector<Pixel>& image,
    int width,
    int height,
    int maxIterations
) {
    const double minReal = -2.5;
    const double maxReal = 1.0;
    const double minImag = -1.0;
    const double maxImag = 1.0;

    #pragma omp parallel for
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

// Version para probar dynamic y guided usando omp_set_schedule.
void generateMandelbrotRuntimeSchedule(
    std::vector<Pixel>& image,
    int width,
    int height,
    int maxIterations
) {
    const double minReal = -2.5;
    const double maxReal = 1.0;
    const double minImag = -1.0;
    const double maxImag = 1.0;

    #pragma omp parallel for schedule(runtime)
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

// Checksum sencillo para evitar que el compilador elimine trabajo.
unsigned long long checksumImage(const std::vector<Pixel>& image) {
    unsigned long long checksum = 0;

    for (const Pixel& p : image) {
        checksum += p.r;
        checksum += p.g;
        checksum += p.b;
    }

    return checksum;
}

double measureDefaultStatic(
    std::vector<Pixel>& image,
    int width,
    int height,
    int maxIterations
) {
    auto start = std::chrono::high_resolution_clock::now();

    generateMandelbrotDefaultStatic(image, width, height, maxIterations);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

double measureRuntimeSchedule(
    std::vector<Pixel>& image,
    int width,
    int height,
    int maxIterations,
    omp_sched_t scheduleKind,
    int chunkSize
) {
    omp_set_schedule(scheduleKind, chunkSize);

    auto start = std::chrono::high_resolution_clock::now();

    generateMandelbrotRuntimeSchedule(image, width, height, maxIterations);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

int main(int argc, char* argv[]) {
    int width = 1920;
    int height = 1080;
    int maxIterations = 500;
    int repetitions = 3;

    if (argc >= 2) width = std::stoi(argv[1]);
    if (argc >= 3) height = std::stoi(argv[2]);
    if (argc >= 4) maxIterations = std::stoi(argv[3]);
    if (argc >= 5) repetitions = std::stoi(argv[4]);

    std::vector<Pixel> image(static_cast<size_t>(width) * height);

    std::cout << "Benchmark de schedulers OpenMP - Tarea A Mandelbrot\n";
    std::cout << "Resolucion: " << width << " x " << height << "\n";
    std::cout << "Iteraciones Mandelbrot: " << maxIterations << "\n";
    std::cout << "Repeticiones por prueba: " << repetitions << "\n";
    std::cout << "Hilos OpenMP: " << omp_get_max_threads() << "\n\n";

    std::cout << "scheduler,chunk,promedio_segundos,mejor_segundos,checksum\n";

    double bestGlobalTime = std::numeric_limits<double>::max();
    std::string bestGlobalName = "";
    int bestGlobalChunk = 0;

    // Prueba del planificador por defecto.
    {
        double sum = 0.0;
        double best = std::numeric_limits<double>::max();
        unsigned long long checksum = 0;

        for (int rep = 0; rep < repetitions; ++rep) {
            double time = measureDefaultStatic(image, width, height, maxIterations);
            checksum = checksumImage(image);

            sum += time;
            best = std::min(best, time);
        }

        double average = sum / repetitions;

        std::cout << "default_static,0," << average << "," << best << "," << checksum << "\n";

        if (best < bestGlobalTime) {
            bestGlobalTime = best;
            bestGlobalName = "default_static";
            bestGlobalChunk = 0;
        }
    }

    std::vector<int> chunks = {1, 2, 4, 8, 16, 32, 64, 128};

    // Pruebas con dynamic.
    for (int chunk : chunks) {
        double sum = 0.0;
        double best = std::numeric_limits<double>::max();
        unsigned long long checksum = 0;

        for (int rep = 0; rep < repetitions; ++rep) {
            double time = measureRuntimeSchedule(
                image,
                width,
                height,
                maxIterations,
                omp_sched_dynamic,
                chunk
            );

            checksum = checksumImage(image);

            sum += time;
            best = std::min(best, time);
        }

        double average = sum / repetitions;

        std::cout << "dynamic," << chunk << "," << average << "," << best << "," << checksum << "\n";

        if (best < bestGlobalTime) {
            bestGlobalTime = best;
            bestGlobalName = "dynamic";
            bestGlobalChunk = chunk;
        }
    }

    // Pruebas con guided.
    for (int chunk : chunks) {
        double sum = 0.0;
        double best = std::numeric_limits<double>::max();
        unsigned long long checksum = 0;

        for (int rep = 0; rep < repetitions; ++rep) {
            double time = measureRuntimeSchedule(
                image,
                width,
                height,
                maxIterations,
                omp_sched_guided,
                chunk
            );

            checksum = checksumImage(image);

            sum += time;
            best = std::min(best, time);
        }

        double average = sum / repetitions;

        std::cout << "guided," << chunk << "," << average << "," << best << "," << checksum << "\n";

        if (best < bestGlobalTime) {
            bestGlobalTime = best;
            bestGlobalName = "guided";
            bestGlobalChunk = chunk;
        }
    }

    std::cout << "\nMejor configuracion encontrada:\n";
    std::cout << "Scheduler: " << bestGlobalName << "\n";
    std::cout << "Chunk size: " << bestGlobalChunk << "\n";
    std::cout << "Mejor tiempo: " << bestGlobalTime << " segundos\n";

    return 0;
}
