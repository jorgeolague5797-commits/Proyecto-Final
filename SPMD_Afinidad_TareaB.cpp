#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <string>
#include <iomanip>
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

float clampFloat(float value) {
    if (value < 0.0f) return 0.0f;
    if (value > 255.0f) return 255.0f;
    return value;
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

void generateMandelbrotPlanes(
    std::vector<float>& red,
    std::vector<float>& green,
    std::vector<float>& blue,
    int width,
    int height,
    int maxIterations
) {
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

            Pixel color = mandelbrotColor(iteration, maxIterations);
            size_t index = static_cast<size_t>(y) * width + x;

            red[index] = static_cast<float>(color.r);
            green[index] = static_cast<float>(color.g);
            blue[index] = static_cast<float>(color.b);
        }
    }
}

std::vector<float> createGaussianKernel(int radius, float sigma) {
    int size = 2 * radius + 1;
    std::vector<float> kernel(size * size);

    float sum = 0.0f;

    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            float exponent = -static_cast<float>(kx * kx + ky * ky) / (2.0f * sigma * sigma);
            float value = std::exp(exponent);

            int index = (ky + radius) * size + (kx + radius);
            kernel[index] = value;
            sum += value;
        }
    }

    for (float& value : kernel) {
        value /= sum;
    }

    return kernel;
}

void applyConvolutionSPMDSIMD(
    const float* __restrict inputR,
    const float* __restrict inputG,
    const float* __restrict inputB,
    float* __restrict outputR,
    float* __restrict outputG,
    float* __restrict outputB,
    const float* __restrict kernel,
    int width,
    int height,
    int radius
) {
    const int kernelSize = 2 * radius + 1;
    const long long totalPixels = static_cast<long long>(width) * height;

    // Primero se copian los bordes. El calculo vectorizado se aplica en la zona interna.
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < totalPixels; ++i) {
        outputR[i] = inputR[i];
        outputG[i] = inputG[i];
        outputB[i] = inputB[i];
    }

    /*
        Estructura SPMD:
        - Se crea una region paralela.
        - Cada hilo ejecuta el mismo codigo.
        - El trabajo se reparte por filas de la imagen.
        - El bucle mas interno kx se fuerza a vectorizar con omp simd.
    */
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int y = radius; y < height - radius; ++y) {
            for (int x = radius; x < width - radius; ++x) {
                float sumR = 0.0f;
                float sumG = 0.0f;
                float sumB = 0.0f;

                for (int ky = -radius; ky <= radius; ++ky) {
                    int imageBase = (y + ky) * width + (x - radius);
                    int kernelBase = (ky + radius) * kernelSize;

                    #pragma omp simd reduction(+:sumR, sumG, sumB)
                    for (int kx = 0; kx < kernelSize; ++kx) {
                        int imageIndex = imageBase + kx;
                        int kernelIndex = kernelBase + kx;

                        float weight = kernel[kernelIndex];

                        sumR += inputR[imageIndex] * weight;
                        sumG += inputG[imageIndex] * weight;
                        sumB += inputB[imageIndex] * weight;
                    }
                }

                int outputIndex = y * width + x;

                outputR[outputIndex] = clampFloat(sumR);
                outputG[outputIndex] = clampFloat(sumG);
                outputB[outputIndex] = clampFloat(sumB);
            }
        }
    }
}

unsigned long long checksumImagePlanes(
    const std::vector<float>& red,
    const std::vector<float>& green,
    const std::vector<float>& blue
) {
    unsigned long long checksum = 0;
    long long totalPixels = static_cast<long long>(red.size());

    #pragma omp parallel for reduction(+:checksum)
    for (long long i = 0; i < totalPixels; ++i) {
        checksum += static_cast<unsigned long long>(red[i]);
        checksum += static_cast<unsigned long long>(green[i]);
        checksum += static_cast<unsigned long long>(blue[i]);
    }

    return checksum;
}

int main(int argc, char* argv[]) {
    int width = 1920;
    int height = 1080;
    int maxIterations = 500;
    int radius = 7;
    float sigma = 3.0f;
    int repetitions = 5;

    if (argc >= 2) width = std::stoi(argv[1]);
    if (argc >= 3) height = std::stoi(argv[2]);
    if (argc >= 4) maxIterations = std::stoi(argv[3]);
    if (argc >= 5) radius = std::stoi(argv[4]);
    if (argc >= 6) sigma = std::stof(argv[5]);
    if (argc >= 7) repetitions = std::stoi(argv[6]);

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Benchmark SPMD + SIMD + Afinidad - Tarea B\n";
    std::cout << "Resolucion: " << width << " x " << height << "\n";
    std::cout << "Iteraciones Mandelbrot: " << maxIterations << "\n";
    std::cout << "Radio convolucion: " << radius << "\n";
    std::cout << "Sigma: " << sigma << "\n";
    std::cout << "Repeticiones: " << repetitions << "\n";
    std::cout << "Hilos OpenMP: " << omp_get_max_threads() << "\n";
    std::cout << "OMP_PROC_BIND enum: " << static_cast<int>(omp_get_proc_bind()) << "\n";
    std::cout << "OMP_PLACES detectados: " << omp_get_num_places() << "\n\n";

    size_t totalPixels = static_cast<size_t>(width) * height;

    std::vector<float> inputR(totalPixels);
    std::vector<float> inputG(totalPixels);
    std::vector<float> inputB(totalPixels);

    std::vector<float> outputR(totalPixels);
    std::vector<float> outputG(totalPixels);
    std::vector<float> outputB(totalPixels);

    std::vector<float> kernel = createGaussianKernel(radius, sigma);

    double startMandelbrot = omp_get_wtime();

    generateMandelbrotPlanes(
        inputR,
        inputG,
        inputB,
        width,
        height,
        maxIterations
    );

    double endMandelbrot = omp_get_wtime();

    std::cout << "Tiempo generacion Mandelbrot: "
              << endMandelbrot - startMandelbrot
              << " segundos\n";

    // Calentamiento para reducir ruido de medicion.
    applyConvolutionSPMDSIMD(
        inputR.data(),
        inputG.data(),
        inputB.data(),
        outputR.data(),
        outputG.data(),
        outputB.data(),
        kernel.data(),
        width,
        height,
        radius
    );

    double sumTime = 0.0;
    double bestTime = std::numeric_limits<double>::max();

    for (int rep = 0; rep < repetitions; ++rep) {
        double start = omp_get_wtime();

        applyConvolutionSPMDSIMD(
            inputR.data(),
            inputG.data(),
            inputB.data(),
            outputR.data(),
            outputG.data(),
            outputB.data(),
            kernel.data(),
            width,
            height,
            radius
        );

        double end = omp_get_wtime();
        double elapsed = end - start;

        sumTime += elapsed;
        bestTime = std::min(bestTime, elapsed);
    }

    unsigned long long checksum = checksumImagePlanes(outputR, outputG, outputB);

    std::cout << "\nResultados Tarea B SPMD + SIMD\n";
    std::cout << "promedio_segundos,mejor_segundos,checksum\n";
    std::cout << sumTime / repetitions << ","
              << bestTime << ","
              << checksum << "\n";

    std::cout << "\nNota:\n";
    std::cout << "El bucle interno kx del filtro usa #pragma omp simd reduction.\n";
    std::cout << "La vectorizacion se verifica con el reporte del compilador.\n";

    return 0;
}