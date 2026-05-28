#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <string>
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

void savePPM(const std::string& filename, const std::vector<Pixel>& image, int width, int height) {
    std::ofstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Error: no se pudo crear el archivo " << filename << std::endl;
        return;
    }

    file << "P6\n" << width << " " << height << "\n255\n";

    for (const Pixel& p : image) {
        file.write(reinterpret_cast<const char*>(&p.r), 1);
        file.write(reinterpret_cast<const char*>(&p.g), 1);
        file.write(reinterpret_cast<const char*>(&p.b), 1);
    }

    file.close();
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

    #pragma omp parallel for
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

int main(int argc, char* argv[]) {
    int width = 7680;
    int height = 4320;
    int maxIterations = 1000;
    int radius = 15;
    double sigma = 6.0;

    if (argc >= 2) width = std::stoi(argv[1]);
    if (argc >= 3) height = std::stoi(argv[2]);
    if (argc >= 4) maxIterations = std::stoi(argv[3]);
    if (argc >= 5) radius = std::stoi(argv[4]);
    if (argc >= 6) sigma = std::stod(argv[5]);

    if (width <= 0 || height <= 0 || maxIterations <= 0 || radius <= 0 || sigma <= 0.0) {
        std::cerr << "Uso inválido.\n";
        std::cerr << "Ejemplo: ./paralelo_ia 7680 4320 1000 15 6.0\n";
        return 1;
    }

    std::cout << "Configuracion OpenMP:\n";
    std::cout << "Resolucion: " << width << " x " << height << "\n";
    std::cout << "Iteraciones Mandelbrot: " << maxIterations << "\n";
    std::cout << "Radio convolucion Gaussiana: " << radius << "\n";
    std::cout << "Sigma: " << sigma << "\n";
    std::cout << "Hilos máximos OpenMP: " << omp_get_max_threads() << "\n\n";

    std::vector<Pixel> image(static_cast<size_t>(width) * height);
    std::vector<Pixel> filteredImage(static_cast<size_t>(width) * height);

    auto startMandelbrot = std::chrono::high_resolution_clock::now();

    generateMandelbrot(image, width, height, maxIterations);

    auto endMandelbrot = std::chrono::high_resolution_clock::now();

    savePPM("mandelbrot_original_openmp.ppm", image, width, height);

    auto kernel = createGaussianKernel(radius, sigma);

    auto startConvolution = std::chrono::high_resolution_clock::now();

    applyConvolution2D(image, filteredImage, width, height, kernel, radius);

    auto endConvolution = std::chrono::high_resolution_clock::now();

    savePPM("mandelbrot_filtrado_openmp.ppm", filteredImage, width, height);

    std::chrono::duration<double> mandelbrotTime = endMandelbrot - startMandelbrot;
    std::chrono::duration<double> convolutionTime = endConvolution - startConvolution;
    double totalTime = mandelbrotTime.count() + convolutionTime.count();

    std::cout << "Resultados de tiempo OpenMP - linea base IA:\n";
    std::cout << "Tiempo Tarea A - Mandelbrot: " << mandelbrotTime.count() << " segundos\n";
    std::cout << "Tiempo Tarea B - Convolución 2D: " << convolutionTime.count() << " segundos\n";
    std::cout << "Tiempo total: " << totalTime << " segundos\n";

    std::cout << "\nArchivos generados:\n";
    std::cout << "mandelbrot_original_openmp.ppm\n";
    std::cout << "mandelbrot_filtrado_openmp.ppm\n";

    return 0;
}