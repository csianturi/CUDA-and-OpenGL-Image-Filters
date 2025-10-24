#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <algorithm>
using namespace std::chrono;

// ─────────────────────────────────────────────────────────────
// CPU 3×3 Sobel Edge Detection
// ─────────────────────────────────────────────────────────────
void sobel_cpu(const unsigned char* input, unsigned char* output,
               int width, int height, int channels)
{
    int Gx[3][3] = {{-1, 0, 1},
                    {-2, 0, 2},
                    {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1},
                    { 0,  0,  0},
                    { 1,  2,  1}};

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int gx = 0, gy = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int pixel = input[(y + ky) * width + (x + kx)];
                    gx += Gx[ky + 1][kx + 1] * pixel;
                    gy += Gy[ky + 1][kx + 1] * pixel;
                }
            }
            int mag = abs(gx) + abs(gy);
            output[y * width + x] = static_cast<unsigned char>(min(255, mag));
        }
    }
}


// ─────────────────────────────────────────────────────────────
// CUDA Sobel Edge Detection
// ─────────────────────────────────────────────────────────────
__global__ void sobel_filter(const unsigned char* input, unsigned char* output,
                             int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Ignore image border
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }

    int Gx = 0;
    int Gy = 0;

    // Sobel filter masks
    int sobel_x[3][3] = {{-1, 0, 1},
                         {-2, 0, 2},
                         {-1, 0, 1}};
    int sobel_y[3][3] = {{-1, -2, -1},
                         { 0,  0,  0},
                         { 1,  2,  1}};

    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            unsigned char pixel = input[(y + ky) * width + (x + kx)];
            Gx += sobel_x[ky + 1][kx + 1] * pixel;
            Gy += sobel_y[ky + 1][kx + 1] * pixel;
        }
    }

    int mag = abs(Gx) + abs(Gy); // faster than sqrt(Gx*Gx + Gy*Gy)
    mag = min(255, mag);         // clamp to 8-bit range
    output[y * width + x] = static_cast<unsigned char>(mag);
}


// ─────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────
int main()
{
    // ─── Load grayscale image ───
    int width, height, channels;
    unsigned char* h_input = stbi_load("../images/input_gray.png", &width, &height, &channels, 1);
    if (!h_input) {
        printf("Failed to load image.\n");
        return -1;
    }
    printf("Loaded image: %dx%d\n", width, height);

    size_t imgSize = width * height;
    unsigned char* h_output_cpu = (unsigned char*)malloc(imgSize);
    unsigned char* h_output_gpu = (unsigned char*)malloc(imgSize);

    // ─── CPU SOBEL ───
    auto t1 = high_resolution_clock::now();
    sobel_cpu(h_input, h_output_cpu, width, height, 1);
    auto t2 = high_resolution_clock::now();
    double cpu_ms = duration<double, std::milli>(t2 - t1).count();
    printf("CPU Sobel: %.3f ms\n", cpu_ms);

    stbi_write_png("../images/output_sobel_cpu.png", width, height, 1, h_output_cpu, width);
    printf("Wrote CPU sobel image -> output_sobel_cpu.png\n");

    // ─── GPU SOBEL ───
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);

    // CUDA event setup for timing
    cudaEvent_t eStart, eAfterH2D, eAfterKernel, eStop;
    cudaEventCreate(&eStart);
    cudaEventCreate(&eAfterH2D);
    cudaEventCreate(&eAfterKernel);
    cudaEventCreate(&eStop);

    cudaEventRecord(eStart);
    cudaMemcpy(d_input, h_input, imgSize, cudaMemcpyHostToDevice);
    cudaEventRecord(eAfterH2D);

    dim3 block(16, 16); // or try (32,8)
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    sobel_filter<<<grid, block>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(eAfterKernel);

    cudaMemcpy(h_output_gpu, d_output, imgSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(eStop);
    cudaEventSynchronize(eStop);

    // ─── Measure CUDA times ───
    float tH2D=0, tKernel=0, tD2H=0, tTotal=0;
    cudaEventElapsedTime(&tH2D, eStart, eAfterH2D);
    cudaEventElapsedTime(&tKernel, eAfterH2D, eAfterKernel);
    cudaEventElapsedTime(&tD2H, eAfterKernel, eStop);
    cudaEventElapsedTime(&tTotal, eStart, eStop);

    printf("\n--- Timing (ms) ---\n");
    printf("CPU Sobel     : %.3f\n", cpu_ms);
    printf("H2D transfer : %.3f\n", tH2D);
    printf("Kernel       : %.3f\n", tKernel);
    printf("D2H transfer : %.3f\n", tD2H);
    printf("Total GPU    : %.3f\n", tTotal);
    printf("Speedup (CPU / Total GPU): %.2fx\n", cpu_ms / tTotal);

    // ─── Validate output ───
    int mismatches = 0;
    for (int i = 0; i < width * height; ++i) {
        int diff = int(h_output_cpu[i]) - int(h_output_gpu[i]);
        if (abs(diff) > 1) ++mismatches;
    }
    printf("Mismatches: %d\n", mismatches);

    stbi_write_png("../images/output_sobel_gpu.png", width, height, 1, h_output_gpu, width);
    printf("Wrote GPU sobel image -> output_sobel_gpu.png\n");

    // ─── Cleanup ───
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output_cpu);
    free(h_output_gpu);
    stbi_image_free(h_input);
    cudaEventDestroy(eStart);
    cudaEventDestroy(eAfterH2D);
    cudaEventDestroy(eAfterKernel);
    cudaEventDestroy(eStop);

    return 0;
}
