#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define BLOCK_W 16
#define BLOCK_H 16
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <algorithm>
using namespace std::chrono;

// ─────────────────────────────────────────────────────────────
// CPU 3×3 Box Blur
// ─────────────────────────────────────────────────────────────
void blur_cpu(const unsigned char* in, unsigned char* out,
              int w, int h, int channels)
{
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int sum = 0, count = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    // Compute neighbor coordinates
                    int nx = x + dx;
                    int ny = y + dy;

                    // Clamp to image boundaries to avoid out-of-bounds access
                    if (nx < 0) nx = 0;
                    if (ny < 0) ny = 0;
                    if (nx >= w) nx = w - 1;
                    if (ny >= h) ny = h - 1;
                    sum += in[ny * w + nx];
                    ++count;
                }
            out[y * w + x] = static_cast<unsigned char>(sum / count);
        }
    }
}

// ─────────────────────────────────────────────────────────────
// CUDA Box Blur using Shared Memory
// ─────────────────────────────────────────────────────────────


__global__ void blur_shared(const unsigned char* in, unsigned char* out,
                            int w, int h)
{
    // Shared memory tile, including halo (+2)
    __shared__ unsigned char tile[BLOCK_H + 2][BLOCK_W + 2];

    // Global coordinates
    int x = blockIdx.x * BLOCK_W + threadIdx.x;
    int y = blockIdx.y * BLOCK_H + threadIdx.y;

    // Shared memory coordinates (with +1 offset for halo)
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    // Clamp coordinates for safe global loads
    int gx = min(max(x, 0), w - 1);
    int gy = min(max(y, 0), h - 1);

    // Load main pixel into shared memory
    tile[ly][lx] = in[gy * w + gx];

    // ───── Load halo borders ─────
    // Left halo
    if (threadIdx.x == 0)
        tile[ly][0] = in[gy * w + max(gx - 1, 0)];

    // Right halo
    if (threadIdx.x == BLOCK_W - 1)
        tile[ly][BLOCK_W + 1] = in[gy * w + min(gx + 1, w - 1)];

    // Top halo
    if (threadIdx.y == 0)
        tile[0][lx] = in[max(gy - 1, 0) * w + gx];

    // Bottom halo
    if (threadIdx.y == BLOCK_H - 1)
        tile[BLOCK_H + 1][lx] = in[min(gy + 1, h - 1) * w + gx];

    // Corner halos (optional)
    if (threadIdx.x == 0 && threadIdx.y == 0)
        tile[0][0] = in[max(gy - 1, 0) * w + max(gx - 1, 0)];
    if (threadIdx.x == BLOCK_W - 1 && threadIdx.y == 0)
        tile[0][BLOCK_W + 1] = in[max(gy - 1, 0) * w + min(gx + 1, w - 1)];
    if (threadIdx.x == 0 && threadIdx.y == BLOCK_H - 1)
        tile[BLOCK_H + 1][0] = in[min(gy + 1, h - 1) * w + max(gx - 1, 0)];
    if (threadIdx.x == BLOCK_W - 1 && threadIdx.y == BLOCK_H - 1)
        tile[BLOCK_H + 1][BLOCK_W + 1] = in[min(gy + 1, h - 1) * w + min(gx + 1, w - 1)];

    // Wait for all threads to finish loading shared memory
    __syncthreads();

    // Compute blur only for threads that correspond to valid output pixels
    if (x < w && y < h) {
        int sum = 0;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                sum += tile[ly + dy][lx + dx];

        out[y * w + x] = static_cast<unsigned char>(sum / 9);
    }
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

    // ─── CPU BLUR ───
    auto t1 = high_resolution_clock::now();
    blur_cpu(h_input, h_output_cpu, width, height, 1);
    auto t2 = high_resolution_clock::now();
    double cpu_ms = duration<double, std::milli>(t2 - t1).count();
    printf("CPU blur: %.3f ms\n", cpu_ms);

    stbi_write_png("../images/output_blur_shared_cpu.png", width, height, 1, h_output_cpu, width);
    printf("Wrote CPU blurred image -> output_blur_shared_cpu.png\n");

    // ─── GPU BLUR ───
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

    blur_shared<<<grid, block>>>(d_input, d_output, width, height);
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
    printf("CPU blur     : %.3f\n", cpu_ms);
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

    stbi_write_png("../images/output_blur_shared_gpu.png", width, height, 1, h_output_gpu, width);
    printf("Wrote GPU blurred image -> output_blur_shared_gpu.png\n");

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
