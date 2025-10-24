#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
using namespace std::chrono;

__global__ void rgb_to_grayscale(unsigned char* input, unsigned char* output,
                                 int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;
    unsigned char r = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char b = input[idx + 2];

    // Luma formula
    unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    output[y * width + x] = gray;
}

void grayscale_cpu(unsigned char* input, unsigned char* output,
                   int width, int height, int channels)
{
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            unsigned char r = input[idx];
            unsigned char g = input[idx + 1];
            unsigned char b = input[idx + 2];

            unsigned char gray = static_cast<unsigned char>(
                0.299f * r + 0.587f * g + 0.114f * b
            );

            output[y * width + x] = gray;
        }
    }
}




int main()
{
    int width, height, channels;
    unsigned char* h_input = stbi_load("../images/input.png", &width, &height, &channels, 3);
    if (!h_input) {
        printf("Failed to load image.\n");
        return -1;
    }

    size_t imgSize = width * height;
    size_t imgSizeRGB = imgSize * 3;

    // Allocate separate host buffers
    unsigned char* h_gpu_output = (unsigned char*)malloc(imgSize); // GPU result
    unsigned char* h_cpu_output = (unsigned char*)malloc(imgSize); // CPU result

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imgSizeRGB);
    cudaMalloc(&d_output, imgSize);

    // Create CUDA timing events
    cudaEvent_t eStart, eAfterH2D, eAfterKernel, eStop;
    cudaEventCreate(&eStart);
    cudaEventCreate(&eAfterH2D);
    cudaEventCreate(&eAfterKernel);
    cudaEventCreate(&eStop);

    // --- GPU pipeline ---
    cudaEventRecord(eStart);
    cudaMemcpy(d_input, h_input, imgSizeRGB, cudaMemcpyHostToDevice);
    cudaEventRecord(eAfterH2D);

    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    rgb_to_grayscale<<<grid, block>>>(d_input, d_output, width, height, 3);
    cudaDeviceSynchronize();
    cudaEventRecord(eAfterKernel);

    cudaMemcpy(h_gpu_output, d_output, imgSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(eStop);
    cudaEventSynchronize(eStop);

    // --- CPU baseline ---
    auto t1 = high_resolution_clock::now();
    grayscale_cpu(h_input, h_cpu_output, width, height, 3);
    auto t2 = high_resolution_clock::now();
    double cpu_ms = duration<double, std::milli>(t2 - t1).count();
    printf("CPU time: %.3f ms\n", cpu_ms);

    // --- Compare outputs ---
    int mismatches = 0;
    for (int i = 0; i < width * height; ++i) {
        int diff = int(h_gpu_output[i]) - int(h_cpu_output[i]);
        if (diff < -1 || diff > 1) ++mismatches;
    }
    printf("Mismatches: %d\n", mismatches);

    // --- Save outputs ---
    stbi_write_png("../images/output_gpu.png", width, height, 1, h_gpu_output, width);
    stbi_write_png("../images/output_cpu.png", width, height, 1, h_cpu_output, width);
    printf("Wrote grayscale images to images/output_cpu.png and output_gpu.png\n");

    // --- Timing breakdown ---
    float tH2D=0, tKernel=0, tD2H=0, tTotal=0;
    cudaEventElapsedTime(&tH2D,   eStart,       eAfterH2D);
    cudaEventElapsedTime(&tKernel,eAfterH2D,    eAfterKernel);
    cudaEventElapsedTime(&tD2H,   eAfterKernel, eStop);
    cudaEventElapsedTime(&tTotal, eStart,       eStop);

    printf("\n--- Timing (ms) ---\n");
    printf("CPU grayscale : %.3f\n", cpu_ms);
    printf("H2D           : %.3f\n", tH2D);
    printf("Kernel        : %.3f\n", tKernel);
    printf("D2H           : %.3f\n", tD2H);
    printf("Total         : %.3f\n", tTotal);
    printf("Speedup (CPU/Total): %.2fx\n", cpu_ms / tTotal);

    // Cleanup
    cudaEventDestroy(eStart);
    cudaEventDestroy(eAfterH2D);
    cudaEventDestroy(eAfterKernel);
    cudaEventDestroy(eStop);

    stbi_image_free(h_input);
    free(h_cpu_output);
    free(h_gpu_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
