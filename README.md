# CUDA and OpenGL Image Filters
Collection of small CUDA and OpenGL projects for GPU programming practice.

# Projects
- **CUDA Image Filters**: Grayscale, blur, Sobel edge detection (with optimized thread block configurations and coalesced memory access patterns)
- **OpenGL Shader Filters**: Real-time image effects with GLSL

# CUDA Image Filters
Applies grayscale, blur, and sobel edge detection to an input 1920x1080 image. CUDA kernels are used to implement the filters and the performance of each are compared against a CPU baseline.
All three filters use the most optimal thread block configurations. 
I also included another blur implementation that uses shared memory instead of the global memory. It is faster to access the shared memory because of it being located right on the chip instead of the VRAM.

## Grayscale Filter

### Input Image
![input](https://github.com/user-attachments/assets/0a8ef5c1-fcab-494d-b005-16b7389f2146)

### Output Image (Both CPU and GPU output images should be the same)
<img width="1920" height="1080" alt="output_cpu" src="https://github.com/user-attachments/assets/74b74590-4e7f-4615-860a-cf49486444da" />

### Performance
` C:\nvidia-prep\cuda-filters\src>grayscale.exe
CPU time: 19.274 ms
Mismatches: 0
Wrote grayscale images to images/output_cpu.png and output_gpu.png

--- Timing (ms) ---
CPU grayscale : 19.274
H2D           : 0.611
Kernel        : 0.742
D2H           : 0.524
Total         : 1.877
Speedup (CPU/Total): 10.27x `

## Blur Filter (using global memory)

### Input Image
<img width="1920" height="1080" alt="input_gray" src="https://github.com/user-attachments/assets/93419459-f513-4e49-8c03-53de98e47185" />

### Output Image (Both CPU and GPU output images should be the same)

<img width="1920" height="1080" alt="output_blur_gpu" src="https://github.com/user-attachments/assets/7a1b7301-6e03-4c03-8d2f-494622612a40" />

### Performance
` C:\nvidia-prep\cuda-filters\src>blur.exe
Loaded image: 1920x1080
CPU blur: 88.056 ms
Wrote CPU blurred image -> output_blur_cpu.png

--- Timing (ms) ---
CPU blur     : 88.056
H2D transfer : 0.408
Kernel       : 0.809
D2H transfer : 0.708
Total GPU    : 1.925
Speedup (CPU / Total GPU): 45.74x
Mismatches: 0
Wrote GPU blurred image -> output_blur_gpu.png `

## Blur Filter (using shared memory)

### Input Image
<img width="1920" height="1080" alt="input_gray" src="https://github.com/user-attachments/assets/621e90d5-e9b4-461c-b105-94d99a991f21" />

### Output Image (Both CPU and GPU output images should be the same)
<img width="1920" height="1080" alt="output_blur_shared_gpu" src="https://github.com/user-attachments/assets/d3741805-fd59-4ed9-81d1-74b3e1fb8bbe" />

### Performance
` C:\nvidia-prep\cuda-filters\src>blur_shared.exe
Loaded image: 1920x1080
CPU blur: 87.500 ms
Wrote CPU blurred image -> output_blur_shared_cpu.png

--- Timing (ms) ---
CPU blur     : 87.500
H2D transfer : 0.249
Kernel       : 0.648
D2H transfer : 0.483
Total GPU    : 1.381
Speedup (CPU / Total GPU): 63.35x
Mismatches: 0
Wrote GPU blurred image -> output_blur_shared_gpu.png `

## Sobel Edge Detection

### Input Image
<img width="1920" height="1080" alt="input_gray" src="https://github.com/user-attachments/assets/3fb2570c-e229-484c-ba5d-2b3c9681b381" />

### Output Image (Both CPU and GPU output images should be the same)
<img width="1920" height="1080" alt="output_sobel_cpu" src="https://github.com/user-attachments/assets/b55a1b88-6f27-458f-826a-0373f26b798a" />

### Performance
` C:\nvidia-prep\cuda-filters\src>sobel.exe
Loaded image: 1920x1080
CPU Sobel: 112.752 ms
Wrote CPU sobel image -> output_sobel_cpu.png

--- Timing (ms) ---
CPU Sobel     : 112.752
H2D transfer : 0.440
Kernel       : 0.789
D2H transfer : 0.678
Total GPU    : 1.908
Speedup (CPU / Total GPU): 59.09x
Mismatches: 0
Wrote GPU sobel image -> output_sobel_gpu.png `

