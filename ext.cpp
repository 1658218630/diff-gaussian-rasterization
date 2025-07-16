/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 */

#include <torch/extension.h>
#include "rasterize_points.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <vector>
#include "cuda_rasterizer/statistical_constants.cuh"

#include <fstream>          
#include <string>           


namespace py = pybind11;


// ──────────────────────────────────────────────────────────────
//  Automatic loader: read 3‑D standard‑normal samples once
// ──────────────────────────────────────────────────────────────
namespace {
    // Only visible in this translation unit
    static bool std_samples_loaded = false;

    // Read txt and write to GPU constant; if already loaded, return immediately
    void load_std_samples_once(const std::string& path = "base_samples.txt")
    {
        if (std_samples_loaded) return;           // Already loaded → return immediately

        // ---------- Read file ----------
        std::ifstream fin(path);
        if (!fin.is_open()) {
            throw std::runtime_error("Cannot open sample file: " + path);
        }

        std::vector<float> host(MAX_STD_SAMPLES * 3);
        int N = 0;
        while (N < MAX_STD_SAMPLES && fin >> host[N*3 + 0] >> host[N*3 + 1] >> host[N*3 + 2]) {
            ++N;
        }
        fin.close();

        if (N == 0)
            throw std::runtime_error("No samples found in " + path);

        // ---------- Copy to GPU constant ----------
        // cudaMemcpyToSymbol(BASE_SAMPLES_MAX, host.data(),
        //                    N * 3 * sizeof(float), 0, cudaMemcpyHostToDevice);
        // cudaMemcpyToSymbol(NUM_STD_SAMPLES, &N,
        //                    sizeof(int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(
            (const void*)BASE_SAMPLES_MAX,   // ← 符号 BASE_SAMPLES_MAX 的地址
            host.data(),                     // ← host 端浮点数组指针
            N * 3 * sizeof(float),           // ← 拷 N*3 个 float
            0,
            cudaMemcpyHostToDevice
        );
        cudaMemcpyToSymbol(
            (const void*)&NUM_STD_SAMPLES,   // ← 符号 NUM_STD_SAMPLES 的地址
            &N,                              // ← host 端的整数 N
            sizeof(int),
            0,
            cudaMemcpyHostToDevice
        );

        std_samples_loaded = true;                // Set, no more disk reads
    }
} // anonymous namespace

// ──────────────────────────────────────────────────────────────
//  Forward  Wrapper: first ensure samples are loaded, then call original implementation
// ──────────────────────────────────────────────────────────────
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
rasterize_gaussians_autoload(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float          scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float          tan_fovx,
    const float          tan_fovy,
    const int            image_height,
    const int            image_width,
    const torch::Tensor& sh,
    const int            degree,
    const torch::Tensor& campos,
    const bool           prefiltered,
    const bool           debug)
{
    load_std_samples_once();      // Key: only execute once
    return RasterizeGaussiansCUDA(background, means3D, colors, opacity, scales, rotations,
                                  scale_modifier, cov3D_precomp, viewmatrix, projmatrix,
                                  tan_fovx, tan_fovy, image_height, image_width,
                                  sh, degree, campos, prefiltered, debug);
}

// ──────────────────────────────────────────────────────────────
//  Backward  Wrapper: same as forward
// ──────────────────────────────────────────────────────────────
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_gaussians_backward_autoload(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float          scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float          tan_fovx,
    const float          tan_fovy,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& sh,
    const int            degree,
    const torch::Tensor& campos,
    const torch::Tensor& geomBuffer,
    const int            R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const bool           debug)
{
    load_std_samples_once();
    return RasterizeGaussiansBackwardCUDA(background, means3D, radii, colors, opacities, scales, rotations,
                                          scale_modifier, cov3D_precomp, viewmatrix, projmatrix,
                                          tan_fovx, tan_fovy, dL_dout_color, sh, degree, campos,
                                          geomBuffer, R, binningBuffer, imageBuffer, debug);
}


// Upload samples.txt to constant memory
void upload_samples_to_constant(
    py::array_t<float, py::array::c_style | py::array::forcecast> arr,
    int N)
{
    auto info = arr.request();
    float* ptr = static_cast<float*>(info.ptr);
    size_t bytes = size_t(N) * 3 * sizeof(float);

    // —— Use C API, and convert symbol address to const void* ——  
    // Copy actual sample number N
    cudaMemcpyToSymbol(
        (const void*)&NUM_STD_SAMPLES,  // symbol address
        &N,                              // src
        sizeof(int),                     // count
        0,                               // offset
        cudaMemcpyHostToDevice           // kind
    );

    // Copy sample data
    cudaMemcpyToSymbol(
        (const void*)BASE_SAMPLES_MAX,   // symbol address
        ptr,                             // src
        bytes,                           // count = N*3*sizeof(float)
        0,                               // offset
        cudaMemcpyHostToDevice
    );
}

// Download samples from constant memory to Host, return (loadedN,3) numpy array
py::array_t<float> get_samples_from_constant() {
    // 1. Download actual N
    int loadedN = 0;
    cudaMemcpyFromSymbol(
        &loadedN,                       // dst
        (const void*)&NUM_STD_SAMPLES,  // symbol address
        sizeof(int),
        0,
        cudaMemcpyDeviceToHost
    );

    // 2. Download sample data
    std::vector<float> host_data(size_t(loadedN) * 3);
    cudaMemcpyFromSymbol(
        host_data.data(),               // dst
        (const void*)BASE_SAMPLES_MAX,  // symbol address
        size_t(loadedN) * 3 * sizeof(float),
        0,
        cudaMemcpyDeviceToHost
    );

    // 3. Construct and return numpy array
    return py::array_t<float>(
        { loadedN, 3 },               // shape
        { 3 * sizeof(float),          // strides
          sizeof(float) },
        host_data.data()
    );
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
//   m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
//   m.def("mark_visible", &markVisible);

//   // New interface
//   m.def("upload_samples_to_constant", &upload_samples_to_constant,
//         "Upload standard normal samples to GPU constant memory");
//   m.def("get_samples_from_constant", &get_samples_from_constant,
//         "Download samples from GPU constant memory for testing");
//   m.attr("MAX_STD_SAMPLES") = py::int_(MAX_STD_SAMPLES);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // ─── Forward / Backward bind to new auto‑load wrapper ───
    m.def("rasterize_gaussians", &rasterize_gaussians_autoload,
          "Rasterize_Gaussians_forward (auto-load std-normal samples)");
    m.def("rasterize_gaussians_backward", &rasterize_gaussians_backward_autoload,
          "Rasterize_Gaussians_backward (auto-load std-normal samples)");

    // ─── Other existing interfaces ───
    m.def("mark_visible", &markVisible);

    // ─── Manual upload / download sampling points interface (for experiment) ───
    m.def("upload_samples_to_constant", &upload_samples_to_constant,
          "Manually upload standard‑normal samples to GPU constant memory");
    m.def("get_samples_from_constant", &get_samples_from_constant,
          "Download samples from GPU constant memory (debug)");

    // ─── Constant also exported to Python ───
    m.attr("MAX_STD_SAMPLES") = py::int_(MAX_STD_SAMPLES);
}
