/******************************************************************************
 * test.cu – runtime & gradient check for computeCov2DCUDA
 *
 * Example compilation (requires separable compilation to call device functions across .cu files):
 *     nvcc -g -G -std=c++17 -arch=sm_70 test.cu forward.cu backward.cu \
 *          -o test -rdc=true
 *
 * At runtime, creates a single Gaussian and compares the analytic gradient of
 * computeCov2DCUDA with forward finite differences, reporting PASS / FAIL.
 *****************************************************************************/

 #define GLM_COMPILER 0
 #include <cstdio>
 #include <cmath>
 #include <fstream>
 #include <vector>
 #include <sstream>
 #include <cuda_runtime.h>
 #include <glm/glm.hpp>
 
 #include "statistical_constants.cuh"
 #include "forward.h"   // forward side forward projection implementation
 #include "backward.h"  // backward side preprocess (which calls computeCov2DCUDA)
 
 //=============== CUDA error check macro ===============
 #define CUDA_CHECK(stmt)                                 \
     do {                                                 \
         cudaError_t __err = stmt;                        \
         if (__err != cudaSuccess) {                      \
             fprintf(stderr,                              \
                 "CUDA error %s at %s:%d\n",              \
                 cudaGetErrorString(__err),               \
                 __FILE__, __LINE__);                     \
             exit(EXIT_FAILURE);                          \
         }                                                \
     } while (0)
 
 //=============== forward side helper ================
 struct MeanCov2D { float2 mean; float3 cov; };
 __device__ MeanCov2D computeMeanCov2D_statistical(
         const float3& mean_world,
         float focal_x, float focal_y,
         float tan_fovx, float tan_fovy,
         const float* cov3D,            // 6-float packed Σ₃ᴅ
         const float* viewmatrix);      // 4×4 row-major
 



 __global__ void ForwardKernel(
         int P,
         const float3* means,
         const float*  cov3Ds,
         float fx, float fy,
         float tan_fx, float tan_fy,
         const float* view,
         float4* out_conic)             // (A, B, C, _)
 {
     int i = threadIdx.x + blockIdx.x * blockDim.x;
     if (i >= P) return;

     // 1) Use forward statistical interface to compute 2D covariance Σ₂D = [a b; b c]
     MeanCov2D mc = computeMeanCov2D_statistical(
         means[i], fx, fy, tan_fx, tan_fy,
         cov3Ds + 6*i, view);

     // 2) Add stable constant consistently with forward.cu
     float a = mc.cov.x + 0.3f;  // σ_xx + ε
     float b = mc.cov.y;         // σ_xy
     float c = mc.cov.z + 0.3f;  // σ_yy + ε

     // 3) Invert to get conic parameters (A, B, C)
     float det    = a * c - b * b + 1e-6f;
     float invDet = 1.0f / det;
     float A =  c * invDet;
     float B = -b * invDet;
     float C =  a * invDet;

     out_conic[i] = make_float4(A, B, C, 0.f);
 }


 
 //=============== backward side declaration ===============
 __global__ void computeCov2DCUDA(int P,
         const float3* means, const int* radii,
         const float*  cov3Ds,
         float fx, float fy,
         float tan_fx, float tan_fy,
         const float* view,
         const float* dL_dconics,
         float3* dL_dmeans,
         float*  dL_dcov,
         const bool use_proj_mean);
 
 int main(int argc, char** argv) {
     // ——— 0. Read and copy stdnormal3D_samples.txt ———
     std::ifstream fin("stdnormal3D_samples.txt");
     if (!fin) {
         fprintf(stderr, "Error: cannot open stdnormal3D_samples.txt\n");
         return -1;
     }
     std::vector<float> samples;
     samples.reserve(1024);
     std::string line;
     while (std::getline(fin, line)) {
         std::istringstream iss(line);
         float x, y, z;
         if (iss >> x >> y >> z) {
             samples.push_back(x);
             samples.push_back(y);
             samples.push_back(z);
         }
     }
     fin.close();
 
     int N = int(samples.size()/3);
     if (N <= 0 || N > MAX_STD_SAMPLES) {
         fprintf(stderr,
                 "Error: Invalid sample count N=%d (valid range 1..%d)\n",
                 N, MAX_STD_SAMPLES);
         return -1;
     }
     CUDA_CHECK(cudaMemcpyToSymbol(NUM_STD_SAMPLES, &N, sizeof(int)));
     CUDA_CHECK(cudaMemcpyToSymbol(BASE_SAMPLES_MAX,
                                   samples.data(),
                                   sizeof(float) * samples.size()));
     {
         int N_dev = -1;
         CUDA_CHECK(cudaMemcpyFromSymbol(&N_dev, NUM_STD_SAMPLES, sizeof(int)));
         printf("DEBUG NUM_STD_SAMPLES = %d\n", N_dev);
     }
     {
         float sample0[3] = {0};
         CUDA_CHECK(cudaMemcpyFromSymbol(sample0, BASE_SAMPLES_MAX, 3*sizeof(float)));
         printf("DEBUG BASE_SAMPLES_MAX[0..2] = (%f, %f, %f)\n",
             sample0[0], sample0[1], sample0[2]);
     }
 
     // ——— 1. Construct minimum input ———
     constexpr int P = 1;
     float3 h_means[P]   = { make_float3(0.f, 0.f, 5.f) };
     int    h_radii[P]   = { 1 };
     float  h_cov3Ds[6]  = { 0.1f, 0.f, 0.f, 0.1f, 0.f, 0.1f };   // symmetric upper triangular
     float  h_view[16]   = { 1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1 };
     float  fx = 1.f, fy = 1.f, tan_fx = 1.f, tan_fy = 1.f;
     float  h_dLdConic[4] = { 1.f, 1.f, 0.f, 1.f };
 
     // ——— 2. Device buffer allocation ———
     float3 *d_means, *d_dLdMeans;
     int    *d_radii;
     float  *d_cov3Ds, *d_view, *d_dLdConic, *d_dLdCov;
     float4 *d_conic_base, *d_conic_eps;
 
     CUDA_CHECK(cudaMalloc(&d_means,        sizeof(h_means)));
     CUDA_CHECK(cudaMalloc(&d_radii,        sizeof(h_radii)));
     CUDA_CHECK(cudaMalloc(&d_cov3Ds,       sizeof(h_cov3Ds)));
     CUDA_CHECK(cudaMalloc(&d_view,         16*sizeof(float)));
     CUDA_CHECK(cudaMalloc(&d_dLdConic,     sizeof(h_dLdConic)));
     CUDA_CHECK(cudaMalloc(&d_dLdMeans,     sizeof(float3)*P));
     CUDA_CHECK(cudaMalloc(&d_dLdCov,       sizeof(float)*6*P));
     CUDA_CHECK(cudaMalloc(&d_conic_base,   sizeof(float4)*P));
     CUDA_CHECK(cudaMalloc(&d_conic_eps,    sizeof(float4)*P));
 
     CUDA_CHECK(cudaMemcpy(d_means,    h_means,    sizeof(h_means),    cudaMemcpyHostToDevice));
     CUDA_CHECK(cudaMemcpy(d_radii,    h_radii,    sizeof(h_radii),    cudaMemcpyHostToDevice));
     CUDA_CHECK(cudaMemcpy(d_cov3Ds,   h_cov3Ds,   sizeof(h_cov3Ds),   cudaMemcpyHostToDevice));
     CUDA_CHECK(cudaMemcpy(d_view,     h_view,     16*sizeof(float),    cudaMemcpyHostToDevice));
     CUDA_CHECK(cudaMemcpy(d_dLdConic, h_dLdConic, sizeof(h_dLdConic),  cudaMemcpyHostToDevice));
 
     //=============== 3. ForwardKernel baseline ===============
     ForwardKernel<<<1,1>>>(P, d_means, d_cov3Ds,
                            fx, fy, tan_fx, tan_fy,
                            d_view, d_conic_base);
     CUDA_CHECK(cudaDeviceSynchronize());
     CUDA_CHECK(cudaGetLastError());
 
     //=============== 4. computeCov2DCUDA (analytic gradient) ===============
     computeCov2DCUDA<<<1,32>>>(P,
         d_means, d_radii, d_cov3Ds,
         fx, fy, tan_fx, tan_fy,
         d_view, d_dLdConic,
         d_dLdMeans, d_dLdCov,
         true);
     CUDA_CHECK(cudaDeviceSynchronize());
     CUDA_CHECK(cudaGetLastError());
 
     // copy back analytic gradient
     float3 g_mean;  float g_cov[6];
     CUDA_CHECK(cudaMemcpy(&g_mean, d_dLdMeans, sizeof(g_mean), cudaMemcpyDeviceToHost));
     CUDA_CHECK(cudaMemcpy(g_cov,   d_dLdCov,   sizeof(g_cov),  cudaMemcpyDeviceToHost));
 
     //=============== 5. Finite difference: central difference & error criterion ===============
     const float eps  = 5e-3f;
     const float rtol = 1e-3f;
     const float atol = 1e-5f;
 
     auto compute_fd = [&](auto setter)->float {
        // 1. save original value
        float3  mean_backup = h_means[0];
        float   cov_backup  = h_cov3Ds[5];
    
        // 2. +eps
        setter(+eps);                                   // baseline + ε
        CUDA_CHECK(cudaMemcpy(d_means,  h_means, sizeof(h_means),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cov3Ds, h_cov3Ds,sizeof(h_cov3Ds), cudaMemcpyHostToDevice));
        ForwardKernel<<<1,1>>>(P, d_means, d_cov3Ds, fx, fy, tan_fx, tan_fy, d_view, d_conic_eps);
        CUDA_CHECK(cudaDeviceSynchronize());
        float4 pos; CUDA_CHECK(cudaMemcpy(&pos, d_conic_eps, sizeof(pos), cudaMemcpyDeviceToHost));
        float Lpos = pos.x*h_dLdConic[0] + pos.y*h_dLdConic[1] + pos.z*h_dLdConic[3];
        
    
        // 3. restore original value
        h_means[0]  = mean_backup;
        h_cov3Ds[5] = cov_backup;
    
        // 4. -eps
        setter(-eps);                                   // baseline − ε
        CUDA_CHECK(cudaMemcpy(d_means,  h_means, sizeof(h_means),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cov3Ds, h_cov3Ds,sizeof(h_cov3Ds), cudaMemcpyHostToDevice));
        ForwardKernel<<<1,1>>>(P, d_means, d_cov3Ds, fx, fy, tan_fx, tan_fy, d_view, d_conic_eps);
        CUDA_CHECK(cudaDeviceSynchronize());
        float4 neg; CUDA_CHECK(cudaMemcpy(&neg, d_conic_eps, sizeof(neg), cudaMemcpyDeviceToHost));
        float Lneg = neg.x*h_dLdConic[0] + neg.y*h_dLdConic[1] + neg.z*h_dLdConic[3];
    
        // 5. restore original value for next call
        h_means[0]  = mean_backup;
        h_cov3Ds[5] = cov_backup;
    
        //return (Lpos - Lneg) / (2.f * eps);
        return float((Lpos - Lneg) / (2.0 * eps));
    };
    

 
     float g_fd_mu[3];
     g_fd_mu[0] = compute_fd([&](float d){ h_means[0].x += d; });
     g_fd_mu[1] = compute_fd([&](float d){ h_means[0].y += d; });
     g_fd_mu[2] = compute_fd([&](float d){ h_means[0].z += d; });
     float g_fd_covzz = compute_fd([&](float d){ h_cov3Ds[5] += d; });

 
     //=============== 6. Print & determine ===============
     printf("\nAnalytic   dL/dμ = (% .6f % .6f % .6f)\n", g_mean.x, g_mean.y, g_mean.z);
     printf("FiniteDiff dL/dμ = (% .6f % .6f % .6f)\n", g_fd_mu[0], g_fd_mu[1], g_fd_mu[2]);
     printf("Analytic   dL/dΣzz = % .6f\n", g_cov[5]);
     printf("FiniteDiff dL/dΣzz = % .6f\n", g_fd_covzz);
 
     bool pass = true;
     auto check = [&](float ana, float fd, const char* name){
         float diff = fabsf(ana - fd);
         float denom = fmaxf(fmaxf(fabsf(ana), fabsf(fd)), 1.0f);
         float rel  = diff / denom;
         if (diff > atol && rel > rtol) {
             printf("Mismatch %s  analytic=% .6g  FD=% .6g  abs=% .6g  rel=% .6g\n",
                    name, ana, fd, diff, rel);
             pass = false;
         }
     };
     check(g_mean.x, g_fd_mu[0], "dL/dμ.x");
     check(g_mean.y, g_fd_mu[1], "dL/dμ.y");
     check(g_mean.z, g_fd_mu[2], "dL/dμ.z");
     check(g_cov[5], g_fd_covzz, "dL/dΣzz");
 
     printf("\nGradient check: %s\n", pass ? "PASS" : "FAIL");
 
     CUDA_CHECK(cudaDeviceReset());
     return pass ? 0 : 1;
 }
 