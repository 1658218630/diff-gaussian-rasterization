/******************************************************************************
 * test.cu – runtime & gradient check for computeCov2DCUDA
 *
 * 编译示例（需要开启可分离编译才能跨 .cu 调用 device 函数）：
 *     nvcc -g -G -std=c++17 -arch=sm_70 test.cu forward.cu backward.cu \
 *          -o test -rdc=true
 *
 * 运行时创建 1 个高斯，对 computeCov2DCUDA 的解析梯度与
 * 前向数值差分做比较，给出 PASS / FAIL。
 *****************************************************************************/

 #define GLM_COMPILER 0
 #include <cstdio>
 #include <cmath>
 #include <cuda_runtime.h>
 #include <glm/glm.hpp>
 
 #include "forward.h"   // forward 侧的前向投影实现  :contentReference[oaicite:0]{index=0}
 #include "backward.h"  // backward 侧的 preprocess（里面会调用 computeCov2DCUDA）:contentReference[oaicite:1]{index=1}
 
 /*** ---------- forward 侧已有的 device 函数声明 --------- ***/
 struct MeanCov2D { float2 mean; float3 cov; };
 __device__ MeanCov2D computeMeanCov2D_statistical(
         const float3& mean_world,
         float focal_x, float focal_y,
         float tan_fovx, float tan_fovy,
         const float* cov3D,            // 6-float packed Σ₃ᴅ
         const float* viewmatrix);      // 4×4 row-major
 
 /* 把 forward 结果写出来，供有限差分用 */
 __global__ void ForwardKernel(
         int P,
         const float3* means,
         const float*  cov3Ds,
         float fx, float fy,
         float tan_fx, float tan_fy,
         const float* view,
         float4* out_conic)             // (μx, μy, σx², σy²)
 {
     int i = threadIdx.x + blockIdx.x * blockDim.x;
     if (i >= P) return;
 
     MeanCov2D mc = computeMeanCov2D_statistical(means[i], fx, fy,
                                                 tan_fx, tan_fy,
                                                 cov3Ds + 6*i, view);
     out_conic[i] = make_float4(mc.mean.x, mc.mean.y, mc.cov.x, mc.cov.z);
 }
 
 /*** --------- backward 侧核函数声明（在 backward.cu 定义） -------- ***/
 __global__ void computeCov2DCUDA(int P,
         const float3* means, const int* radii,
         const float*  cov3Ds,
         float fx, float fy,
         float tan_fx, float tan_fy,
         const float* view,
         const float* dL_dconics,
         float3* dL_dmeans,
         float*  dL_dcov);
 
 int main()
 {
     /* ---------- 1. 构造最小输入 ---------- */
     constexpr int P = 1;
     float3 h_means[P]   = { make_float3(0.f, 0.f, 5.f) };
     int    h_radii[P]   = { 1 };
     float  h_cov3Ds[6]  = { 0.1f, 0.f, 0.f, 0.1f, 0.f, 0.1f };   // 对称矩阵上三角
     float  h_view[16]   = { 1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1 };
     float  fx = 1.f, fy = 1.f, tan_fx = 1.f, tan_fy = 1.f;
 
     // 上游对 conic 的梯度（只用到 0,1,3 下标）
     float  h_dLdConic[4] = { 1.f, 1.f, 0.f, 1.f };
 
     /* ---------- 2. 设备端 buffer ---------- */
     float3 *d_means, *d_dLdMeans;
     int    *d_radii;
     float  *d_cov3Ds, *d_view, *d_dLdConic, *d_dLdCov;
     float4 *d_conic_base, *d_conic_eps;
 
     cudaMalloc(&d_means, sizeof(h_means));
     cudaMalloc(&d_radii, sizeof(h_radii));
     cudaMalloc(&d_cov3Ds, sizeof(h_cov3Ds));
     cudaMalloc(&d_view, 16*sizeof(float));
     cudaMalloc(&d_dLdConic, sizeof(h_dLdConic));
     cudaMalloc(&d_dLdMeans, sizeof(float3)*P);
     cudaMalloc(&d_dLdCov,   sizeof(float)*6*P);
     cudaMalloc(&d_conic_base, sizeof(float4)*P);
     cudaMalloc(&d_conic_eps,  sizeof(float4)*P);
 
     cudaMemcpy(d_means,  h_means,  sizeof(h_means),  cudaMemcpyHostToDevice);
     cudaMemcpy(d_radii,  h_radii,  sizeof(h_radii),  cudaMemcpyHostToDevice);
     cudaMemcpy(d_cov3Ds, h_cov3Ds,sizeof(h_cov3Ds), cudaMemcpyHostToDevice);
     cudaMemcpy(d_view,   h_view,   16*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(d_dLdConic, h_dLdConic,sizeof(h_dLdConic), cudaMemcpyHostToDevice);
 
     /* ---------- 3. forward 基线 ---------- */
     ForwardKernel<<<1,1>>>(P, d_means, d_cov3Ds,
                            fx, fy, tan_fx, tan_fy,
                            d_view, d_conic_base);
 
     /* ---------- 4. backward（解析梯度） ---------- */
     computeCov2DCUDA<<<1,32>>>(P,
         d_means, d_radii, d_cov3Ds,
         fx, fy, tan_fx, tan_fy,
         d_view, d_dLdConic,
         d_dLdMeans, d_dLdCov);
 
     cudaDeviceSynchronize();
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         printf("CUDA launch error: %s\n", cudaGetErrorString(err));
         return -1;
     }
 
     /* 拷回解析梯度 */
     float3 g_mean;  float g_cov[6];
     cudaMemcpy(&g_mean, d_dLdMeans, sizeof(g_mean), cudaMemcpyDeviceToHost);
     cudaMemcpy(g_cov,   d_dLdCov,   sizeof(g_cov), cudaMemcpyDeviceToHost);
 
     /* ---------- 5. 有限差分：只验证 μ 与 Σzz ---------- */
     const float eps = 1e-3f;
     auto fd = [&](auto setter)
     {
         setter(+eps);
         cudaMemcpy(d_means,  h_means,  sizeof(h_means),  cudaMemcpyHostToDevice);
         cudaMemcpy(d_cov3Ds, h_cov3Ds,sizeof(h_cov3Ds), cudaMemcpyHostToDevice);
         ForwardKernel<<<1,1>>>(P, d_means, d_cov3Ds, fx, fy, tan_fx, tan_fy, d_view, d_conic_eps);
         float4 pos;  cudaMemcpy(&pos, d_conic_eps, sizeof(float4), cudaMemcpyDeviceToHost);
 
         setter(-eps);
         cudaMemcpy(d_means,  h_means,  sizeof(h_means),  cudaMemcpyHostToDevice);
         cudaMemcpy(d_cov3Ds, h_cov3Ds,sizeof(h_cov3Ds), cudaMemcpyHostToDevice);
         ForwardKernel<<<1,1>>>(P, d_means, d_cov3Ds, fx, fy, tan_fx, tan_fy, d_view, d_conic_eps);
         float4 neg;  cudaMemcpy(&neg, d_conic_eps, sizeof(float4), cudaMemcpyDeviceToHost);
 
         setter(0.f);                                     // 还原
         float Lpos = pos.x*h_dLdConic[0] + pos.y*h_dLdConic[1] + pos.w*h_dLdConic[3];
         float Lneg = neg.x*h_dLdConic[0] + neg.y*h_dLdConic[1] + neg.w*h_dLdConic[3];
         return (Lpos - Lneg)/(2*eps);
     };
 
     float g_fd_mu[3];
     g_fd_mu[0] = fd([&](float d){ h_means[0].x += d; });
     g_fd_mu[1] = fd([&](float d){ h_means[0].y += d; });
     g_fd_mu[2] = fd([&](float d){ h_means[0].z += d; });
     float g_fd_covzz = fd([&](float d){ h_cov3Ds[5] += d; });
 
     /* ---------- 6. 打印并判断 ---------- */
     printf("\nAnalytic   dL/dμ = (%f %f %f)\n", g_mean.x, g_mean.y, g_mean.z);
     printf("FiniteDiff dL/dμ = (%f %f %f)\n", g_fd_mu[0], g_fd_mu[1], g_fd_mu[2]);
     printf("Analytic   dL/dΣzz = %f\n", g_cov[5]);
     printf("FiniteDiff dL/dΣzz = %f\n", g_fd_covzz);
 
     const float tol = 1e-2f;
     bool pass = fabsf(g_mean.x-g_fd_mu[0])<tol &&
                 fabsf(g_mean.y-g_fd_mu[1])<tol &&
                 fabsf(g_mean.z-g_fd_mu[2])<tol &&
                 fabsf(g_cov[5] -g_fd_covzz)<tol;
 
     printf("\nGradient check: %s\n", pass ? "PASS" : "FAIL");
     return pass ? 0 : 1;
 }
 