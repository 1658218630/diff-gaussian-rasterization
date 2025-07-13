#include <vector>
#include <cstdio>
#include <cmath>

// 声明 Forward/Backward 接口
extern "C" {

    // ——— forward.cu 中的 preprocessCUDA kernel ———
    // template<int C> 在源文件中定义，此处不再重复 template 声明
    // Launch 时请显式指定 C 的值，例如 preprocessCUDA<3><<<grid,block>>>(…);
    __global__ void preprocessCUDA(
        int P,                      // 总共的 Gaussian 数量
        int D,                      // 聚类数量（见 forward.h）
        int M,                      // 样本数 N_SAMPLES
        const float*     orig_points,    // [3*P]
        const glm::vec3* scales,          // [P]
        float            scale_modifier,
        const glm::vec4* rotations,       // [P]
        const float*     opacities,       // [P]
        const float*     shs,             // [P*#SH_COEFF]
        bool*            clamped,         // [3*P] 剪裁标记
        const float*     cov3D_precomp,   // [6*P]
        const float*     colors_precomp,  // [P*3]
        const float*     viewmatrix,      // [16] 4×4
        const float*     projmatrix,      // [16] 4×4
        const glm::vec3* cam_pos,         // [P]（可广播）
        int              W,               // 渲染宽
        int              H,               // 渲染高
        float            focal_x,
        float            focal_y,
        float            tan_fovx,
        float            tan_fovy,
        int*             radii,           // [P]
        float2*          points_xy_image, // [P]
        float*           depths,          // [P]
        float*           cov3Ds,          // [6*P] 输出 Σw
        float*           colors,          // [3*P] 输出颜色
        float4*          conic_opacity,   // [P]
        const dim3       grid,
        bool             antialiasing,
        uint32_t*        tiles_touched,   // [P]
        bool             prefiltered,
        bool             use_proj_mean    // 选择投影均值算法
    );
    
    // ——— forward.cu 中的 computeMeanCov2D_statistical 设备函数 ———
    struct MeanCov2D {
        float2 mean;   // (ū, v̄)
        float3 cov;    // (σx², cov_xy, σy²)
    };
    __device__ MeanCov2D computeMeanCov2D_statistical(
        const float3& mean_world,    // 世界坐标下的 μw
        float          focal_x,
        float          focal_y,
        float          tan_fovx,
        float          tan_fovy,
        const float*   cov3D,        // [6] 世界协方差 Σw（packed）
        const float*   viewmatrix    // [16] 4×4 row-major
    );
    
    // ——— backward.cu 中的 computeCov2DCUDA kernel ———
    __global__ void computeCov2DCUDA(
        int               P,            // Gaussian 数量
        const float3*     means,        // [P] μw
        const int*        radii,        // [P]
        const float*      cov3Ds,       // [6*P] Σw（packed）
        float             h_x,          // = focal_x
        float             h_y,          // = focal_y
        float             tan_fovx,
        float             tan_fovy,
        const float*      view_matrix,  // [16]
        const float*      dL_dconics,   // [4*P] 上游对投影 conic 的梯度
        float3*           dL_dmeans,    // [P] 输出 dL/dμw
        float*            dL_dcov        // [6*P] 输出 dL/dΣw（packed）
    );
    
    } // extern "C"
    

// 一个简单 forward+loss：只对 u_bar、v_bar 做 L2
float forward_and_loss(
  const std::vector<float>& y_flat,
  int N,
  std::vector<float>& u_bar,
  std::vector<float>& v_bar,
  std::vector<float>& cov_w)
{
  // 1) 调用你的前向核
  preprocessCUDA(y_flat.data(), N,
                 /*y_mean*/nullptr, /*y_cov*/nullptr);
  computeMeanCov2D_statistical(
    y_flat.data(), N,
    u_bar.data(), v_bar.data(),
    cov_w.data());

  // 2) 计算 loss = Σ (u_bar^2 + v_bar^2)
  float L = 0.f;
  for(int i = 0; i < N; ++i){
    L += u_bar[i]*u_bar[i] + v_bar[i]*v_bar[i];
  }
  return L;
}

int main(){
  const int N = 128;
  // 随机初始化 y
  std::vector<float> y_flat(2*N);
  for(auto& v : y_flat) v = rand()/(float)RAND_MAX;

  // 准备梯度 buffer
  std::vector<float> u_bar(N), v_bar(N), cov_w(6*N);
  // 第一次前向拿 u_bar,v_bar
  float L = forward_and_loss(y_flat, N, u_bar, v_bar, cov_w);

  // upstream grad dL/du_bar=2*u_bar, dL/dv_bar=2*v_bar
  std::vector<float> dL_du_bar(N), dL_dv_bar(N);
  for(int i=0; i<N; ++i){
    dL_du_bar[i] = 2.f * u_bar[i];
    dL_dv_bar[i] = 2.f * v_bar[i];
  }

  // analytic grad
  std::vector<float> dL_dy(2*N, 0.f);
  computeCov2DCUDA(
    dL_du_bar.data(), dL_dv_bar.data(),
    /*dL_dcov=*/nullptr,
    /*dL_dmeans=*/nullptr,
    dL_dy.data(), N
    /*…传入 W, mu_w, cov_w buffer 等*/);

  // 做 finite-difference
  float eps = 1e-3f;
  for(int i=0; i<N; ++i){
    for(int k=0; k<2; ++k){
      auto y_pos = y_flat, y_neg = y_flat;
      y_pos[2*i+k] += eps;
      y_neg[2*i+k] -= eps;
      std::vector<float> dummy1(N), dummy2(N), dummycov(6*N);
      float Lp = forward_and_loss(y_pos, N, dummy1, dummy2, dummycov);
      float Ln = forward_and_loss(y_neg, N, dummy1, dummy2, dummycov);
      float num = (Lp - Ln)/(2*eps);
      printf("i=%d k=%d → ana=%.6f  num=%.6f  diff=%.6e\n",
             i, k, dL_dy[2*i+k], num,
             fabs(dL_dy[2*i+k]-num));
    }
  }
  return 0;
}
