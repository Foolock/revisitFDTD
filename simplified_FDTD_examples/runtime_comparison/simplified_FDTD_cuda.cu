#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define PI 3.14159265359f

#define eps0 1.0f
#define mu0 1.0f
#define eta0 std::sqrt(mu0 / eps0)
#define c0 (1.0f / std::sqrt(mu0 * eps0))
#define hbar 1.0f

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#define CUDACHECK(call)                                                        \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if(err__ != cudaSuccess) {                                                 \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__          \
                << " -> " << cudaGetErrorString(err__) << "\n";             \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while(0)

class Complex {
public:
  float real, imag;

  Complex() : real(0.0f), imag(0.0f) {}
  Complex(float r, float i) : real(r), imag(i) {}

  Complex operator+(const Complex& b) const {
    return Complex(real + b.real, imag + b.imag);
  }

  Complex operator-(const Complex& b) const {
    return Complex(real - b.real, imag - b.imag);
  }

  Complex operator*(const Complex& b) const {
    return Complex(real * b.real - imag * b.imag,
                   real * b.imag + imag * b.real);
  }

  Complex operator/(const Complex& b) const {
    float den = b.real * b.real + b.imag * b.imag;
    return Complex((real * b.real + imag * b.imag) / den,
                   (imag * b.real - real * b.imag) / den);
  }

  Complex operator*(float b) const {
    return Complex(real * b, imag * b);
  }
};

static void set_FDTD_matrices_3D_structure(
  std::vector<float>& Cax, std::vector<float>& Cbx,
  std::vector<float>& Cay, std::vector<float>& Cby,
  std::vector<float>& Caz, std::vector<float>& Cbz,
  std::vector<float>& Dax, std::vector<float>& Dbx,
  std::vector<float>& Day, std::vector<float>& Dby,
  std::vector<float>& Daz, std::vector<float>& Dbz,
  int Nx, int Ny, int Nz,
  float dx, float dt,
  bool* mask,
  Complex eps_air,
  Complex eps_structure,
  int k_min, int k_max,
  float OMEGA0,
  int t_PML
) {
  float a_max = 2.0f;
  int p = 3;
  float sigma_max = -(p + 1) * std::log(1e-5f) / (2.0f * eta0 * t_PML * dx);

  for(int i = 0; i < Nx; ++i) {
    for(int j = 0; j < Ny; ++j) {
      for(int k = 0; k < Nz; ++k) {

        Complex eps_r = eps_air;
        Complex mu_r(1.0f, 0.0f);

        int idx = i + j * Nx + k * Nx * Ny;
        int idx_2D = i + j * Nx;

        if(k_min <= k && k <= k_max) {
          if(mask[idx_2D]) {
            eps_r = eps_structure;
          }
        }

        Complex sx(1.0f, 0.0f);
        Complex sy(1.0f, 0.0f);
        Complex sz(1.0f, 0.0f);

        float bound_x_dist = 0.0f;
        if(i < t_PML) {
          bound_x_dist = 1.0f - float(i) / t_PML;
        }
        if(i + t_PML >= Nx) {
          bound_x_dist = 1.0f - float(Nx - i - 1) / t_PML;
        }
        sx.real = 1.0f + a_max * std::pow(bound_x_dist, p);
        sx.imag = sigma_max * std::pow(bound_x_dist, p) / (OMEGA0 * eps0);

        float bound_y_dist = 0.0f;
        if(j < t_PML) {
          bound_y_dist = 1.0f - float(j) / t_PML;
        }
        if(j + t_PML >= Ny) {
          bound_y_dist = 1.0f - float(Ny - j - 1) / t_PML;
        }
        sy.real = 1.0f + a_max * std::pow(bound_y_dist, p);
        sy.imag = sigma_max * std::pow(bound_y_dist, p) / (OMEGA0 * eps0);

        float bound_z_dist = 0.0f;
        if(k < t_PML) {
          bound_z_dist = 1.0f - float(k) / t_PML;
        }
        if(k + t_PML >= Nz) {
          bound_z_dist = 1.0f - float(Nz - k - 1) / t_PML;
        }
        sz.real = 1.0f + a_max * std::pow(bound_z_dist, p);
        sz.imag = sigma_max * std::pow(bound_z_dist, p) / (OMEGA0 * eps0);

        Complex eps_xx_complex = (eps_r * sy * sz) / sx;
        Complex eps_yy_complex = (eps_r * sx * sz) / sy;
        Complex eps_zz_complex = (eps_r * sx * sy) / sz;

        Complex mu_xx_complex = (mu_r * sy * sz) / sx;
        Complex mu_yy_complex = (mu_r * sx * sz) / sy;
        Complex mu_zz_complex = (mu_r * sx * sy) / sz;

        float eps_xx = eps_xx_complex.real;
        if(eps_xx < 1.0f) eps_xx = 1.0f;
        float eps_yy = eps_yy_complex.real;
        if(eps_yy < 1.0f) eps_yy = 1.0f;
        float eps_zz = eps_zz_complex.real;
        if(eps_zz < 1.0f) eps_zz = 1.0f;

        float mu_xx = mu_xx_complex.real;
        if(mu_xx < 1.0f) mu_xx = 1.0f;
        float mu_yy = mu_yy_complex.real;
        if(mu_yy < 1.0f) mu_yy = 1.0f;
        float mu_zz = mu_zz_complex.real;
        if(mu_zz < 1.0f) mu_zz = 1.0f;

        float sigma_e_xx = std::abs(OMEGA0 * eps0 * eps_xx_complex.imag);
        float sigma_e_yy = std::abs(OMEGA0 * eps0 * eps_yy_complex.imag);
        float sigma_e_zz = std::abs(OMEGA0 * eps0 * eps_zz_complex.imag);

        float sigma_h_xx = std::abs(OMEGA0 * mu0 * mu_xx_complex.imag);
        float sigma_h_yy = std::abs(OMEGA0 * mu0 * mu_yy_complex.imag);
        float sigma_h_zz = std::abs(OMEGA0 * mu0 * mu_zz_complex.imag);

        float tmp_x = sigma_e_xx * dt / (2.0f * eps_xx * eps0);
        Cax[idx] = (1.0f - tmp_x) / (1.0f + tmp_x);
        Cbx[idx] = (dt / (eps_xx * eps0)) / (1.0f + tmp_x) / dx;

        float tmp_y = sigma_e_yy * dt / (2.0f * eps_yy * eps0);
        Cay[idx] = (1.0f - tmp_y) / (1.0f + tmp_y);
        Cby[idx] = (dt / (eps_yy * eps0)) / (1.0f + tmp_y) / dx;

        float tmp_z = sigma_e_zz * dt / (2.0f * eps_zz * eps0);
        Caz[idx] = (1.0f - tmp_z) / (1.0f + tmp_z);
        Cbz[idx] = (dt / (eps_zz * eps0)) / (1.0f + tmp_z) / dx;

        tmp_x = sigma_h_xx * dt / (2.0f * mu_xx * mu0);
        Dax[idx] = (1.0f - tmp_x) / (1.0f + tmp_x);
        Dbx[idx] = (dt / (mu_xx * mu0)) / (1.0f + tmp_x) / dx;

        tmp_y = sigma_h_yy * dt / (2.0f * mu_yy * mu0);
        Day[idx] = (1.0f - tmp_y) / (1.0f + tmp_y);
        Dby[idx] = (dt / (mu_yy * mu0)) / (1.0f + tmp_y) / dx;

        tmp_z = sigma_h_zz * dt / (2.0f * mu_zz * mu0);
        Daz[idx] = (1.0f - tmp_z) / (1.0f + tmp_z);
        Dbz[idx] = (dt / (mu_zz * mu0)) / (1.0f + tmp_z) / dx;
      }
    }
  }
}

__global__ void updateE_3Dmap_fix(float *Ex, float *Ey, float *Ez,
                                  float *Hx, float *Hy, float *Hz,
                                  float *Cax, float *Cbx, float *Cay,
                                  float *Cby, float *Caz, float *Cbz,
                                  float *Jx, float *Jy, float *Jz,
                                  float dx, int Nx, int Ny, int Nz) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int i = tid % Nx;
  unsigned int j = (tid % (Nx * Ny)) / Nx;
  unsigned int k = tid / (Nx * Ny);

  if(i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1) {
    int idx = i + j * Nx + k * (Nx * Ny);

    Ex[idx] = Cax[idx] * Ex[idx] + Cbx[idx] *
              ((Hz[idx] - Hz[idx - Nx]) - (Hy[idx] - Hy[idx - Nx * Ny]) - Jx[idx] * dx);

    Ey[idx] = Cay[idx] * Ey[idx] + Cby[idx] *
              ((Hx[idx] - Hx[idx - Nx * Ny]) - (Hz[idx] - Hz[idx - 1]) - Jy[idx] * dx);

    Ez[idx] = Caz[idx] * Ez[idx] + Cbz[idx] *
              ((Hy[idx] - Hy[idx - 1]) - (Hx[idx] - Hx[idx - Nx]) - Jz[idx] * dx);
  }
}

__global__ void updateH_3Dmap_fix(float *Ex, float *Ey, float *Ez,
                                  float *Hx, float *Hy, float *Hz,
                                  float *Dax, float *Dbx,
                                  float *Day, float *Dby,
                                  float *Daz, float *Dbz,
                                  float *Mx, float *My, float *Mz,
                                  float dx, int Nx, int Ny, int Nz) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int i = tid % Nx;
  unsigned int j = (tid % (Nx * Ny)) / Nx;
  unsigned int k = tid / (Nx * Ny);

  if(i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1) {
    int idx = i + j * Nx + k * (Nx * Ny);

    Hx[idx] = Dax[idx] * Hx[idx] + Dbx[idx] *
              ((Ey[idx + Nx * Ny] - Ey[idx]) - (Ez[idx + Nx] - Ez[idx]) - Mx[idx] * dx);

    Hy[idx] = Day[idx] * Hy[idx] + Dby[idx] *
              ((Ez[idx + 1] - Ez[idx]) - (Ex[idx + Nx * Ny] - Ex[idx]) - My[idx] * dx);

    Hz[idx] = Daz[idx] * Hz[idx] + Dbz[idx] *
              ((Ex[idx + Nx] - Ex[idx]) - (Ey[idx + 1] - Ey[idx]) - Mz[idx] * dx);
  }
}

class gDiamondGPU {
public:
  gDiamondGPU(size_t Nx, size_t Ny, size_t Nz)
    : _Nx(Nx), _Ny(Ny), _Nz(Nz),
      _Ex_gpu(Nx * Ny * Nz, 0.0f), _Ey_gpu(Nx * Ny * Nz, 0.0f), _Ez_gpu(Nx * Ny * Nz, 0.0f),
      _Hx_gpu(Nx * Ny * Nz, 0.0f), _Hy_gpu(Nx * Ny * Nz, 0.0f), _Hz_gpu(Nx * Ny * Nz, 0.0f),
      _Jx(Nx * Ny * Nz, 0.0f), _Jy(Nx * Ny * Nz, 0.0f), _Jz(Nx * Ny * Nz, 0.0f),
      _Mx(Nx * Ny * Nz, 0.0f), _My(Nx * Ny * Nz, 0.0f), _Mz(Nx * Ny * Nz, 0.0f),
      _Cax(Nx * Ny * Nz, 0.0f), _Cay(Nx * Ny * Nz, 0.0f), _Caz(Nx * Ny * Nz, 0.0f),
      _Cbx(Nx * Ny * Nz, 0.0f), _Cby(Nx * Ny * Nz, 0.0f), _Cbz(Nx * Ny * Nz, 0.0f),
      _Dax(Nx * Ny * Nz, 0.0f), _Day(Nx * Ny * Nz, 0.0f), _Daz(Nx * Ny * Nz, 0.0f),
      _Dbx(Nx * Ny * Nz, 0.0f), _Dby(Nx * Ny * Nz, 0.0f), _Dbz(Nx * Ny * Nz, 0.0f) {

    const float um = 1.0f;
    const float nm = um / 1.0e3f;

    SOURCE_WAVELENGTH = 1.0f * um;
    SOURCE_FREQUENCY = c0 / SOURCE_WAVELENGTH;
    SOURCE_OMEGA = 2.0f * PI * SOURCE_FREQUENCY;
    _dx = SOURCE_WAVELENGTH / 10.0f;
    dt = 0.05f;

    J_source_amp = 5e4f;
    M_source_amp = J_source_amp * std::pow(eta0, 3.0f);

    freq_sigma = 0.05f * SOURCE_FREQUENCY;
    t_sigma = 1.0f / freq_sigma / (2.0f * PI);
    t_peak = 5.0f * t_sigma;

    _source_idx = Nx / 2 + (Ny / 2) * Nx + (Nz / 2) * Nx * Ny;

    bool* mask = (bool*)std::malloc(Nx * Ny * sizeof(bool));
    std::memset(mask, 0, Nx * Ny * sizeof(bool));

    Complex eps_air(1.0f, 0.0f);
    Complex eps_Si(12.0f, 0.001f);

    float t_slab = 200.0f * nm;
    int t_slab_grid = static_cast<int>(std::round(t_slab / _dx));
    int k_mid = static_cast<int>(Nz / 2);
    int slab_k_min = k_mid - t_slab_grid / 2;
    int slab_k_max = slab_k_min + t_slab_grid;

    float h_PML = 1.0f * um;
    int t_PML = static_cast<int>(std::ceil(h_PML / _dx));

    set_FDTD_matrices_3D_structure(
      _Cax, _Cbx, _Cay, _Cby, _Caz, _Cbz,
      _Dax, _Dbx, _Day, _Dby, _Daz, _Dbz,
      static_cast<int>(Nx), static_cast<int>(Ny), static_cast<int>(Nz),
      _dx, dt, mask, eps_air, eps_Si,
      slab_k_min, slab_k_max, SOURCE_OMEGA, t_PML
    );

    std::free(mask);
  }

  void update_FDTD_gpu(size_t num_timesteps) {

    const size_t N = _Nx * _Ny * _Nz;

    float *Ex, *Ey, *Ez, *Hx, *Hy, *Hz;
    float *Jx, *Jy, *Jz, *Mx, *My, *Mz;
    float *Cax, *Cbx, *Cay, *Cby, *Caz, *Cbz;
    float *Dax, *Dbx, *Day, *Dby, *Daz, *Dbz;

    CUDACHECK(cudaMalloc(&Ex, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Ey, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Ez, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Hx, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Hy, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Hz, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Jx, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Jy, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Jz, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Mx, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&My, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Mz, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Cax, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Cbx, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Cay, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Cby, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Caz, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Cbz, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Dax, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Dbx, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Day, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Dby, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Daz, sizeof(float) * N));
    CUDACHECK(cudaMalloc(&Dbz, sizeof(float) * N));

    CUDACHECK(cudaMemset(Ex, 0, sizeof(float) * N));
    CUDACHECK(cudaMemset(Ey, 0, sizeof(float) * N));
    CUDACHECK(cudaMemset(Ez, 0, sizeof(float) * N));
    CUDACHECK(cudaMemset(Hx, 0, sizeof(float) * N));
    CUDACHECK(cudaMemset(Hy, 0, sizeof(float) * N));
    CUDACHECK(cudaMemset(Hz, 0, sizeof(float) * N));
    CUDACHECK(cudaMemset(Jx, 0, sizeof(float) * N));
    CUDACHECK(cudaMemset(Jy, 0, sizeof(float) * N));
    CUDACHECK(cudaMemset(Jz, 0, sizeof(float) * N));
    CUDACHECK(cudaMemset(Mx, 0, sizeof(float) * N));
    CUDACHECK(cudaMemset(My, 0, sizeof(float) * N));
    CUDACHECK(cudaMemset(Mz, 0, sizeof(float) * N));

    std::chrono::duration<double> h2d_runtime(0.0);
    auto h2d_start = std::chrono::high_resolution_clock::now();

    CUDACHECK(cudaMemcpy(Cax, _Cax.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(Cay, _Cay.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(Caz, _Caz.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(Cbx, _Cbx.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(Cby, _Cby.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(Cbz, _Cbz.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(Dax, _Dax.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(Day, _Day.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(Daz, _Daz.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(Dbx, _Dbx.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(Dby, _Dby.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(Dbz, _Dbz.data(), sizeof(float) * N, cudaMemcpyHostToDevice));

    // no need to synchronize as we are using blocking memcpy

    auto h2d_end = std::chrono::high_resolution_clock::now();
    h2d_runtime += h2d_end - h2d_start;

    size_t block_size = BLOCK_SIZE;
    size_t grid_size = (N + block_size - 1) / block_size;

    std::chrono::duration<double> kernel_runtime(0.0);
    auto kernel_start = std::chrono::high_resolution_clock::now();

    for(size_t t = 0; t < num_timesteps; ++t) {

      float Mz_value = M_source_amp * sinf(SOURCE_OMEGA * static_cast<float>(t) * dt);
      cudaMemcpy(Mz + _source_idx, &Mz_value, sizeof(float), cudaMemcpyHostToDevice);

      updateE_3Dmap_fix<<<grid_size, block_size>>>(
        Ex, Ey, Ez,
        Hx, Hy, Hz,
        Cax, Cbx, Cay, Cby, Caz, Cbz,
        Jx, Jy, Jz,
        _dx, static_cast<int>(_Nx), static_cast<int>(_Ny), static_cast<int>(_Nz)
      );

      updateH_3Dmap_fix<<<grid_size, block_size>>>(
        Ex, Ey, Ez,
        Hx, Hy, Hz,
        Dax, Dbx, Day, Dby, Daz, Dbz,
        Mx, My, Mz,
        _dx, static_cast<int>(_Nx), static_cast<int>(_Ny), static_cast<int>(_Nz)
      );

    }

    CUDACHECK(cudaDeviceSynchronize());
    auto kernel_end = std::chrono::high_resolution_clock::now();
    kernel_runtime += kernel_end - kernel_start;

    std::chrono::duration<double> d2h_runtime(0.0);
    auto d2h_start = std::chrono::high_resolution_clock::now();

    CUDACHECK(cudaMemcpy(_Ex_gpu.data(), Ex, sizeof(float) * N, cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(_Ey_gpu.data(), Ey, sizeof(float) * N, cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(_Ez_gpu.data(), Ez, sizeof(float) * N, cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(_Hx_gpu.data(), Hx, sizeof(float) * N, cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(_Hy_gpu.data(), Hy, sizeof(float) * N, cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(_Hz_gpu.data(), Hz, sizeof(float) * N, cudaMemcpyDeviceToHost));

    auto d2h_end = std::chrono::high_resolution_clock::now();
    d2h_runtime += d2h_end - d2h_start;

    std::chrono::duration<double> total_runtime(0.0);
    total_runtime = h2d_runtime + kernel_runtime + d2h_runtime;

    std::cout << "end to end throughput: "
              << (_Nx * _Ny * _Nz / 1.0e6 * num_timesteps) / total_runtime.count()
              << " Mcells/s\n";
    std::cout << "kernel throughput: "
              << (_Nx * _Ny * _Nz / 1.0e6 * num_timesteps) / kernel_runtime.count()
              << " Mcells/s\n";

    double total_s  = total_runtime.count();
    double h2d_s    = h2d_runtime.count();
    double kernel_s = kernel_runtime.count();
    double d2h_s    = d2h_runtime.count();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "runtime breakdown (%):\n";
    std::cout << "  H2D memcpy : " << (100.0 * h2d_s / total_s) << "%\n";
    std::cout << "  kernels    : " << (100.0 * kernel_s / total_s) << "%\n";
    std::cout << "  D2H memcpy : " << (100.0 * d2h_s / total_s) << "%\n";

    CUDACHECK(cudaFree(Ex));
    CUDACHECK(cudaFree(Ey));
    CUDACHECK(cudaFree(Ez));
    CUDACHECK(cudaFree(Hx));
    CUDACHECK(cudaFree(Hy));
    CUDACHECK(cudaFree(Hz));
    CUDACHECK(cudaFree(Jx));
    CUDACHECK(cudaFree(Jy));
    CUDACHECK(cudaFree(Jz));
    CUDACHECK(cudaFree(Mx));
    CUDACHECK(cudaFree(My));
    CUDACHECK(cudaFree(Mz));
    CUDACHECK(cudaFree(Cax));
    CUDACHECK(cudaFree(Cbx));
    CUDACHECK(cudaFree(Cay));
    CUDACHECK(cudaFree(Cby));
    CUDACHECK(cudaFree(Caz));
    CUDACHECK(cudaFree(Cbz));
    CUDACHECK(cudaFree(Dax));
    CUDACHECK(cudaFree(Dbx));
    CUDACHECK(cudaFree(Day));
    CUDACHECK(cudaFree(Dby));
    CUDACHECK(cudaFree(Daz));
    CUDACHECK(cudaFree(Dbz));
  }

private:
  size_t _Nx, _Ny, _Nz;
  size_t _source_idx;

  float _dx {0.0f};
  float dt {0.0f};

  float SOURCE_WAVELENGTH {0.0f};
  float SOURCE_FREQUENCY {0.0f};
  float SOURCE_OMEGA {0.0f};

  float J_source_amp {0.0f};
  float M_source_amp {0.0f};

  float freq_sigma {0.0f};
  float t_sigma {0.0f};
  float t_peak {0.0f};

  std::vector<float> _Ex_gpu, _Ey_gpu, _Ez_gpu;
  std::vector<float> _Hx_gpu, _Hy_gpu, _Hz_gpu;

  std::vector<float> _Jx, _Jy, _Jz;
  std::vector<float> _Mx, _My, _Mz;

  std::vector<float> _Cax, _Cay, _Caz;
  std::vector<float> _Cbx, _Cby, _Cbz;
  std::vector<float> _Dax, _Day, _Daz;
  std::vector<float> _Dbx, _Dby, _Dbz;
};

int main(int argc, char** argv) {
  size_t Nx = 32;
  size_t Ny = 32;
  size_t Nz = 32;
  size_t num_timesteps = 100;

  if(argc == 5) {
    Nx = static_cast<size_t>(std::stoul(argv[1]));
    Ny = static_cast<size_t>(std::stoul(argv[2]));
    Nz = static_cast<size_t>(std::stoul(argv[3]));
    num_timesteps = static_cast<size_t>(std::stoul(argv[4]));
  }
  else {
    std::cout << "Usage: " << argv[0] << " Nx Ny Nz num_timesteps\n";
    std::cout << "Using default: 32 32 32 100\n";
  }

  gDiamondGPU sim(Nx, Ny, Nz);
  sim.update_FDTD_gpu(num_timesteps);

  return 0;
}
