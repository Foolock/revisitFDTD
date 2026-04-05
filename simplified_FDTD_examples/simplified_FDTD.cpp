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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define PI 3.14159265359f

#define eps0 1.0f
#define mu0 1.0f
#define eta0 std::sqrt(mu0 / eps0)
#define c0 (1.0f / std::sqrt(mu0 * eps0))
#define hbar 1.0f

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

static void save_field_png(float* u, const char filename[], int Nx, int Ny, float vmax) {
  unsigned char* img = (unsigned char*)std::malloc(Nx * Ny * 3 * sizeof(unsigned char));
  if(img == nullptr) {
    std::perror("Failed to allocate memory for img");
    return;
  }

  for(int i = 0; i < Nx; ++i) {
    for(int j = 0; j < Ny; ++j) {
      int idx_field = i + j * Nx;
      int idx_img = idx_field * 3;

      float value = u[idx_field] / vmax;

      unsigned char red, green, blue;

      if(value >= 0.0f) {
        if(value > 1.0f) value = 1.0f;
        red = 255;
        green = blue = static_cast<unsigned char>(255.0f * (1.0f - value));
      }
      else {
        if(value < -1.0f) value = -1.0f;
        blue = 255;
        red = green = static_cast<unsigned char>(255.0f * (1.0f + value));
      }

      img[idx_img + 0] = red;
      img[idx_img + 1] = green;
      img[idx_img + 2] = blue;
    }
  }

  if(stbi_write_png(filename, Nx, Ny, 3, img, Nx * 3) == 0) {
    std::perror("Failed to write image");
  }

  std::free(img);
}

static void write_binary_field(const std::string& filename, const std::vector<float>& data) {
  std::ofstream ofs(filename, std::ios::binary);
  if(!ofs) {
    std::cerr << "Failed to open " << filename << " for writing.\n";
    std::exit(EXIT_FAILURE);
  }

  ofs.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));

  if(!ofs) {
    std::cerr << "Failed to write " << filename << ".\n";
    std::exit(EXIT_FAILURE);
  }
}

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

class gDiamondSeq {
public:
  gDiamondSeq(size_t Nx, size_t Ny, size_t Nz)
    : _Nx(Nx), _Ny(Ny), _Nz(Nz),
      _Ex_seq(Nx * Ny * Nz, 0.0f), _Ey_seq(Nx * Ny * Nz, 0.0f), _Ez_seq(Nx * Ny * Nz, 0.0f),
      _Hx_seq(Nx * Ny * Nz, 0.0f), _Hy_seq(Nx * Ny * Nz, 0.0f), _Hz_seq(Nx * Ny * Nz, 0.0f),
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

    std::cerr << "initializing Ca, Cb, Da, Db...\n";

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
    std::cerr << "finish initialization\n";
  }

  void update_FDTD_seq_figures(size_t num_timesteps) {
    const std::string outdir = "seq_figures_cpp";

    std::filesystem::remove_all(outdir);
    if(!std::filesystem::create_directory(outdir)) {
      std::cerr << "failed to create " + outdir + "\n";
      std::exit(EXIT_FAILURE);
    }
    std::cerr << outdir + " created successfully.\n";

    std::vector<float> Ex_temp(_Nx * _Ny * _Nz, 0.0f);
    std::vector<float> Ey_temp(_Nx * _Ny * _Nz, 0.0f);
    std::vector<float> Ez_temp(_Nx * _Ny * _Nz, 0.0f);
    std::vector<float> Hx_temp(_Nx * _Ny * _Nz, 0.0f);
    std::vector<float> Hy_temp(_Nx * _Ny * _Nz, 0.0f);
    std::vector<float> Hz_temp(_Nx * _Ny * _Nz, 0.0f);

    std::chrono::duration<double> seq_runtime(0.0);
    size_t record_stride = std::max<size_t>(1, num_timesteps / 10);

    for(size_t t = 0; t < num_timesteps; ++t) {
      auto start = std::chrono::high_resolution_clock::now();

      std::fill(_Mz.begin(), _Mz.end(), 0.0f);
      float Mz_value = M_source_amp * std::sin(SOURCE_OMEGA * static_cast<float>(t) * dt);
      _Mz[_source_idx] = Mz_value;

      // update E
      for(size_t k = 1; k < _Nz - 1; ++k) {
        for(size_t j = 1; j < _Ny - 1; ++j) {
          for(size_t i = 1; i < _Nx - 1; ++i) {
            size_t idx = i + j * _Nx + k * (_Nx * _Ny);

            Ex_temp[idx] = _Cax[idx] * Ex_temp[idx] + _Cbx[idx] *
              ((Hz_temp[idx] - Hz_temp[idx - _Nx]) -
               (Hy_temp[idx] - Hy_temp[idx - _Nx * _Ny]) -
               _Jx[idx] * _dx);

            Ey_temp[idx] = _Cay[idx] * Ey_temp[idx] + _Cby[idx] *
              ((Hx_temp[idx] - Hx_temp[idx - _Nx * _Ny]) -
               (Hz_temp[idx] - Hz_temp[idx - 1]) -
               _Jy[idx] * _dx);

            Ez_temp[idx] = _Caz[idx] * Ez_temp[idx] + _Cbz[idx] *
              ((Hy_temp[idx] - Hy_temp[idx - 1]) -
               (Hx_temp[idx] - Hx_temp[idx - _Nx]) -
               _Jz[idx] * _dx);
          }
        }
      }

      // update H
      for(size_t k = 1; k < _Nz - 1; ++k) {
        for(size_t j = 1; j < _Ny - 1; ++j) {
          for(size_t i = 1; i < _Nx - 1; ++i) {
            size_t idx = i + j * _Nx + k * (_Nx * _Ny);

            Hx_temp[idx] = _Dax[idx] * Hx_temp[idx] + _Dbx[idx] *
              ((Ey_temp[idx + _Nx * _Ny] - Ey_temp[idx]) -
               (Ez_temp[idx + _Nx] - Ez_temp[idx]) -
               _Mx[idx] * _dx);

            Hy_temp[idx] = _Day[idx] * Hy_temp[idx] + _Dby[idx] *
              ((Ez_temp[idx + 1] - Ez_temp[idx]) -
               (Ex_temp[idx + _Nx * _Ny] - Ex_temp[idx]) -
               _My[idx] * _dx);

            Hz_temp[idx] = _Daz[idx] * Hz_temp[idx] + _Dbz[idx] *
              ((Ex_temp[idx + _Nx] - Ex_temp[idx]) -
               (Ey_temp[idx + 1] - Ey_temp[idx]) -
               _Mz[idx] * _dx);
          }
        }
      }

      auto end = std::chrono::high_resolution_clock::now();
      seq_runtime += end - start;

      if(t % record_stride == 0) {
        std::printf("Iter: %zu / %zu\n", t, num_timesteps);

        float* H_time_monitor_xy = (float*)std::malloc(_Nx * _Ny * sizeof(float));
        std::memset(H_time_monitor_xy, 0, _Nx * _Ny * sizeof(float));

        char field_filename[256];
        size_t k_mid = _Nz / 2;

        for(size_t i = 0; i < _Nx; ++i) {
          for(size_t j = 0; j < _Ny; ++j) {
            H_time_monitor_xy[i + j * _Nx] =
              Hz_temp[i + j * _Nx + k_mid * _Nx * _Ny];
          }
        }

        std::snprintf(field_filename, sizeof(field_filename),
                      "%s/Hz_seq_%04zu.png", outdir.c_str(), t);
        save_field_png(H_time_monitor_xy, field_filename,
                       static_cast<int>(_Nx), static_cast<int>(_Ny),
                       1.0f / std::sqrt(mu0 / eps0));

        std::free(H_time_monitor_xy);
      }
    }

    std::cout << "seq runtime (excluding figures output): "
              << seq_runtime.count() << "s\n";
    std::cout << "seq performance (excluding figures output): "
              << (_Nx * _Ny * _Nz / 1.0e6 * num_timesteps) / seq_runtime.count()
              << " Mcells/s\n";

    _Ex_seq = std::move(Ex_temp);
    _Ey_seq = std::move(Ey_temp);
    _Ez_seq = std::move(Ez_temp);
    _Hx_seq = std::move(Hx_temp);
    _Hy_seq = std::move(Hy_temp);
    _Hz_seq = std::move(Hz_temp);
  }

  void dump_fields_binary(const std::string& outdir = "seq_fields_cpp") const {
    std::filesystem::remove_all(outdir);
    if(!std::filesystem::create_directory(outdir)) {
      std::cerr << "failed to create " << outdir << ".\n";
      std::exit(EXIT_FAILURE);
    }

    write_binary_field(outdir + "/Ex_seq.bin", _Ex_seq);
    write_binary_field(outdir + "/Ey_seq.bin", _Ey_seq);
    write_binary_field(outdir + "/Ez_seq.bin", _Ez_seq);
    write_binary_field(outdir + "/Hx_seq.bin", _Hx_seq);
    write_binary_field(outdir + "/Hy_seq.bin", _Hy_seq);
    write_binary_field(outdir + "/Hz_seq.bin", _Hz_seq);

    std::cout << "saved final fields to " << outdir << "/\n";
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

  std::vector<float> _Ex_seq, _Ey_seq, _Ez_seq;
  std::vector<float> _Hx_seq, _Hy_seq, _Hz_seq;

  std::vector<float> _Jx, _Jy, _Jz;
  std::vector<float> _Mx, _My, _Mz;

  std::vector<float> _Cax, _Cay, _Caz;
  std::vector<float> _Cbx, _Cby, _Cbz;
  std::vector<float> _Dax, _Day, _Daz;
  std::vector<float> _Dbx, _Dby, _Dbz;
};

int main(int argc, char** argv) {
  size_t Nx = 64;
  size_t Ny = 64;
  size_t Nz = 64;
  size_t num_timesteps = 20;

  if(argc == 5) {
    Nx = static_cast<size_t>(std::stoul(argv[1]));
    Ny = static_cast<size_t>(std::stoul(argv[2]));
    Nz = static_cast<size_t>(std::stoul(argv[3]));
    num_timesteps = static_cast<size_t>(std::stoul(argv[4]));
  }
  else {
    std::cout << "Usage: " << argv[0] << " Nx Ny Nz num_timesteps\n";
    std::cout << "Using default: 64 64 64 20\n";
  }

  gDiamondSeq sim(Nx, Ny, Nz);
  sim.update_FDTD_seq_figures(num_timesteps);
  sim.dump_fields_binary("seq_fields_cpp");

  return 0;
}
