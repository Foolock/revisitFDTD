import os
import sys
import math
import shutil
import time
import numpy as np
from PIL import Image

try:
  import cupy as cp
except ImportError:
  cp = None


class ComplexVal:
  def __init__(self, real: float, imag: float):
    self.real = float(real)
    self.imag = float(imag)

  def __add__(self, other):
    if isinstance(other, ComplexVal):
      return ComplexVal(self.real + other.real, self.imag + other.imag)
    return ComplexVal(self.real + other, self.imag)

  def __sub__(self, other):
    if isinstance(other, ComplexVal):
      return ComplexVal(self.real - other.real, self.imag - other.imag)
    return ComplexVal(self.real - other, self.imag)

  def __mul__(self, other):
    if isinstance(other, ComplexVal):
      return ComplexVal(
        self.real * other.real - self.imag * other.imag,
        self.real * other.imag + self.imag * other.real,
      )
    return ComplexVal(self.real * other, self.imag * other)

  def __truediv__(self, other):
    if isinstance(other, ComplexVal):
      den = other.real * other.real + other.imag * other.imag
      return ComplexVal(
        (self.real * other.real + self.imag * other.imag) / den,
        (self.imag * other.real - self.real * other.imag) / den,
      )
    return ComplexVal(self.real / other, self.imag / other)


class GDiamondNumpy:
  def __init__(self, Nx: int, Ny: int, Nz: int):
    self._Nx = Nx
    self._Ny = Ny
    self._Nz = Nz
    self._N = Nx * Ny * Nz

    self.PI = np.float32(3.14159265359)
    self.eps0 = np.float32(1.0)
    self.mu0 = np.float32(1.0)
    self.eta0 = np.float32(np.sqrt(self.mu0 / self.eps0))
    self.c0 = np.float32(1.0 / np.sqrt(self.mu0 * self.eps0))
    self.hbar = np.float32(1.0)

    self.um = np.float32(1.0)
    self.nm = np.float32(self.um / 1.0e3)

    self.SOURCE_WAVELENGTH = np.float32(1.0 * self.um)
    self.SOURCE_FREQUENCY = np.float32(self.c0 / self.SOURCE_WAVELENGTH)
    self.SOURCE_OMEGA = np.float32(2.0 * self.PI * self.SOURCE_FREQUENCY)
    self._dx = np.float32(self.SOURCE_WAVELENGTH / 10.0)
    self.dt = np.float32(0.05)

    self.J_source_amp = np.float32(5e4)
    self.M_source_amp = np.float32(self.J_source_amp * (self.eta0 ** 3.0))

    self.freq_sigma = np.float32(0.05 * self.SOURCE_FREQUENCY)
    self.t_sigma = np.float32(1.0 / self.freq_sigma / (2.0 * self.PI))
    self.t_peak = np.float32(5.0 * self.t_sigma)

    self._source_idx = Nx // 2 + (Ny // 2) * Nx + (Nz // 2) * Nx * Ny

    self._Ex_seq = np.zeros(self._N, dtype=np.float32)
    self._Ey_seq = np.zeros(self._N, dtype=np.float32)
    self._Ez_seq = np.zeros(self._N, dtype=np.float32)
    self._Hx_seq = np.zeros(self._N, dtype=np.float32)
    self._Hy_seq = np.zeros(self._N, dtype=np.float32)
    self._Hz_seq = np.zeros(self._N, dtype=np.float32)

    self._Jx = np.zeros(self._N, dtype=np.float32)
    self._Jy = np.zeros(self._N, dtype=np.float32)
    self._Jz = np.zeros(self._N, dtype=np.float32)
    self._Mx = np.zeros(self._N, dtype=np.float32)
    self._My = np.zeros(self._N, dtype=np.float32)
    self._Mz = np.zeros(self._N, dtype=np.float32)

    self._Cax = np.zeros(self._N, dtype=np.float32)
    self._Cay = np.zeros(self._N, dtype=np.float32)
    self._Caz = np.zeros(self._N, dtype=np.float32)
    self._Cbx = np.zeros(self._N, dtype=np.float32)
    self._Cby = np.zeros(self._N, dtype=np.float32)
    self._Cbz = np.zeros(self._N, dtype=np.float32)

    self._Dax = np.zeros(self._N, dtype=np.float32)
    self._Day = np.zeros(self._N, dtype=np.float32)
    self._Daz = np.zeros(self._N, dtype=np.float32)
    self._Dbx = np.zeros(self._N, dtype=np.float32)
    self._Dby = np.zeros(self._N, dtype=np.float32)
    self._Dbz = np.zeros(self._N, dtype=np.float32)

    print("initializing Ca, Cb, Da, Db...")

    mask = np.zeros((Nx * Ny,), dtype=bool)

    eps_air = ComplexVal(1.0, 0.0)
    eps_Si = ComplexVal(12.0, 0.001)

    t_slab = np.float32(200.0 * self.nm)
    t_slab_grid = int(round(float(t_slab / self._dx)))
    k_mid = Nz // 2
    slab_k_min = k_mid - t_slab_grid // 2
    slab_k_max = slab_k_min + t_slab_grid

    h_PML = np.float32(1.0 * self.um)
    t_PML = int(np.ceil(float(h_PML / self._dx)))

    self.set_FDTD_matrices_3D_structure(
      self._Cax, self._Cbx, self._Cay, self._Cby, self._Caz, self._Cbz,
      self._Dax, self._Dbx, self._Day, self._Dby, self._Daz, self._Dbz,
      Nx, Ny, Nz, self._dx, self.dt, mask, eps_air, eps_Si,
      slab_k_min, slab_k_max, self.SOURCE_OMEGA, t_PML
    )

    print("finish initialization")

  def set_FDTD_matrices_3D_structure(
    self,
    Cax, Cbx, Cay, Cby, Caz, Cbz,
    Dax, Dbx, Day, Dby, Daz, Dbz,
    Nx, Ny, Nz, dx, dt, mask,
    eps_air, eps_structure,
    k_min, k_max, OMEGA0, t_PML
  ):
    a_max = np.float32(2.0)
    p = 3
    sigma_max = np.float32(
      -(p + 1) * np.log(1e-5) / (2.0 * float(self.eta0) * t_PML * float(dx))
    )

    # ------------------------------------------------------------------
    # Build 3D coordinates
    # ------------------------------------------------------------------
    x = np.arange(Nx, dtype=np.float32)
    y = np.arange(Ny, dtype=np.float32)
    z = np.arange(Nz, dtype=np.float32)

    ix = x[None, None, :]   # (1, 1, Nx)
    iy = y[None, :, None]   # (1, Ny, 1)
    iz = z[:, None, None]   # (Nz, 1, 1)

    # ------------------------------------------------------------------
    # Structure mask in 3D
    # mask is originally 2D flattened as (Ny, Nx)
    # ------------------------------------------------------------------
    mask_2d = mask.reshape(Ny, Nx)
    slab_xy = mask_2d[None, :, :]                         # (1, Ny, Nx)
    slab_z = (iz >= k_min) & (iz <= k_max)               # (Nz, 1, 1)
    structure_mask = slab_z & slab_xy                    # (Nz, Ny, Nx)

    # eps_r / mu_r
    eps_real = np.where(structure_mask, eps_structure.real, eps_air.real).astype(np.float32)
    eps_imag = np.where(structure_mask, eps_structure.imag, eps_air.imag).astype(np.float32)

    mu_real = np.ones((Nz, Ny, Nx), dtype=np.float32)
    mu_imag = np.zeros((Nz, Ny, Nx), dtype=np.float32)

    # ------------------------------------------------------------------
    # Distance to PML boundary in each direction
    # Keep the same semantics as your original code
    # ------------------------------------------------------------------
    bx = np.zeros((Nx,), dtype=np.float32)
    by = np.zeros((Ny,), dtype=np.float32)
    bz = np.zeros((Nz,), dtype=np.float32)

    left_x = x < t_PML
    right_x = (x + t_PML) >= Nx
    bx[left_x] = 1.0 - x[left_x] / np.float32(t_PML)
    bx[right_x] = 1.0 - (np.float32(Nx) - x[right_x] - 1.0) / np.float32(t_PML)

    left_y = y < t_PML
    right_y = (y + t_PML) >= Ny
    by[left_y] = 1.0 - y[left_y] / np.float32(t_PML)
    by[right_y] = 1.0 - (np.float32(Ny) - y[right_y] - 1.0) / np.float32(t_PML)

    left_z = z < t_PML
    right_z = (z + t_PML) >= Nz
    bz[left_z] = 1.0 - z[left_z] / np.float32(t_PML)
    bz[right_z] = 1.0 - (np.float32(Nz) - z[right_z] - 1.0) / np.float32(t_PML)

    bx = bx[None, None, :]   # (1, 1, Nx)
    by = by[None, :, None]   # (1, Ny, 1)
    bz = bz[:, None, None]   # (Nz, 1, 1)

    bx_p = bx ** p
    by_p = by ** p
    bz_p = bz ** p

    coeff_imag = np.float32(sigma_max / (float(OMEGA0) * float(self.eps0)))

    # s = real + j imag
    sx_r = 1.0 + a_max * bx_p
    sy_r = 1.0 + a_max * by_p
    sz_r = 1.0 + a_max * bz_p

    sx_i = coeff_imag * bx_p
    sy_i = coeff_imag * by_p
    sz_i = coeff_imag * bz_p

    # ------------------------------------------------------------------
    # Complex helper ops on arrays
    # ------------------------------------------------------------------
    def cmul(ar, ai, br, bi):
      return ar * br - ai * bi, ar * bi + ai * br

    def cdiv(ar, ai, br, bi):
      den = br * br + bi * bi
      return (ar * br + ai * bi) / den, (ai * br - ar * bi) / den

    # ------------------------------------------------------------------
    # eps_xx = (eps_r * sy * sz) / sx
    # eps_yy = (eps_r * sx * sz) / sy
    # eps_zz = (eps_r * sx * sy) / sz
    # ------------------------------------------------------------------
    tmp_r, tmp_i = cmul(sy_r, sy_i, sz_r, sz_i)
    num_r, num_i = cmul(eps_real, eps_imag, tmp_r, tmp_i)
    eps_xx_r, eps_xx_i = cdiv(num_r, num_i, sx_r, sx_i)

    tmp_r, tmp_i = cmul(sx_r, sx_i, sz_r, sz_i)
    num_r, num_i = cmul(eps_real, eps_imag, tmp_r, tmp_i)
    eps_yy_r, eps_yy_i = cdiv(num_r, num_i, sy_r, sy_i)

    tmp_r, tmp_i = cmul(sx_r, sx_i, sy_r, sy_i)
    num_r, num_i = cmul(eps_real, eps_imag, tmp_r, tmp_i)
    eps_zz_r, eps_zz_i = cdiv(num_r, num_i, sz_r, sz_i)

    # ------------------------------------------------------------------
    # mu_xx = (mu_r * sy * sz) / sx
    # mu_yy = (mu_r * sx * sz) / sy
    # mu_zz = (mu_r * sx * sy) / sz
    # here mu_r = 1 + j0
    # ------------------------------------------------------------------
    tmp_r, tmp_i = cmul(sy_r, sy_i, sz_r, sz_i)
    mu_xx_r, mu_xx_i = cdiv(tmp_r, tmp_i, sx_r, sx_i)

    tmp_r, tmp_i = cmul(sx_r, sx_i, sz_r, sz_i)
    mu_yy_r, mu_yy_i = cdiv(tmp_r, tmp_i, sy_r, sy_i)

    tmp_r, tmp_i = cmul(sx_r, sx_i, sy_r, sy_i)
    mu_zz_r, mu_zz_i = cdiv(tmp_r, tmp_i, sz_r, sz_i)

    # ------------------------------------------------------------------
    # Real parts clipped exactly like before
    # ------------------------------------------------------------------
    eps_xx = np.maximum(eps_xx_r, 1.0).astype(np.float32)
    eps_yy = np.maximum(eps_yy_r, 1.0).astype(np.float32)
    eps_zz = np.maximum(eps_zz_r, 1.0).astype(np.float32)

    mu_xx = np.maximum(mu_xx_r, 1.0).astype(np.float32)
    mu_yy = np.maximum(mu_yy_r, 1.0).astype(np.float32)
    mu_zz = np.maximum(mu_zz_r, 1.0).astype(np.float32)

    sigma_e_xx = np.abs(float(OMEGA0) * float(self.eps0) * eps_xx_i).astype(np.float32)
    sigma_e_yy = np.abs(float(OMEGA0) * float(self.eps0) * eps_yy_i).astype(np.float32)
    sigma_e_zz = np.abs(float(OMEGA0) * float(self.eps0) * eps_zz_i).astype(np.float32)

    sigma_h_xx = np.abs(float(OMEGA0) * float(self.mu0) * mu_xx_i).astype(np.float32)
    sigma_h_yy = np.abs(float(OMEGA0) * float(self.mu0) * mu_yy_i).astype(np.float32)
    sigma_h_zz = np.abs(float(OMEGA0) * float(self.mu0) * mu_zz_i).astype(np.float32)

    # ------------------------------------------------------------------
    # Fill Ca/Cb/Da/Db
    # ------------------------------------------------------------------
    tmp_x = sigma_e_xx * np.float32(dt) / (2.0 * eps_xx * np.float32(self.eps0))
    tmp_y = sigma_e_yy * np.float32(dt) / (2.0 * eps_yy * np.float32(self.eps0))
    tmp_z = sigma_e_zz * np.float32(dt) / (2.0 * eps_zz * np.float32(self.eps0))

    Cax[:] = ((1.0 - tmp_x) / (1.0 + tmp_x)).astype(np.float32).reshape(-1)
    Cbx[:] = ((np.float32(dt) / (eps_xx * np.float32(self.eps0))) / (1.0 + tmp_x) / np.float32(dx)).astype(np.float32).reshape(-1)

    Cay[:] = ((1.0 - tmp_y) / (1.0 + tmp_y)).astype(np.float32).reshape(-1)
    Cby[:] = ((np.float32(dt) / (eps_yy * np.float32(self.eps0))) / (1.0 + tmp_y) / np.float32(dx)).astype(np.float32).reshape(-1)

    Caz[:] = ((1.0 - tmp_z) / (1.0 + tmp_z)).astype(np.float32).reshape(-1)
    Cbz[:] = ((np.float32(dt) / (eps_zz * np.float32(self.eps0))) / (1.0 + tmp_z) / np.float32(dx)).astype(np.float32).reshape(-1)

    tmp_x = sigma_h_xx * np.float32(dt) / (2.0 * mu_xx * np.float32(self.mu0))
    tmp_y = sigma_h_yy * np.float32(dt) / (2.0 * mu_yy * np.float32(self.mu0))
    tmp_z = sigma_h_zz * np.float32(dt) / (2.0 * mu_zz * np.float32(self.mu0))

    Dax[:] = ((1.0 - tmp_x) / (1.0 + tmp_x)).astype(np.float32).reshape(-1)
    Dbx[:] = ((np.float32(dt) / (mu_xx * np.float32(self.mu0))) / (1.0 + tmp_x) / np.float32(dx)).astype(np.float32).reshape(-1)

    Day[:] = ((1.0 - tmp_y) / (1.0 + tmp_y)).astype(np.float32).reshape(-1)
    Dby[:] = ((np.float32(dt) / (mu_yy * np.float32(self.mu0))) / (1.0 + tmp_y) / np.float32(dx)).astype(np.float32).reshape(-1)

    Daz[:] = ((1.0 - tmp_z) / (1.0 + tmp_z)).astype(np.float32).reshape(-1)
    Dbz[:] = ((np.float32(dt) / (mu_zz * np.float32(self.mu0))) / (1.0 + tmp_z) / np.float32(dx)).astype(np.float32).reshape(-1)

  def save_field_png(self, u: np.ndarray, filename: str, Nx: int, Ny: int, vmax: float):
    img = np.empty((Ny, Nx, 3), dtype=np.uint8)

    for i in range(Nx):
      for j in range(Ny):
        idx_field = i + j * Nx
        value = float(u[idx_field]) / float(vmax)

        if value >= 0.0:
          if value > 1.0:
            value = 1.0
          red = 255
          green = int(255.0 * (1.0 - value))
          blue = int(255.0 * (1.0 - value))
        else:
          if value < -1.0:
            value = -1.0
          blue = 255
          red = int(255.0 * (1.0 + value))
          green = int(255.0 * (1.0 + value))

        img[j, i, 0] = red
        img[j, i, 1] = green
        img[j, i, 2] = blue

    Image.fromarray(img, mode="RGB").save(filename)

  def _finalize_from_host_arrays(self, Ex_temp, Ey_temp, Ez_temp, Hx_temp, Hy_temp, Hz_temp, seq_runtime, num_timesteps):
    self._Ex_seq[:] = Ex_temp.reshape(-1)
    self._Ey_seq[:] = Ey_temp.reshape(-1)
    self._Ez_seq[:] = Ez_temp.reshape(-1)
    self._Hx_seq[:] = Hx_temp.reshape(-1)
    self._Hy_seq[:] = Hy_temp.reshape(-1)
    self._Hz_seq[:] = Hz_temp.reshape(-1)

    print(f"seq runtime (excluding figures output): {seq_runtime}s")
    print(
      "seq performance (excluding figures output): "
      f"{(self._Nx * self._Ny * self._Nz / 1.0e6 * num_timesteps) / seq_runtime} Mcells/s"
    )

  def update_FDTD_seq_figures_indexing(self, num_timesteps: int, outdir: str):
    if os.path.exists(outdir):
      shutil.rmtree(outdir)
    os.mkdir(outdir)
    print(f"{outdir} created successfully.")

    Nx = self._Nx
    Ny = self._Ny
    Nz = self._Nz
    Nxy = Nx * Ny

    Ex_temp = np.zeros(self._N, dtype=np.float32)
    Ey_temp = np.zeros(self._N, dtype=np.float32)
    Ez_temp = np.zeros(self._N, dtype=np.float32)
    Hx_temp = np.zeros(self._N, dtype=np.float32)
    Hy_temp = np.zeros(self._N, dtype=np.float32)
    Hz_temp = np.zeros(self._N, dtype=np.float32)

    seq_runtime = 0.0
    record_stride = max(1, num_timesteps // 10)

    for t in range(num_timesteps):
      start = time.perf_counter()

      Mz_value = np.float32(self.M_source_amp * np.sin(self.SOURCE_OMEGA * np.float32(t) * self.dt))
      self._Mz[self._source_idx] = Mz_value

      for k in range(1, Nz - 1):
        for j in range(1, Ny - 1):
          for i in range(1, Nx - 1):
            idx = i + j * Nx + k * Nxy

            Ex_temp[idx] = (
              self._Cax[idx] * Ex_temp[idx]
              + self._Cbx[idx] * (
                (Hz_temp[idx] - Hz_temp[idx - Nx])
                - (Hy_temp[idx] - Hy_temp[idx - Nxy])
                - self._Jx[idx] * self._dx
              )
            )

            Ey_temp[idx] = (
              self._Cay[idx] * Ey_temp[idx]
              + self._Cby[idx] * (
                (Hx_temp[idx] - Hx_temp[idx - Nxy])
                - (Hz_temp[idx] - Hz_temp[idx - 1])
                - self._Jy[idx] * self._dx
              )
            )

            Ez_temp[idx] = (
              self._Caz[idx] * Ez_temp[idx]
              + self._Cbz[idx] * (
                (Hy_temp[idx] - Hy_temp[idx - 1])
                - (Hx_temp[idx] - Hx_temp[idx - Nx])
                - self._Jz[idx] * self._dx
              )
            )

      for k in range(1, Nz - 1):
        for j in range(1, Ny - 1):
          for i in range(1, Nx - 1):
            idx = i + j * Nx + k * Nxy

            Hx_temp[idx] = (
              self._Dax[idx] * Hx_temp[idx]
              + self._Dbx[idx] * (
                (Ey_temp[idx + Nxy] - Ey_temp[idx])
                - (Ez_temp[idx + Nx] - Ez_temp[idx])
                - self._Mx[idx] * self._dx
              )
            )

            Hy_temp[idx] = (
              self._Day[idx] * Hy_temp[idx]
              + self._Dby[idx] * (
                (Ez_temp[idx + 1] - Ez_temp[idx])
                - (Ex_temp[idx + Nxy] - Ex_temp[idx])
                - self._My[idx] * self._dx
              )
            )

            Hz_temp[idx] = (
              self._Daz[idx] * Hz_temp[idx]
              + self._Dbz[idx] * (
                (Ex_temp[idx + Nx] - Ex_temp[idx])
                - (Ey_temp[idx + 1] - Ey_temp[idx])
                - self._Mz[idx] * self._dx
              )
            )

      end = time.perf_counter()
      seq_runtime += (end - start)

      if t % record_stride == 0:
        print(f"Iter: {t} / {num_timesteps}")
        H_time_monitor_xy = np.zeros(Nx * Ny, dtype=np.float32)
        k_mid = Nz // 2

        for i in range(Nx):
          for j in range(Ny):
            src_idx = i + j * Nx + k_mid * Nx * Ny
            H_time_monitor_xy[i + j * Nx] = Hz_temp[src_idx]

        field_filename = os.path.join(outdir, f"Hz_seq_{t:04d}.png")
        self.save_field_png(
          H_time_monitor_xy,
          field_filename,
          Nx,
          Ny,
          1.0 / math.sqrt(float(self.mu0 / self.eps0))
        )

    self._finalize_from_host_arrays(
      Ex_temp, Ey_temp, Ez_temp, Hx_temp, Hy_temp, Hz_temp, seq_runtime, num_timesteps
    )

  def update_FDTD_seq_figures_vectorized(self, num_timesteps: int, outdir: str):
    if os.path.exists(outdir):
      shutil.rmtree(outdir)
    os.mkdir(outdir)
    print(f"{outdir} created successfully.")

    Nx = self._Nx
    Ny = self._Ny
    Nz = self._Nz

    Ex_temp = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    Ey_temp = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    Ez_temp = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    Hx_temp = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    Hy_temp = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    Hz_temp = np.zeros((Nz, Ny, Nx), dtype=np.float32)

    Jx = self._Jx.reshape(Nz, Ny, Nx)
    Jy = self._Jy.reshape(Nz, Ny, Nx)
    Jz = self._Jz.reshape(Nz, Ny, Nx)
    Mx = self._Mx.reshape(Nz, Ny, Nx)
    My = self._My.reshape(Nz, Ny, Nx)
    Mz = self._Mz.reshape(Nz, Ny, Nx)

    Cax = self._Cax.reshape(Nz, Ny, Nx)
    Cay = self._Cay.reshape(Nz, Ny, Nx)
    Caz = self._Caz.reshape(Nz, Ny, Nx)
    Cbx = self._Cbx.reshape(Nz, Ny, Nx)
    Cby = self._Cby.reshape(Nz, Ny, Nx)
    Cbz = self._Cbz.reshape(Nz, Ny, Nx)

    Dax = self._Dax.reshape(Nz, Ny, Nx)
    Day = self._Day.reshape(Nz, Ny, Nx)
    Daz = self._Daz.reshape(Nz, Ny, Nx)
    Dbx = self._Dbx.reshape(Nz, Ny, Nx)
    Dby = self._Dby.reshape(Nz, Ny, Nx)
    Dbz = self._Dbz.reshape(Nz, Ny, Nx)

    core = np.s_[1:Nz-1, 1:Ny-1, 1:Nx-1]

    seq_runtime = 0.0
    record_stride = max(1, num_timesteps // 10)

    for t in range(num_timesteps):
      start = time.perf_counter()

      mz_k = self._source_idx // (Nx * Ny)
      rem = self._source_idx % (Nx * Ny)
      mz_j = rem // Nx
      mz_i = rem % Nx
      Mz[mz_k, mz_j, mz_i] = np.float32(
        self.M_source_amp * np.sin(self.SOURCE_OMEGA * np.float32(t) * self.dt)
      )

      Ex_temp[core] = (
        Cax[core] * Ex_temp[core]
        + Cbx[core] * (
          (Hz_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hz_temp[1:Nz-1, 0:Ny-2, 1:Nx-1])
          - (Hy_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hy_temp[0:Nz-2, 1:Ny-1, 1:Nx-1])
          - Jx[core] * self._dx
        )
      )

      Ey_temp[core] = (
        Cay[core] * Ey_temp[core]
        + Cby[core] * (
          (Hx_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hx_temp[0:Nz-2, 1:Ny-1, 1:Nx-1])
          - (Hz_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hz_temp[1:Nz-1, 1:Ny-1, 0:Nx-2])
          - Jy[core] * self._dx
        )
      )

      Ez_temp[core] = (
        Caz[core] * Ez_temp[core]
        + Cbz[core] * (
          (Hy_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hy_temp[1:Nz-1, 1:Ny-1, 0:Nx-2])
          - (Hx_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hx_temp[1:Nz-1, 0:Ny-2, 1:Nx-1])
          - Jz[core] * self._dx
        )
      )

      Hx_temp[core] = (
        Dax[core] * Hx_temp[core]
        + Dbx[core] * (
          (Ey_temp[2:Nz, 1:Ny-1, 1:Nx-1] - Ey_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - (Ez_temp[1:Nz-1, 2:Ny, 1:Nx-1] - Ez_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - Mx[core] * self._dx
        )
      )

      Hy_temp[core] = (
        Day[core] * Hy_temp[core]
        + Dby[core] * (
          (Ez_temp[1:Nz-1, 1:Ny-1, 2:Nx] - Ez_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - (Ex_temp[2:Nz, 1:Ny-1, 1:Nx-1] - Ex_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - My[core] * self._dx
        )
      )

      Hz_temp[core] = (
        Daz[core] * Hz_temp[core]
        + Dbz[core] * (
          (Ex_temp[1:Nz-1, 2:Ny, 1:Nx-1] - Ex_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - (Ey_temp[1:Nz-1, 1:Ny-1, 2:Nx] - Ey_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - Mz[core] * self._dx
        )
      )

      end = time.perf_counter()
      seq_runtime += (end - start)

      if t % record_stride == 0:
        print(f"Iter: {t} / {num_timesteps}")
        k_mid = Nz // 2
        H_time_monitor_xy = Hz_temp[k_mid, :, :].reshape(-1).astype(np.float32)

        field_filename = os.path.join(outdir, f"Hz_seq_{t:04d}.png")
        self.save_field_png(
          H_time_monitor_xy,
          field_filename,
          Nx,
          Ny,
          1.0 / math.sqrt(float(self.mu0 / self.eps0))
        )

    self._finalize_from_host_arrays(
      Ex_temp, Ey_temp, Ez_temp, Hx_temp, Hy_temp, Hz_temp, seq_runtime, num_timesteps
    )

  def update_FDTD_seq_figures_cuda(self, num_timesteps: int, outdir: str):
    if cp is None:
      raise RuntimeError("CuPy is not installed. Install CuPy first.")

    if os.path.exists(outdir):
      shutil.rmtree(outdir)
    os.mkdir(outdir)
    print(f"{outdir} created successfully.")

    Nx = self._Nx
    Ny = self._Ny
    Nz = self._Nz

    Ex_temp = cp.zeros((Nz, Ny, Nx), dtype=cp.float32)
    Ey_temp = cp.zeros((Nz, Ny, Nx), dtype=cp.float32)
    Ez_temp = cp.zeros((Nz, Ny, Nx), dtype=cp.float32)
    Hx_temp = cp.zeros((Nz, Ny, Nx), dtype=cp.float32)
    Hy_temp = cp.zeros((Nz, Ny, Nx), dtype=cp.float32)
    Hz_temp = cp.zeros((Nz, Ny, Nx), dtype=cp.float32)

    Jx = cp.asarray(self._Jx.reshape(Nz, Ny, Nx))
    Jy = cp.asarray(self._Jy.reshape(Nz, Ny, Nx))
    Jz = cp.asarray(self._Jz.reshape(Nz, Ny, Nx))
    Mx = cp.asarray(self._Mx.reshape(Nz, Ny, Nx))
    My = cp.asarray(self._My.reshape(Nz, Ny, Nx))
    Mz = cp.zeros((Nz, Ny, Nx), dtype=cp.float32)

    Cax = cp.asarray(self._Cax.reshape(Nz, Ny, Nx))
    Cay = cp.asarray(self._Cay.reshape(Nz, Ny, Nx))
    Caz = cp.asarray(self._Caz.reshape(Nz, Ny, Nx))
    Cbx = cp.asarray(self._Cbx.reshape(Nz, Ny, Nx))
    Cby = cp.asarray(self._Cby.reshape(Nz, Ny, Nx))
    Cbz = cp.asarray(self._Cbz.reshape(Nz, Ny, Nx))

    Dax = cp.asarray(self._Dax.reshape(Nz, Ny, Nx))
    Day = cp.asarray(self._Day.reshape(Nz, Ny, Nx))
    Daz = cp.asarray(self._Daz.reshape(Nz, Ny, Nx))
    Dbx = cp.asarray(self._Dbx.reshape(Nz, Ny, Nx))
    Dby = cp.asarray(self._Dby.reshape(Nz, Ny, Nx))
    Dbz = cp.asarray(self._Dbz.reshape(Nz, Ny, Nx))

    core = np.s_[1:Nz-1, 1:Ny-1, 1:Nx-1]

    mz_k = self._source_idx // (Nx * Ny)
    rem = self._source_idx % (Nx * Ny)
    mz_j = rem // Nx
    mz_i = rem % Nx

    seq_runtime = 0.0
    record_stride = max(1, num_timesteps // 10)

    for t in range(num_timesteps):
      start = time.perf_counter()

      Mz[mz_k, mz_j, mz_i] = cp.float32(
        self.M_source_amp * np.sin(self.SOURCE_OMEGA * np.float32(t) * self.dt)
      )

      Ex_temp[core] = (
        Cax[core] * Ex_temp[core]
        + Cbx[core] * (
          (Hz_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hz_temp[1:Nz-1, 0:Ny-2, 1:Nx-1])
          - (Hy_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hy_temp[0:Nz-2, 1:Ny-1, 1:Nx-1])
          - Jx[core] * self._dx
        )
      )

      Ey_temp[core] = (
        Cay[core] * Ey_temp[core]
        + Cby[core] * (
          (Hx_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hx_temp[0:Nz-2, 1:Ny-1, 1:Nx-1])
          - (Hz_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hz_temp[1:Nz-1, 1:Ny-1, 0:Nx-2])
          - Jy[core] * self._dx
        )
      )

      Ez_temp[core] = (
        Caz[core] * Ez_temp[core]
        + Cbz[core] * (
          (Hy_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hy_temp[1:Nz-1, 1:Ny-1, 0:Nx-2])
          - (Hx_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hx_temp[1:Nz-1, 0:Ny-2, 1:Nx-1])
          - Jz[core] * self._dx
        )
      )

      Hx_temp[core] = (
        Dax[core] * Hx_temp[core]
        + Dbx[core] * (
          (Ey_temp[2:Nz, 1:Ny-1, 1:Nx-1] - Ey_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - (Ez_temp[1:Nz-1, 2:Ny, 1:Nx-1] - Ez_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - Mx[core] * self._dx
        )
      )

      Hy_temp[core] = (
        Day[core] * Hy_temp[core]
        + Dby[core] * (
          (Ez_temp[1:Nz-1, 1:Ny-1, 2:Nx] - Ez_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - (Ex_temp[2:Nz, 1:Ny-1, 1:Nx-1] - Ex_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - My[core] * self._dx
        )
      )

      Hz_temp[core] = (
        Daz[core] * Hz_temp[core]
        + Dbz[core] * (
          (Ex_temp[1:Nz-1, 2:Ny, 1:Nx-1] - Ex_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - (Ey_temp[1:Nz-1, 1:Ny-1, 2:Nx] - Ey_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - Mz[core] * self._dx
        )
      )

      cp.cuda.Stream.null.synchronize()
      end = time.perf_counter()
      seq_runtime += (end - start)

      if t % record_stride == 0:
        print(f"Iter: {t} / {num_timesteps}")
        k_mid = Nz // 2
        H_time_monitor_xy = cp.asnumpy(Hz_temp[k_mid, :, :].reshape(-1))

        field_filename = os.path.join(outdir, f"Hz_seq_{t:04d}.png")
        self.save_field_png(
          H_time_monitor_xy,
          field_filename,
          Nx,
          Ny,
          1.0 / math.sqrt(float(self.mu0 / self.eps0))
        )

    self._finalize_from_host_arrays(
      cp.asnumpy(Ex_temp),
      cp.asnumpy(Ey_temp),
      cp.asnumpy(Ez_temp),
      cp.asnumpy(Hx_temp),
      cp.asnumpy(Hy_temp),
      cp.asnumpy(Hz_temp),
      seq_runtime,
      num_timesteps
    )

  def print_probe(self):
    idx = self._source_idx
    print("source_idx =", idx)
    print("Ex_seq[source_idx] =", float(self._Ex_seq[idx]))
    print("Ey_seq[source_idx] =", float(self._Ey_seq[idx]))
    print("Ez_seq[source_idx] =", float(self._Ez_seq[idx]))
    print("Hx_seq[source_idx] =", float(self._Hx_seq[idx]))
    print("Hy_seq[source_idx] =", float(self._Hy_seq[idx]))
    print("Hz_seq[source_idx] =", float(self._Hz_seq[idx]))

  def max_abs_field(self):
    return {
      "Ex": float(np.max(np.abs(self._Ex_seq))),
      "Ey": float(np.max(np.abs(self._Ey_seq))),
      "Ez": float(np.max(np.abs(self._Ez_seq))),
      "Hx": float(np.max(np.abs(self._Hx_seq))),
      "Hy": float(np.max(np.abs(self._Hy_seq))),
      "Hz": float(np.max(np.abs(self._Hz_seq))),
    }

  def compare_with_binary_dir(self, bin_dir: str, suffix: str):
    N = self._Nx * self._Ny * self._Nz

    fields_np = {
      "Ex": self._Ex_seq,
      "Ey": self._Ey_seq,
      "Ez": self._Ez_seq,
      "Hx": self._Hx_seq,
      "Hy": self._Hy_seq,
      "Hz": self._Hz_seq,
    }

    found_any = False

    for name, arr_np in fields_np.items():
      path = os.path.join(bin_dir, f"{name}_{suffix}.bin")
      if not os.path.exists(path):
        print(f"[WARN] missing {path}")
        continue

      found_any = True
      arr_ref = np.fromfile(path, dtype=np.float32)
      if arr_ref.size != N:
        print(f"[WARN] {path} size mismatch: got {arr_ref.size}, expect {N}")
        continue

      abs_err = np.abs(arr_ref - arr_np)
      max_abs_err = float(np.max(abs_err))

      ref_norm = float(np.linalg.norm(arr_ref))
      rel_l2 = float(np.linalg.norm(arr_ref - arr_np) / ref_norm) if ref_norm != 0.0 else float(np.linalg.norm(arr_ref - arr_np))

      print(f"{name} max abs err = {max_abs_err:.8e}")
      print(f"{name} rel l2      = {rel_l2:.8e}")

    return found_any

  def compare_with_cpp_bin(self, cpp_dir: str = "seq_fields_cpp"):
    return self.compare_with_binary_dir(cpp_dir, "seq")

  def compare_with_gpu_bin(self, gpu_dir: str = "gpu_fields_cpp"):
    return self.compare_with_binary_dir(gpu_dir, "gpu")


def parse_args():
  if len(sys.argv) != 7:
    print(f"Usage: {sys.argv[0]} Nx Ny Nz num_timesteps mode fig_dir")
    print("mode must be one of: indexing, vectorized, cuda")
    sys.exit(1)

  Nx = int(sys.argv[1])
  Ny = int(sys.argv[2])
  Nz = int(sys.argv[3])
  num_timesteps = int(sys.argv[4])
  mode = sys.argv[5]
  fig_dir = sys.argv[6]

  if mode not in ("indexing", "vectorized", "cuda"):
    print("mode must be one of: indexing, vectorized, cuda")
    sys.exit(1)

  return Nx, Ny, Nz, num_timesteps, mode, fig_dir


def main():
  Nx, Ny, Nz, num_timesteps, mode, fig_dir = parse_args()

  sim = GDiamondNumpy(Nx, Ny, Nz)

  if mode == "indexing":
    sim.update_FDTD_seq_figures_indexing(num_timesteps, outdir=fig_dir)
  elif mode == "vectorized":
    sim.update_FDTD_seq_figures_vectorized(num_timesteps, outdir=fig_dir)
  else:
    sim.update_FDTD_seq_figures_cuda(num_timesteps, outdir=fig_dir)

  # sim.print_probe()
  # print("max abs fields =", sim.max_abs_field())

  print("\nComparing against C++ binary in seq_fields_cpp/")
  sim.compare_with_cpp_bin("seq_fields_cpp")

  print("\nComparing against GPU binary in gpu_fields_cpp/")
  sim.compare_with_gpu_bin("gpu_fields_cpp")


if __name__ == "__main__":
  main()
