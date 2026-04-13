
import os
import time
import argparse
import importlib.util
from dataclasses import dataclass

import numpy as np

try:
  import cupy as cp
except ImportError:
  cp = None

try:
  import cuda.tile as ct
except ImportError:
  ct = None


@ct.kernel if ct is not None else (lambda f: f)
def update_e_kernel_inplace(
  Ex, Ey, Ez,
  Hx, Hy, Hz,
  Cax, Cbx, Cay, Cby, Caz, Cbz,
  Jx, Jy, Jz,
  dx: float,
  Nx: int, Ny: int, Nz: int,
  tile_size: "ct.Constant[int]",
):
  bid = ct.bid(0)
  base = bid * tile_size
  idx = base + ct.arange(tile_size, dtype=ct.int32)

  Nxy = Nx * Ny
  k = idx // Nxy
  rem = idx - k * Nxy
  j = rem // Nx
  i = rem - j * Nx

  interior = (i >= 1) & (i < Nx - 1) & (j >= 1) & (j < Ny - 1) & (k >= 1) & (k < Nz - 1)

  ex_old = ct.gather(Ex, idx, mask=interior)
  ey_old = ct.gather(Ey, idx, mask=interior)
  ez_old = ct.gather(Ez, idx, mask=interior)

  hx = ct.gather(Hx, idx, mask=interior)
  hy = ct.gather(Hy, idx, mask=interior)
  hz = ct.gather(Hz, idx, mask=interior)

  hz_jm1 = ct.gather(Hz, idx - Nx, mask=interior)
  hy_km1 = ct.gather(Hy, idx - Nxy, mask=interior)
  hx_km1 = ct.gather(Hx, idx - Nxy, mask=interior)
  hz_im1 = ct.gather(Hz, idx - 1, mask=interior)
  hy_im1 = ct.gather(Hy, idx - 1, mask=interior)
  hx_jm1 = ct.gather(Hx, idx - Nx, mask=interior)

  jx = ct.gather(Jx, idx, mask=interior)
  jy = ct.gather(Jy, idx, mask=interior)
  jz = ct.gather(Jz, idx, mask=interior)

  cax = ct.gather(Cax, idx, mask=interior)
  cbx = ct.gather(Cbx, idx, mask=interior)
  cay = ct.gather(Cay, idx, mask=interior)
  cby = ct.gather(Cby, idx, mask=interior)
  caz = ct.gather(Caz, idx, mask=interior)
  cbz = ct.gather(Cbz, idx, mask=interior)

  ex_new = cax * ex_old + cbx * ((hz - hz_jm1) - (hy - hy_km1) - jx * dx)
  ey_new = cay * ey_old + cby * ((hx - hx_km1) - (hz - hz_im1) - jy * dx)
  ez_new = caz * ez_old + cbz * ((hy - hy_im1) - (hx - hx_jm1) - jz * dx)

  ct.scatter(Ex, idx, ex_new, mask=interior)
  ct.scatter(Ey, idx, ey_new, mask=interior)
  ct.scatter(Ez, idx, ez_new, mask=interior)


@ct.kernel if ct is not None else (lambda f: f)
def update_h_kernel_inplace(
  Ex, Ey, Ez,
  Hx, Hy, Hz,
  Dax, Dbx, Day, Dby, Daz, Dbz,
  Mx, My, Mz,
  dx: float,
  Nx: int, Ny: int, Nz: int,
  tile_size: "ct.Constant[int]",
):
  bid = ct.bid(0)
  base = bid * tile_size
  idx = base + ct.arange(tile_size, dtype=ct.int32)

  Nxy = Nx * Ny
  k = idx // Nxy
  rem = idx - k * Nxy
  j = rem // Nx
  i = rem - j * Nx

  interior = (i >= 1) & (i < Nx - 1) & (j >= 1) & (j < Ny - 1) & (k >= 1) & (k < Nz - 1)

  hx_old = ct.gather(Hx, idx, mask=interior)
  hy_old = ct.gather(Hy, idx, mask=interior)
  hz_old = ct.gather(Hz, idx, mask=interior)

  ex = ct.gather(Ex, idx, mask=interior)
  ey = ct.gather(Ey, idx, mask=interior)
  ez = ct.gather(Ez, idx, mask=interior)

  ey_kp1 = ct.gather(Ey, idx + Nxy, mask=interior)
  ez_jp1 = ct.gather(Ez, idx + Nx, mask=interior)
  ez_ip1 = ct.gather(Ez, idx + 1, mask=interior)
  ex_kp1 = ct.gather(Ex, idx + Nxy, mask=interior)
  ex_jp1 = ct.gather(Ex, idx + Nx, mask=interior)
  ey_ip1 = ct.gather(Ey, idx + 1, mask=interior)

  mx = ct.gather(Mx, idx, mask=interior)
  my = ct.gather(My, idx, mask=interior)
  mz = ct.gather(Mz, idx, mask=interior)

  dax = ct.gather(Dax, idx, mask=interior)
  dbx = ct.gather(Dbx, idx, mask=interior)
  day = ct.gather(Day, idx, mask=interior)
  dby = ct.gather(Dby, idx, mask=interior)
  daz = ct.gather(Daz, idx, mask=interior)
  dbz = ct.gather(Dbz, idx, mask=interior)

  hx_new = dax * hx_old + dbx * ((ey_kp1 - ey) - (ez_jp1 - ez) - mx * dx)
  hy_new = day * hy_old + dby * ((ez_ip1 - ez) - (ex_kp1 - ex) - my * dx)
  hz_new = daz * hz_old + dbz * ((ex_jp1 - ex) - (ey_ip1 - ey) - mz * dx)

  ct.scatter(Hx, idx, hx_new, mask=interior)
  ct.scatter(Hy, idx, hy_new, mask=interior)
  ct.scatter(Hz, idx, hz_new, mask=interior)


class ComplexVal:
  def __init__(self, real: float, imag: float):
    self.real = float(real)
    self.imag = float(imag)


@dataclass
class CompareStats:
  max_abs_err: float
  rel_l2: float


class GDiamondCuTileCUDAStyle:
  def __init__(self, Nx: int, Ny: int, Nz: int):
    if cp is None:
      raise RuntimeError("CuPy is required.")
    if ct is None:
      raise RuntimeError("cuTile Python is required.")

    self._Nx = Nx
    self._Ny = Ny
    self._Nz = Nz
    self._N = Nx * Ny * Nz

    self.PI = np.float32(3.14159265359)
    self.eps0 = np.float32(1.0)
    self.mu0 = np.float32(1.0)
    self.eta0 = np.float32(np.sqrt(self.mu0 / self.eps0))
    self.c0 = np.float32(1.0 / np.sqrt(self.mu0 * self.eps0))

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

    self._Ex_host = np.zeros(self._N, dtype=np.float32)
    self._Ey_host = np.zeros(self._N, dtype=np.float32)
    self._Ez_host = np.zeros(self._N, dtype=np.float32)
    self._Hx_host = np.zeros(self._N, dtype=np.float32)
    self._Hy_host = np.zeros(self._N, dtype=np.float32)
    self._Hz_host = np.zeros(self._N, dtype=np.float32)

    self._Jx = np.zeros(self._N, dtype=np.float32)
    self._Jy = np.zeros(self._N, dtype=np.float32)
    self._Jz = np.zeros(self._N, dtype=np.float32)
    self._Mx = np.zeros(self._N, dtype=np.float32)
    self._My = np.zeros(self._N, dtype=np.float32)

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

    x = np.arange(Nx, dtype=np.float32)
    y = np.arange(Ny, dtype=np.float32)
    z = np.arange(Nz, dtype=np.float32)

    ix = x[None, None, :]
    iy = y[None, :, None]
    iz = z[:, None, None]

    mask_2d = mask.reshape(Ny, Nx)
    slab_xy = mask_2d[None, :, :]
    slab_z = (iz >= k_min) & (iz <= k_max)
    structure_mask = slab_z & slab_xy

    eps_real = np.where(structure_mask, eps_structure.real, eps_air.real).astype(np.float32)
    eps_imag = np.where(structure_mask, eps_structure.imag, eps_air.imag).astype(np.float32)

    mu_real = np.ones((Nz, Ny, Nx), dtype=np.float32)
    mu_imag = np.zeros((Nz, Ny, Nx), dtype=np.float32)

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

    bx = bx[None, None, :]
    by = by[None, :, None]
    bz = bz[:, None, None]

    bx_p = bx ** p
    by_p = by ** p
    bz_p = bz ** p

    coeff_imag = np.float32(sigma_max / (float(OMEGA0) * float(self.eps0)))

    sx_r = 1.0 + a_max * bx_p
    sy_r = 1.0 + a_max * by_p
    sz_r = 1.0 + a_max * bz_p

    sx_i = coeff_imag * bx_p
    sy_i = coeff_imag * by_p
    sz_i = coeff_imag * bz_p

    def cmul(ar, ai, br, bi):
      return ar * br - ai * bi, ar * bi + ai * br

    def cdiv(ar, ai, br, bi):
      den = br * br + bi * bi
      return (ar * br + ai * bi) / den, (ai * br - ar * bi) / den

    tmp_r, tmp_i = cmul(sy_r, sy_i, sz_r, sz_i)
    num_r, num_i = cmul(eps_real, eps_imag, tmp_r, tmp_i)
    eps_xx_r, eps_xx_i = cdiv(num_r, num_i, sx_r, sx_i)

    tmp_r, tmp_i = cmul(sx_r, sx_i, sz_r, sz_i)
    num_r, num_i = cmul(eps_real, eps_imag, tmp_r, tmp_i)
    eps_yy_r, eps_yy_i = cdiv(num_r, num_i, sy_r, sy_i)

    tmp_r, tmp_i = cmul(sx_r, sx_i, sy_r, sy_i)
    num_r, num_i = cmul(eps_real, eps_imag, tmp_r, tmp_i)
    eps_zz_r, eps_zz_i = cdiv(num_r, num_i, sz_r, sz_i)

    tmp_r, tmp_i = cmul(sy_r, sy_i, sz_r, sz_i)
    mu_xx_r, mu_xx_i = cdiv(tmp_r, tmp_i, sx_r, sx_i)

    tmp_r, tmp_i = cmul(sx_r, sx_i, sz_r, sz_i)
    mu_yy_r, mu_yy_i = cdiv(tmp_r, tmp_i, sy_r, sy_i)

    tmp_r, tmp_i = cmul(sx_r, sx_i, sy_r, sy_i)
    mu_zz_r, mu_zz_i = cdiv(tmp_r, tmp_i, sz_r, sz_i)

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

  def run_cutile_cuda_style(self, num_timesteps: int, tile_size: int = 256):
    if tile_size <= 0 or (tile_size & (tile_size - 1)) != 0:
      raise ValueError("tile_size must be a positive power of two.")

    N = self._N
    Nx, Ny, Nz = self._Nx, self._Ny, self._Nz
    grid = (int(ct.cdiv(N, tile_size)), 1, 1)
    stream = cp.cuda.get_current_stream()

    start = time.perf_counter()

    Ex = cp.zeros(N, dtype=cp.float32)
    Ey = cp.zeros(N, dtype=cp.float32)
    Ez = cp.zeros(N, dtype=cp.float32)
    Hx = cp.zeros(N, dtype=cp.float32)
    Hy = cp.zeros(N, dtype=cp.float32)
    Hz = cp.zeros(N, dtype=cp.float32)

    Jx = cp.zeros(N, dtype=cp.float32)
    Jy = cp.zeros(N, dtype=cp.float32)
    Jz = cp.zeros(N, dtype=cp.float32)
    Mx = cp.zeros(N, dtype=cp.float32)
    My = cp.zeros(N, dtype=cp.float32)
    Mz = cp.zeros(N, dtype=cp.float32)

    Cax = cp.asarray(self._Cax)
    Cay = cp.asarray(self._Cay)
    Caz = cp.asarray(self._Caz)
    Cbx = cp.asarray(self._Cbx)
    Cby = cp.asarray(self._Cby)
    Cbz = cp.asarray(self._Cbz)

    Dax = cp.asarray(self._Dax)
    Day = cp.asarray(self._Day)
    Daz = cp.asarray(self._Daz)
    Dbx = cp.asarray(self._Dbx)
    Dby = cp.asarray(self._Dby)
    Dbz = cp.asarray(self._Dbz)

    mz_idx = int(self._source_idx)

    for t in range(num_timesteps):
      mz_value = np.float32(self.M_source_amp * np.sin(self.SOURCE_OMEGA * np.float32(t) * self.dt))
      Mz[mz_idx] = cp.float32(mz_value)

      ct.launch(
        stream,
        grid,
        update_e_kernel_inplace,
        (
          Ex, Ey, Ez,
          Hx, Hy, Hz,
          Cax, Cbx, Cay, Cby, Caz, Cbz,
          Jx, Jy, Jz,
          float(self._dx),
          Nx, Ny, Nz,
          tile_size,
        ),
      )

      ct.launch(
        stream,
        grid,
        update_h_kernel_inplace,
        (
          Ex, Ey, Ez,
          Hx, Hy, Hz,
          Dax, Dbx, Day, Dby, Daz, Dbz,
          Mx, My, Mz,
          float(self._dx),
          Nx, Ny, Nz,
          tile_size,
        ),
      )

    self._Ex_host[:] = cp.asnumpy(Ex)
    self._Ey_host[:] = cp.asnumpy(Ey)
    self._Ez_host[:] = cp.asnumpy(Ez)
    self._Hx_host[:] = cp.asnumpy(Hx)
    self._Hy_host[:] = cp.asnumpy(Hy)
    self._Hz_host[:] = cp.asnumpy(Hz)

    stream.synchronize()
    runtime = time.perf_counter() - start

    print(f"cuTile runtime (including device alloc/H2D/kernels/D2H): {runtime}s")
    print(
      "cuTile performance (including device alloc/H2D/kernels/D2H): "
      f"{(self._Nx * self._Ny * self._Nz / 1.0e6 * num_timesteps) / runtime} Mcells/s"
    )

  def compare_with_numpy_vectorized(self, ref_module_path: str, num_timesteps: int):
    if not os.path.exists(ref_module_path):
      raise FileNotFoundError(f"Cannot find NumPy reference script: {ref_module_path}")

    spec = importlib.util.spec_from_file_location("fdtd_numpy_ref", ref_module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    ref_sim = module.GDiamondNumpy(self._Nx, self._Ny, self._Nz)

    Nx, Ny, Nz = self._Nx, self._Ny, self._Nz
    Ex_temp = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    Ey_temp = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    Ez_temp = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    Hx_temp = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    Hy_temp = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    Hz_temp = np.zeros((Nz, Ny, Nx), dtype=np.float32)

    Jx = ref_sim._Jx.reshape(Nz, Ny, Nx)
    Jy = ref_sim._Jy.reshape(Nz, Ny, Nx)
    Jz = ref_sim._Jz.reshape(Nz, Ny, Nx)
    Mx = ref_sim._Mx.reshape(Nz, Ny, Nx)
    My = ref_sim._My.reshape(Nz, Ny, Nx)
    Mz = np.zeros((Nz, Ny, Nx), dtype=np.float32)

    Cax = ref_sim._Cax.reshape(Nz, Ny, Nx)
    Cay = ref_sim._Cay.reshape(Nz, Ny, Nx)
    Caz = ref_sim._Caz.reshape(Nz, Ny, Nx)
    Cbx = ref_sim._Cbx.reshape(Nz, Ny, Nx)
    Cby = ref_sim._Cby.reshape(Nz, Ny, Nx)
    Cbz = ref_sim._Cbz.reshape(Nz, Ny, Nx)

    Dax = ref_sim._Dax.reshape(Nz, Ny, Nx)
    Day = ref_sim._Day.reshape(Nz, Ny, Nx)
    Daz = ref_sim._Daz.reshape(Nz, Ny, Nx)
    Dbx = ref_sim._Dbx.reshape(Nz, Ny, Nx)
    Dby = ref_sim._Dby.reshape(Nz, Ny, Nx)
    Dbz = ref_sim._Dbz.reshape(Nz, Ny, Nx)

    core = np.s_[1:Nz-1, 1:Ny-1, 1:Nx-1]
    mz_k = ref_sim._source_idx // (Nx * Ny)
    rem = ref_sim._source_idx % (Nx * Ny)
    mz_j = rem // Nx
    mz_i = rem % Nx

    start = time.perf_counter()
    for t in range(num_timesteps):
      Mz[mz_k, mz_j, mz_i] = np.float32(
        ref_sim.M_source_amp * np.sin(ref_sim.SOURCE_OMEGA * np.float32(t) * ref_sim.dt)
      )

      Ex_temp[core] = (
        Cax[core] * Ex_temp[core]
        + Cbx[core] * (
          (Hz_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hz_temp[1:Nz-1, 0:Ny-2, 1:Nx-1])
          - (Hy_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hy_temp[0:Nz-2, 1:Ny-1, 1:Nx-1])
          - Jx[core] * ref_sim._dx
        )
      )

      Ey_temp[core] = (
        Cay[core] * Ey_temp[core]
        + Cby[core] * (
          (Hx_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hx_temp[0:Nz-2, 1:Ny-1, 1:Nx-1])
          - (Hz_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hz_temp[1:Nz-1, 1:Ny-1, 0:Nx-2])
          - Jy[core] * ref_sim._dx
        )
      )

      Ez_temp[core] = (
        Caz[core] * Ez_temp[core]
        + Cbz[core] * (
          (Hy_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hy_temp[1:Nz-1, 1:Ny-1, 0:Nx-2])
          - (Hx_temp[1:Nz-1, 1:Ny-1, 1:Nx-1] - Hx_temp[1:Nz-1, 0:Ny-2, 1:Nx-1])
          - Jz[core] * ref_sim._dx
        )
      )

      Hx_temp[core] = (
        Dax[core] * Hx_temp[core]
        + Dbx[core] * (
          (Ey_temp[2:Nz, 1:Ny-1, 1:Nx-1] - Ey_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - (Ez_temp[1:Nz-1, 2:Ny, 1:Nx-1] - Ez_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - Mx[core] * ref_sim._dx
        )
      )

      Hy_temp[core] = (
        Day[core] * Hy_temp[core]
        + Dby[core] * (
          (Ez_temp[1:Nz-1, 1:Ny-1, 2:Nx] - Ez_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - (Ex_temp[2:Nz, 1:Ny-1, 1:Nx-1] - Ex_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - My[core] * ref_sim._dx
        )
      )

      Hz_temp[core] = (
        Daz[core] * Hz_temp[core]
        + Dbz[core] * (
          (Ex_temp[1:Nz-1, 2:Ny, 1:Nx-1] - Ex_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - (Ey_temp[1:Nz-1, 1:Ny-1, 2:Nx] - Ey_temp[1:Nz-1, 1:Ny-1, 1:Nx-1])
          - Mz[core] * ref_sim._dx
        )
      )
    numpy_runtime = time.perf_counter() - start

    print(f"NumPy vectorized runtime: {numpy_runtime}s")
    print(
      "NumPy vectorized performance: "
      f"{(self._Nx * self._Ny * self._Nz / 1.0e6 * num_timesteps) / numpy_runtime} Mcells/s"
    )

    ref_fields = {
      "Ex": Ex_temp.reshape(-1),
      "Ey": Ey_temp.reshape(-1),
      "Ez": Ez_temp.reshape(-1),
      "Hx": Hx_temp.reshape(-1),
      "Hy": Hy_temp.reshape(-1),
      "Hz": Hz_temp.reshape(-1),
    }
    cur_fields = {
      "Ex": self._Ex_host,
      "Ey": self._Ey_host,
      "Ez": self._Ez_host,
      "Hx": self._Hx_host,
      "Hy": self._Hy_host,
      "Hz": self._Hz_host,
    }

    print("\nComparing cuTile result against NumPy vectorized reference")
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
      ref = ref_fields[name]
      cur = cur_fields[name]
      abs_err = np.abs(ref - cur)
      max_abs_err = float(np.max(abs_err))
      ref_norm = float(np.linalg.norm(ref))
      rel_l2 = float(np.linalg.norm(ref - cur) / ref_norm) if ref_norm != 0.0 else float(np.linalg.norm(ref - cur))
      print(f"{name} max abs err = {max_abs_err:.8e}")
      print(f"{name} rel l2      = {rel_l2:.8e}")


def parse_args():
  parser = argparse.ArgumentParser(description="cuTile version updated to follow the uploaded CUDA C++ semantics.")
  parser.add_argument("Nx", type=int)
  parser.add_argument("Ny", type=int)
  parser.add_argument("Nz", type=int)
  parser.add_argument("num_timesteps", type=int)
  parser.add_argument("--tile-size", type=int, default=256)
  parser.add_argument("--numpy-ref", type=str, default=None)
  return parser.parse_args()


def main():
  args = parse_args()
  sim = GDiamondCuTileCUDAStyle(args.Nx, args.Ny, args.Nz)
  sim.run_cutile_cuda_style(args.num_timesteps, tile_size=args.tile_size)
  if args.numpy_ref:
    sim.compare_with_numpy_vectorized(args.numpy_ref, args.num_timesteps)


if __name__ == "__main__":
  main()
