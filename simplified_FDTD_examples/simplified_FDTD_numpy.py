import os
import sys
import math
import shutil
import time
import numpy as np
from PIL import Image


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

        # constants: exactly follow C++
        self.PI = np.float32(3.14159265359)
        self.eps0 = np.float32(1.0)
        self.mu0 = np.float32(1.0)
        self.eta0 = np.float32(np.sqrt(self.mu0 / self.eps0))
        self.c0 = np.float32(1.0 / np.sqrt(self.mu0 * self.eps0))
        self.hbar = np.float32(1.0)

        # FDTD parameters
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

        # output field arrays
        self._Ex_seq = np.zeros(self._N, dtype=np.float32)
        self._Ey_seq = np.zeros(self._N, dtype=np.float32)
        self._Ez_seq = np.zeros(self._N, dtype=np.float32)
        self._Hx_seq = np.zeros(self._N, dtype=np.float32)
        self._Hy_seq = np.zeros(self._N, dtype=np.float32)
        self._Hz_seq = np.zeros(self._N, dtype=np.float32)

        # source/current arrays
        self._Jx = np.zeros(self._N, dtype=np.float32)
        self._Jy = np.zeros(self._N, dtype=np.float32)
        self._Jz = np.zeros(self._N, dtype=np.float32)
        self._Mx = np.zeros(self._N, dtype=np.float32)
        self._My = np.zeros(self._N, dtype=np.float32)
        self._Mz = np.zeros(self._N, dtype=np.float32)

        # coefficient arrays
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
        a_max = 2.0
        p = 3
        sigma_max = -(p + 1) * np.log(1e-5) / (2.0 * float(self.eta0) * t_PML * float(dx))

        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    eps_r = eps_air
                    mu_r = ComplexVal(1.0, 0.0)

                    idx = i + j * Nx + k * Nx * Ny
                    idx_2D = i + j * Nx

                    if k_min <= k <= k_max and mask[idx_2D]:
                        eps_r = eps_structure

                    bound_x_dist = 0.0
                    if i < t_PML:
                        bound_x_dist = 1.0 - float(i) / t_PML
                    if i + t_PML >= Nx:
                        bound_x_dist = 1.0 - float(Nx - i - 1) / t_PML

                    bound_y_dist = 0.0
                    if j < t_PML:
                        bound_y_dist = 1.0 - float(j) / t_PML
                    if j + t_PML >= Ny:
                        bound_y_dist = 1.0 - float(Ny - j - 1) / t_PML

                    bound_z_dist = 0.0
                    if k < t_PML:
                        bound_z_dist = 1.0 - float(k) / t_PML
                    if k + t_PML >= Nz:
                        bound_z_dist = 1.0 - float(Nz - k - 1) / t_PML

                    sx = ComplexVal(
                        1.0 + a_max * (bound_x_dist ** p),
                        sigma_max * (bound_x_dist ** p) / (float(OMEGA0) * float(self.eps0))
                    )
                    sy = ComplexVal(
                        1.0 + a_max * (bound_y_dist ** p),
                        sigma_max * (bound_y_dist ** p) / (float(OMEGA0) * float(self.eps0))
                    )
                    sz = ComplexVal(
                        1.0 + a_max * (bound_z_dist ** p),
                        sigma_max * (bound_z_dist ** p) / (float(OMEGA0) * float(self.eps0))
                    )

                    eps_xx_complex = (eps_r * sy * sz) / sx
                    eps_yy_complex = (eps_r * sx * sz) / sy
                    eps_zz_complex = (eps_r * sx * sy) / sz

                    mu_xx_complex = (mu_r * sy * sz) / sx
                    mu_yy_complex = (mu_r * sx * sz) / sy
                    mu_zz_complex = (mu_r * sx * sy) / sz

                    eps_xx = max(eps_xx_complex.real, 1.0)
                    eps_yy = max(eps_yy_complex.real, 1.0)
                    eps_zz = max(eps_zz_complex.real, 1.0)

                    mu_xx = max(mu_xx_complex.real, 1.0)
                    mu_yy = max(mu_yy_complex.real, 1.0)
                    mu_zz = max(mu_zz_complex.real, 1.0)

                    sigma_e_xx = abs(float(OMEGA0) * float(self.eps0) * eps_xx_complex.imag)
                    sigma_e_yy = abs(float(OMEGA0) * float(self.eps0) * eps_yy_complex.imag)
                    sigma_e_zz = abs(float(OMEGA0) * float(self.eps0) * eps_zz_complex.imag)

                    sigma_h_xx = abs(float(OMEGA0) * float(self.mu0) * mu_xx_complex.imag)
                    sigma_h_yy = abs(float(OMEGA0) * float(self.mu0) * mu_yy_complex.imag)
                    sigma_h_zz = abs(float(OMEGA0) * float(self.mu0) * mu_zz_complex.imag)

                    tmp_x = sigma_e_xx * float(dt) / (2.0 * eps_xx * float(self.eps0))
                    Cax[idx] = np.float32((1.0 - tmp_x) / (1.0 + tmp_x))
                    Cbx[idx] = np.float32((float(dt) / (eps_xx * float(self.eps0))) / (1.0 + tmp_x) / float(dx))

                    tmp_y = sigma_e_yy * float(dt) / (2.0 * eps_yy * float(self.eps0))
                    Cay[idx] = np.float32((1.0 - tmp_y) / (1.0 + tmp_y))
                    Cby[idx] = np.float32((float(dt) / (eps_yy * float(self.eps0))) / (1.0 + tmp_y) / float(dx))

                    tmp_z = sigma_e_zz * float(dt) / (2.0 * eps_zz * float(self.eps0))
                    Caz[idx] = np.float32((1.0 - tmp_z) / (1.0 + tmp_z))
                    Cbz[idx] = np.float32((float(dt) / (eps_zz * float(self.eps0))) / (1.0 + tmp_z) / float(dx))

                    tmp_x = sigma_h_xx * float(dt) / (2.0 * mu_xx * float(self.mu0))
                    Dax[idx] = np.float32((1.0 - tmp_x) / (1.0 + tmp_x))
                    Dbx[idx] = np.float32((float(dt) / (mu_xx * float(self.mu0))) / (1.0 + tmp_x) / float(dx))

                    tmp_y = sigma_h_yy * float(dt) / (2.0 * mu_yy * float(self.mu0))
                    Day[idx] = np.float32((1.0 - tmp_y) / (1.0 + tmp_y))
                    Dby[idx] = np.float32((float(dt) / (mu_yy * float(self.mu0))) / (1.0 + tmp_y) / float(dx))

                    tmp_z = sigma_h_zz * float(dt) / (2.0 * mu_zz * float(self.mu0))
                    Daz[idx] = np.float32((1.0 - tmp_z) / (1.0 + tmp_z))
                    Dbz[idx] = np.float32((float(dt) / (mu_zz * float(self.mu0))) / (1.0 + tmp_z) / float(dx))

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

    def update_FDTD_seq_figures(self, num_timesteps: int, outdir: str = "numpy_figures"):
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

            self._Mz.fill(0.0)
            Mz_value = np.float32(self.M_source_amp * np.sin(self.SOURCE_OMEGA * np.float32(t) * self.dt))
            self._Mz[self._source_idx] = Mz_value

            # update E
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

            # update H
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

        print(f"seq runtime (excluding figures output): {seq_runtime}s")
        print(
            "seq performance (excluding figures output): "
            f"{(Nx * Ny * Nz / 1.0e6 * num_timesteps) / seq_runtime} Mcells/s"
        )

        self._Ex_seq[:] = Ex_temp
        self._Ey_seq[:] = Ey_temp
        self._Ez_seq[:] = Ez_temp
        self._Hx_seq[:] = Hx_temp
        self._Hy_seq[:] = Hy_temp
        self._Hz_seq[:] = Hz_temp

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

    def compare_with_cpp_bin(self, cpp_dir: str = "seq_fields_cpp"):
        Nx = self._Nx
        Ny = self._Ny
        Nz = self._Nz
        N = Nx * Ny * Nz

        fields_np = {
            "Ex": self._Ex_seq,
            "Ey": self._Ey_seq,
            "Ez": self._Ez_seq,
            "Hx": self._Hx_seq,
            "Hy": self._Hy_seq,
            "Hz": self._Hz_seq,
        }

        for name, arr_np in fields_np.items():
            path = os.path.join(cpp_dir, f"{name}_seq.bin")
            if not os.path.exists(path):
                print(f"[WARN] missing {path}")
                continue

            arr_cpp = np.fromfile(path, dtype=np.float32)
            if arr_cpp.size != N:
                print(f"[WARN] {path} size mismatch: got {arr_cpp.size}, expect {N}")
                continue

            abs_err = np.abs(arr_cpp - arr_np)
            max_abs_err = float(np.max(abs_err))

            cpp_norm = float(np.linalg.norm(arr_cpp))
            rel_l2 = float(np.linalg.norm(arr_cpp - arr_np) / cpp_norm) if cpp_norm != 0.0 else float(np.linalg.norm(arr_cpp - arr_np))

            print(f"{name} max abs err = {max_abs_err:.8e}")
            print(f"{name} rel l2      = {rel_l2:.8e}")


def parse_args():
    if len(sys.argv) == 5:
        Nx = int(sys.argv[1])
        Ny = int(sys.argv[2])
        Nz = int(sys.argv[3])
        num_timesteps = int(sys.argv[4])
    else:
        print(f"Usage: {sys.argv[0]} Nx Ny Nz num_timesteps")
        print("Using default: 64 64 64 20")
        Nx, Ny, Nz, num_timesteps = 64, 64, 64, 20
    return Nx, Ny, Nz, num_timesteps


def main():
    Nx, Ny, Nz, num_timesteps = parse_args()

    sim = GDiamondNumpy(Nx, Ny, Nz)
    sim.update_FDTD_seq_figures(num_timesteps, outdir="numpy_figures")
    sim.print_probe()
    print("max abs fields =", sim.max_abs_field())

    print("\nComparing against C++ binary in seq_fields_cpp/")
    sim.compare_with_cpp_bin("seq_fields_cpp")


if __name__ == "__main__":
    main()
