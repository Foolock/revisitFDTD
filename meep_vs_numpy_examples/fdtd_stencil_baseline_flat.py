import os
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Config:
    # Physical domain size
    cell_x: float = 4.0
    cell_y: float = 4.0
    cell_z: float = 4.0

    # Grid resolution: points per unit length
    resolution: int = 8

    # Time stepping
    dt: float = 0.05
    num_steps: int = 20
    save_interval: int = 2

    # Initial condition on Ez only
    init_sigma: float = 0.25
    init_amplitude: float = 1.0

    # Output
    outdir: str = "baseline_flat_output"
    dtype: type = np.float64


class FDTDStencilBaselineFlat:
    """
    CUDA-friendly 3D E/H stencil baseline.

    Main design choices:
      - flattened 1D arrays
      - explicit (i, j, k) -> idx mapping
      - six logically separate update functions
      - no source during stepping
      - no boundary conditions / PML
      - boundary cells are simply not updated

    This is a simplified stencil system inspired by FDTD updates.
    It is intended as a deterministic implementation baseline,
    not as a full physical FDTD simulator.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.nx = int(cfg.cell_x * cfg.resolution)
        self.ny = int(cfg.cell_y * cfg.resolution)
        self.nz = int(cfg.cell_z * cfg.resolution)

        self.dx = 1.0 / cfg.resolution
        self.dy = 1.0 / cfg.resolution
        self.dz = 1.0 / cfg.resolution

        self.num_cells = self.nx * self.ny * self.nz
        self.shape = (self.nx, self.ny, self.nz)

        dtype = cfg.dtype

        # Flattened arrays
        self.Ex = np.zeros(self.num_cells, dtype=dtype)
        self.Ey = np.zeros(self.num_cells, dtype=dtype)
        self.Ez = np.zeros(self.num_cells, dtype=dtype)

        self.Hx = np.zeros(self.num_cells, dtype=dtype)
        self.Hy = np.zeros(self.num_cells, dtype=dtype)
        self.Hz = np.zeros(self.num_cells, dtype=dtype)

        self._initialize_fields()

    # -------------------------------------------------------------------------
    # Index helpers
    # -------------------------------------------------------------------------
    def idx(self, i: int, j: int, k: int) -> int:
        """
        Flattened row-major indexing:
          idx = i * (ny * nz) + j * nz + k

        k is the fastest-changing dimension.
        """
        return i * (self.ny * self.nz) + j * self.nz + k

    def reshape_field(self, arr_1d: np.ndarray) -> np.ndarray:
        return arr_1d.reshape(self.shape)

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def _initialize_fields(self):
        """
        Initialize Ez with a centered 3D Gaussian pulse.
        All other fields start from zero.
        """
        sigma2 = self.cfg.init_sigma ** 2
        amp = self.cfg.init_amplitude

        for i in range(self.nx):
            x = (i + 0.5) * self.dx - self.cfg.cell_x / 2.0
            for j in range(self.ny):
                y = (j + 0.5) * self.dy - self.cfg.cell_y / 2.0
                for k in range(self.nz):
                    z = (k + 0.5) * self.dz - self.cfg.cell_z / 2.0
                    r2 = x * x + y * y + z * z
                    self.Ez[self.idx(i, j, k)] = amp * np.exp(-r2 / (2.0 * sigma2))

    # -------------------------------------------------------------------------
    # H updates
    # -------------------------------------------------------------------------
    def update_Hx(self):
        """
        Hx -= dt * (dEz/dy - dEy/dz)

        Valid range:
          i in [0, nx-1]
          j in [0, ny-2]
          k in [0, nz-2]
        """
        dt, dy, dz = self.cfg.dt, self.dy, self.dz
        Hx, Ey, Ez = self.Hx, self.Ey, self.Ez

        for i in range(self.nx):
            for j in range(self.ny - 1):
                for k in range(self.nz - 1):
                    c = self.idx(i, j, k)
                    jp = self.idx(i, j + 1, k)
                    kp = self.idx(i, j, k + 1)

                    dEz_dy = (Ez[jp] - Ez[c]) / dy
                    dEy_dz = (Ey[kp] - Ey[c]) / dz

                    Hx[c] -= dt * (dEz_dy - dEy_dz)

    def update_Hy(self):
        """
        Hy -= dt * (dEx/dz - dEz/dx)

        Valid range:
          i in [0, nx-2]
          j in [0, ny-1]
          k in [0, nz-2]
        """
        dt, dx, dz = self.cfg.dt, self.dx, self.dz
        Hy, Ex, Ez = self.Hy, self.Ex, self.Ez

        for i in range(self.nx - 1):
            for j in range(self.ny):
                for k in range(self.nz - 1):
                    c = self.idx(i, j, k)
                    ip = self.idx(i + 1, j, k)
                    kp = self.idx(i, j, k + 1)

                    dEx_dz = (Ex[kp] - Ex[c]) / dz
                    dEz_dx = (Ez[ip] - Ez[c]) / dx

                    Hy[c] -= dt * (dEx_dz - dEz_dx)

    def update_Hz(self):
        """
        Hz -= dt * (dEy/dx - dEx/dy)

        Valid range:
          i in [0, nx-2]
          j in [0, ny-2]
          k in [0, nz-1]
        """
        dt, dx, dy = self.cfg.dt, self.dx, self.dy
        Hz, Ex, Ey = self.Hz, self.Ex, self.Ey

        for i in range(self.nx - 1):
            for j in range(self.ny - 1):
                for k in range(self.nz):
                    c = self.idx(i, j, k)
                    ip = self.idx(i + 1, j, k)
                    jp = self.idx(i, j + 1, k)

                    dEy_dx = (Ey[ip] - Ey[c]) / dx
                    dEx_dy = (Ex[jp] - Ex[c]) / dy

                    Hz[c] -= dt * (dEy_dx - dEx_dy)

    def update_H(self):
        self.update_Hx()
        self.update_Hy()
        self.update_Hz()

    # -------------------------------------------------------------------------
    # E updates
    # -------------------------------------------------------------------------
    def update_Ex(self):
        """
        Ex += dt * (dHz/dy - dHy/dz)

        Valid range:
          i in [0, nx-1]
          j in [1, ny-1]
          k in [1, nz-1]
        """
        dt, dy, dz = self.cfg.dt, self.dy, self.dz
        Ex, Hy, Hz = self.Ex, self.Hy, self.Hz

        for i in range(self.nx):
            for j in range(1, self.ny):
                for k in range(1, self.nz):
                    c = self.idx(i, j, k)
                    jm = self.idx(i, j - 1, k)
                    km = self.idx(i, j, k - 1)

                    dHz_dy = (Hz[c] - Hz[jm]) / dy
                    dHy_dz = (Hy[c] - Hy[km]) / dz

                    Ex[c] += dt * (dHz_dy - dHy_dz)

    def update_Ey(self):
        """
        Ey += dt * (dHx/dz - dHz/dx)

        Valid range:
          i in [1, nx-1]
          j in [0, ny-1]
          k in [1, nz-1]
        """
        dt, dx, dz = self.cfg.dt, self.dx, self.dz
        Ey, Hx, Hz = self.Ey, self.Hx, self.Hz

        for i in range(1, self.nx):
            for j in range(self.ny):
                for k in range(1, self.nz):
                    c = self.idx(i, j, k)
                    im = self.idx(i - 1, j, k)
                    km = self.idx(i, j, k - 1)

                    dHx_dz = (Hx[c] - Hx[km]) / dz
                    dHz_dx = (Hz[c] - Hz[im]) / dx

                    Ey[c] += dt * (dHx_dz - dHz_dx)

    def update_Ez(self):
        """
        Ez += dt * (dHy/dx - dHx/dy)

        Valid range:
          i in [1, nx-1]
          j in [1, ny-1]
          k in [0, nz-1]
        """
        dt, dx, dy = self.cfg.dt, self.dx, self.dy
        Ez, Hx, Hy = self.Ez, self.Hx, self.Hy

        for i in range(1, self.nx):
            for j in range(1, self.ny):
                for k in range(self.nz):
                    c = self.idx(i, j, k)
                    im = self.idx(i - 1, j, k)
                    jm = self.idx(i, j - 1, k)

                    dHy_dx = (Hy[c] - Hy[im]) / dx
                    dHx_dy = (Hx[c] - Hx[jm]) / dy

                    Ez[c] += dt * (dHy_dx - dHx_dy)

    def update_E(self):
        self.update_Ex()
        self.update_Ey()
        self.update_Ez()

    # -------------------------------------------------------------------------
    # Simulation control
    # -------------------------------------------------------------------------
    def step(self):
        """
        One full timestep:
          1) update H
          2) update E
        """
        self.update_H()
        self.update_E()

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    def field_stats_1d(self, arr: np.ndarray):
        cx = self.nx // 2
        cy = self.ny // 2
        cz = self.nz // 2
        center = arr[self.idx(cx, cy, cz)]
        return arr.min(), arr.max(), center

    def print_stats(self, step: int):
        print(f"step {step:03d}")

        for name, arr in [
            ("Ex", self.Ex),
            ("Ey", self.Ey),
            ("Ez", self.Ez),
            ("Hx", self.Hx),
            ("Hy", self.Hy),
            ("Hz", self.Hz),
        ]:
            vmin, vmax, center = self.field_stats_1d(arr)
            print(f"  {name}: min={vmin:.6e}, max={vmax:.6e}, center={center:.6e}")

        print()

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    def save_snapshot(self, step: int):
        os.makedirs(self.cfg.outdir, exist_ok=True)

        # Save full 3D fields as reshaped arrays
        Ex3 = self.reshape_field(self.Ex)
        Ey3 = self.reshape_field(self.Ey)
        Ez3 = self.reshape_field(self.Ez)
        Hx3 = self.reshape_field(self.Hx)
        Hy3 = self.reshape_field(self.Hy)
        Hz3 = self.reshape_field(self.Hz)

        np.save(os.path.join(self.cfg.outdir, f"Ex_step_{step:03d}.npy"), Ex3)
        np.save(os.path.join(self.cfg.outdir, f"Ey_step_{step:03d}.npy"), Ey3)
        np.save(os.path.join(self.cfg.outdir, f"Ez_step_{step:03d}.npy"), Ez3)
        np.save(os.path.join(self.cfg.outdir, f"Hx_step_{step:03d}.npy"), Hx3)
        np.save(os.path.join(self.cfg.outdir, f"Hy_step_{step:03d}.npy"), Hy3)
        np.save(os.path.join(self.cfg.outdir, f"Hz_step_{step:03d}.npy"), Hz3)

        # Save middle-z Ez slice for quick visualization
        z_mid = self.nz // 2
        Ez_slice = Ez3[:, :, z_mid]

        np.save(os.path.join(self.cfg.outdir, f"Ez_slice_step_{step:03d}.npy"), Ez_slice)

        plt.figure()
        plt.imshow(Ez_slice, origin="lower")
        plt.colorbar()
        plt.title(f"Ez middle-z slice, step {step}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.cfg.outdir, f"Ez_slice_step_{step:03d}.png"),
            dpi=140
        )
        plt.close()

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------
    def run(self):
        print("=== CUDA-Friendly FDTD Stencil Baseline ===")
        print(f"grid shape = ({self.nx}, {self.ny}, {self.nz})")
        print(f"num_cells  = {self.num_cells}")
        print(f"dx = {self.dx}, dy = {self.dy}, dz = {self.dz}")
        print(f"dt = {self.cfg.dt}")
        print(f"num_steps = {self.cfg.num_steps}")
        print(f"save_interval = {self.cfg.save_interval}")
        print()

        self.save_snapshot(step=0)
        self.print_stats(step=0)

        for step in range(1, self.cfg.num_steps + 1):
            self.step()

            if step % self.cfg.save_interval == 0:
                self.save_snapshot(step)
                self.print_stats(step)

        print(f"Outputs saved to ./{self.cfg.outdir}/")


def main():
    cfg = Config()
    sim = FDTDStencilBaselineFlat(cfg)
    sim.run()


if __name__ == "__main__":
    main()
