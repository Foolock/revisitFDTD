'''
This baseline is a simplified 3D E/H stencil system inspired by FDTD updates.
It is intended as a deterministic reference for implementation comparison, not as a full physical FDTD simulator.
'''

import os
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Config:
    # logical domain size
    cell_x: float = 4.0
    cell_y: float = 4.0
    cell_z: float = 4.0

    # grid resolution: points per unit length
    resolution: int = 8

    # time stepping
    dt: float = 0.05
    num_steps: int = 20
    save_interval: int = 2

    # initial condition
    init_sigma: float = 0.25
    init_amplitude: float = 1.0

    # output
    outdir: str = "baseline_output"
    dtype: type = np.float64


class FDTDStencilBaseline:
    """
    A simplified 3D E/H stencil system inspired by FDTD.

    Design goals:
      - clean stencil updates
      - no boundary conditions / PML
      - no material models
      - no source injection during stepping
      - deterministic baseline for CUDA comparison

    Notes:
      - all field components use the same array shape
      - boundary cells are left untouched by interior updates
      - this is intentionally a simplified colocated-grid baseline
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.nx = int(cfg.cell_x * cfg.resolution)
        self.ny = int(cfg.cell_y * cfg.resolution)
        self.nz = int(cfg.cell_z * cfg.resolution)

        self.dx = 1.0 / cfg.resolution
        self.dy = 1.0 / cfg.resolution
        self.dz = 1.0 / cfg.resolution

        shape = (self.nx, self.ny, self.nz)
        dtype = cfg.dtype

        self.Ex = np.zeros(shape, dtype=dtype)
        self.Ey = np.zeros(shape, dtype=dtype)
        self.Ez = np.zeros(shape, dtype=dtype)

        self.Hx = np.zeros(shape, dtype=dtype)
        self.Hy = np.zeros(shape, dtype=dtype)
        self.Hz = np.zeros(shape, dtype=dtype)

        self._initialize_fields()

    def _initialize_fields(self):
        """
        Initialize only Ez with a centered 3D Gaussian pulse.
        No external source is used afterward.
        """
        xs = (np.arange(self.nx) + 0.5) * self.dx - self.cfg.cell_x / 2.0
        ys = (np.arange(self.ny) + 0.5) * self.dy - self.cfg.cell_y / 2.0
        zs = (np.arange(self.nz) + 0.5) * self.dz - self.cfg.cell_z / 2.0

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

        sigma2 = self.cfg.init_sigma ** 2
        amp = self.cfg.init_amplitude

        self.Ez[:, :, :] = amp * np.exp(-(X**2 + Y**2 + Z**2) / (2.0 * sigma2))

    def update_H(self):
        """
        Update H from spatial differences of E.

        Only interior-adjacent regions are updated.
        Boundary cells remain unchanged.
        """
        dt = self.cfg.dt
        dx, dy, dz = self.dx, self.dy, self.dz

        Ex, Ey, Ez = self.Ex, self.Ey, self.Ez
        Hx, Hy, Hz = self.Hx, self.Hy, self.Hz

        # Hx -= dt * (dEz/dy - dEy/dz)
        Hx[:, :-1, :-1] -= dt * (
            (Ez[:, 1:, :-1] - Ez[:, :-1, :-1]) / dy
            - (Ey[:, :-1, 1:] - Ey[:, :-1, :-1]) / dz
        )

        # Hy -= dt * (dEx/dz - dEz/dx)
        Hy[:-1, :, :-1] -= dt * (
            (Ex[:-1, :, 1:] - Ex[:-1, :, :-1]) / dz
            - (Ez[1:, :, :-1] - Ez[:-1, :, :-1]) / dx
        )

        # Hz -= dt * (dEy/dx - dEx/dy)
        Hz[:-1, :-1, :] -= dt * (
            (Ey[1:, :-1, :] - Ey[:-1, :-1, :]) / dx
            - (Ex[:-1, 1:, :] - Ex[:-1, :-1, :]) / dy
        )

    def update_E(self):
        """
        Update E from spatial differences of H.

        Only interior-adjacent regions are updated.
        Boundary cells remain unchanged.
        """
        dt = self.cfg.dt
        dx, dy, dz = self.dx, self.dy, self.dz

        Ex, Ey, Ez = self.Ex, self.Ey, self.Ez
        Hx, Hy, Hz = self.Hx, self.Hy, self.Hz

        # Ex += dt * (dHz/dy - dHy/dz)
        Ex[:, 1:, 1:] += dt * (
            (Hz[:, 1:, 1:] - Hz[:, :-1, 1:]) / dy
            - (Hy[:, 1:, 1:] - Hy[:, 1:, :-1]) / dz
        )

        # Ey += dt * (dHx/dz - dHz/dx)
        Ey[1:, :, 1:] += dt * (
            (Hx[1:, :, 1:] - Hx[1:, :, :-1]) / dz
            - (Hz[1:, :, 1:] - Hz[:-1, :, 1:]) / dx
        )

        # Ez += dt * (dHy/dx - dHx/dy)
        Ez[1:, 1:, :] += dt * (
            (Hy[1:, 1:, :] - Hy[:-1, 1:, :]) / dx
            - (Hx[1:, 1:, :] - Hx[1:, :-1, :]) / dy
        )

    def step(self):
        """
        One full timestep:
          1) update H
          2) update E
        """
        self.update_H()
        self.update_E()

    def field_stats(self):
        cx = self.nx // 2
        cy = self.ny // 2
        cz = self.nz // 2

        return {
            "Ex": (self.Ex.min(), self.Ex.max(), self.Ex[cx, cy, cz]),
            "Ey": (self.Ey.min(), self.Ey.max(), self.Ey[cx, cy, cz]),
            "Ez": (self.Ez.min(), self.Ez.max(), self.Ez[cx, cy, cz]),
            "Hx": (self.Hx.min(), self.Hx.max(), self.Hx[cx, cy, cz]),
            "Hy": (self.Hy.min(), self.Hy.max(), self.Hy[cx, cy, cz]),
            "Hz": (self.Hz.min(), self.Hz.max(), self.Hz[cx, cy, cz]),
        }

    def save_snapshot(self, step: int):
        """
        Save all field arrays and the middle-z slice of Ez.
        """
        outdir = self.cfg.outdir
        os.makedirs(outdir, exist_ok=True)

        # save full 3D fields
        np.save(os.path.join(outdir, f"Ex_step_{step:03d}.npy"), self.Ex)
        np.save(os.path.join(outdir, f"Ey_step_{step:03d}.npy"), self.Ey)
        np.save(os.path.join(outdir, f"Ez_step_{step:03d}.npy"), self.Ez)
        np.save(os.path.join(outdir, f"Hx_step_{step:03d}.npy"), self.Hx)
        np.save(os.path.join(outdir, f"Hy_step_{step:03d}.npy"), self.Hy)
        np.save(os.path.join(outdir, f"Hz_step_{step:03d}.npy"), self.Hz)

        # save middle z slice for quick visualization
        z_mid = self.nz // 2
        ez_slice = self.Ez[:, :, z_mid]

        np.save(os.path.join(outdir, f"Ez_slice_step_{step:03d}.npy"), ez_slice)

        plt.figure()
        plt.imshow(ez_slice, origin="lower")
        plt.colorbar()
        plt.title(f"Ez middle-z slice, step {step}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"Ez_slice_step_{step:03d}.png"), dpi=140)
        plt.close()

    def run(self):
        """
        Run the simulation and save snapshots.
        """
        print("=== FDTD Stencil Baseline ===")
        print(f"grid shape = ({self.nx}, {self.ny}, {self.nz})")
        print(f"dx = dy = dz = {self.dx}")
        print(f"dt = {self.cfg.dt}")
        print(f"num_steps = {self.cfg.num_steps}")
        print()

        self.save_snapshot(step=0)
        self._print_stats(step=0)

        for step in range(1, self.cfg.num_steps + 1):
            self.step()

            if step % self.cfg.save_interval == 0:
                self.save_snapshot(step=step)
                self._print_stats(step=step)

        print()
        print(f"Outputs saved to ./{self.cfg.outdir}/")

    def _print_stats(self, step: int):
        stats = self.field_stats()
        print(f"step {step:03d}")
        for name in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            vmin, vmax, center = stats[name]
            print(
                f"  {name}: min={vmin:.6e}, max={vmax:.6e}, center={center:.6e}"
            )
        print()


def main():
    cfg = Config()
    sim = FDTDStencilBaseline(cfg)
    sim.run()


if __name__ == "__main__":
    main()
