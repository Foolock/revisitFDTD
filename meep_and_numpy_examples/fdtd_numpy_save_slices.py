import os
import numpy as np
import matplotlib.pyplot as plt


def initialize_fields(nx, ny, nz, dtype=np.float64):
    Ex = np.zeros((nx, ny, nz), dtype=dtype)
    Ey = np.zeros((nx, ny, nz), dtype=dtype)
    Ez = np.zeros((nx, ny, nz), dtype=dtype)

    Hx = np.zeros((nx, ny, nz), dtype=dtype)
    Hy = np.zeros((nx, ny, nz), dtype=dtype)
    Hz = np.zeros((nx, ny, nz), dtype=dtype)

    return Ex, Ey, Ez, Hx, Hy, Hz


def gaussian_source_t(t, frequency=1.0, fwidth=0.4, t0=8.0):
    """
    A simple time-domain Gaussian-modulated sinusoid.
    This is only an analogous source to Meep's GaussianSource,
    not an exact reproduction of its internal behavior.
    """
    envelope = np.exp(-((t - t0) ** 2) / (2.0 * (1.0 / fwidth) ** 2))
    carrier = np.sin(2.0 * np.pi * frequency * t)
    return envelope * carrier


def add_source(Ez, t, amplitude=1.0, frequency=1.0, fwidth=0.4, t0=8.0):
    cx = Ez.shape[0] // 2
    cy = Ez.shape[1] // 2
    cz = Ez.shape[2] // 2
    Ez[cx, cy, cz] += amplitude * gaussian_source_t(
        t, frequency=frequency, fwidth=fwidth, t0=t0
    )


def update_H(Ex, Ey, Ez, Hx, Hy, Hz, dt, dx, dy, dz):
    # Hx = Hx - dt * (dEz/dy - dEy/dz)
    Hx[:, :-1, :-1] -= dt * (
        (Ez[:, 1:, :-1] - Ez[:, :-1, :-1]) / dy
        - (Ey[:, :-1, 1:] - Ey[:, :-1, :-1]) / dz
    )

    # Hy = Hy - dt * (dEx/dz - dEz/dx)
    Hy[:-1, :, :-1] -= dt * (
        (Ex[:-1, :, 1:] - Ex[:-1, :, :-1]) / dz
        - (Ez[1:, :, :-1] - Ez[:-1, :, :-1]) / dx
    )

    # Hz = Hz - dt * (dEy/dx - dEx/dy)
    Hz[:-1, :-1, :] -= dt * (
        (Ey[1:, :-1, :] - Ey[:-1, :-1, :]) / dx
        - (Ex[:-1, 1:, :] - Ex[:-1, :-1, :]) / dy
    )


def update_E(Ex, Ey, Ez, Hx, Hy, Hz, dt, dx, dy, dz):
    # Ex = Ex + dt * (dHz/dy - dHy/dz)
    Ex[:, 1:, 1:] += dt * (
        (Hz[:, 1:, 1:] - Hz[:, :-1, 1:]) / dy
        - (Hy[:, 1:, 1:] - Hy[:, 1:, :-1]) / dz
    )

    # Ey = Ey + dt * (dHx/dz - dHz/dx)
    Ey[1:, :, 1:] += dt * (
        (Hx[1:, :, 1:] - Hx[1:, :, :-1]) / dz
        - (Hz[1:, :, 1:] - Hz[:-1, :, 1:]) / dx
    )

    # Ez = Ez + dt * (dHy/dx - dHx/dy)
    Ez[1:, 1:, :] += dt * (
        (Hy[1:, 1:, :] - Hy[:-1, 1:, :]) / dx
        - (Hx[1:, 1:, :] - Hx[1:, :-1, :]) / dy
    )


def save_ez_slice(Ez, step, outdir="numpy_slices"):
    os.makedirs(outdir, exist_ok=True)

    z_mid = Ez.shape[2] // 2
    slice_2d = Ez[:, :, z_mid]

    np.save(f"{outdir}/ez_step_{step:03d}.npy", slice_2d)

    plt.figure()
    plt.imshow(slice_2d, origin="lower")
    plt.colorbar()
    plt.title(f"Numpy Ez slice at step {step}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/ez_step_{step:03d}.png", dpi=120)
    plt.close()


def main():
    # --------------------------------------------------
    # 1. Match the same space setup as the Meep script
    # --------------------------------------------------
    resolution = 8
    cell_x, cell_y, cell_z = 4.0, 4.0, 4.0

    nx = int(cell_x * resolution)
    ny = int(cell_y * resolution)
    nz = int(cell_z * resolution)

    dx = 1.0 / resolution
    dy = 1.0 / resolution
    dz = 1.0 / resolution

    # A conservative dt for stability
    dt = 0.05

    total_steps = 20
    save_interval = 2

    Ex, Ey, Ez, Hx, Hy, Hz = initialize_fields(nx, ny, nz)

    print(f"Grid shape: ({nx}, {ny}, {nz})")
    print(f"dx = dy = dz = {dx}")
    print(f"dt = {dt}")

    for step in range(1, total_steps + 1):
        update_H(Ex, Ey, Ez, Hx, Hy, Hz, dt, dx, dy, dz)
        update_E(Ex, Ey, Ez, Hx, Hy, Hz, dt, dx, dy, dz)

        # Use the same conceptual source setup:
        # source at center, applied to Ez
        add_source(
            Ez,
            t=step * dt,
            amplitude=1.0,
            frequency=1.0,
            fwidth=0.4,
            t0=8.0 * dt
        )

        if step % save_interval == 0:
            print("=" * 50)
            print(f"Step {step}")
            print(f"Ez min/max: {Ez.min():.6e}, {Ez.max():.6e}")
            save_ez_slice(Ez, step)

    print("\nAll numpy slices saved in ./numpy_slices/")


if __name__ == "__main__":
    main()
