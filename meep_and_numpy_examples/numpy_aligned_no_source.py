import os
import numpy as np
import matplotlib.pyplot as plt


def initialize_fields(nx, ny, nz, dx, dy, dz):
    Ex = np.zeros((nx, ny, nz), dtype=np.float64)
    Ey = np.zeros((nx, ny, nz), dtype=np.float64)
    Ez = np.zeros((nx, ny, nz), dtype=np.float64)

    Hx = np.zeros((nx, ny, nz), dtype=np.float64)
    Hy = np.zeros((nx, ny, nz), dtype=np.float64)
    Hz = np.zeros((nx, ny, nz), dtype=np.float64)

    # Same initial condition idea: Gaussian packet on Ez
    sigma = 0.25

    # Coordinates centered at zero, similar to Meep cell coordinates
    xs = (np.arange(nx) + 0.5) * dx - (nx * dx) / 2.0
    ys = (np.arange(ny) + 0.5) * dy - (ny * dy) / 2.0
    zs = (np.arange(nz) + 0.5) * dz - (nz * dz) / 2.0

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    Ez[:] = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma * sigma))

    return Ex, Ey, Ez, Hx, Hy, Hz


def update_H(Ex, Ey, Ez, Hx, Hy, Hz, dt, dx, dy, dz):
    Hx[:, :-1, :-1] -= dt * (
        (Ez[:, 1:, :-1] - Ez[:, :-1, :-1]) / dy
        - (Ey[:, :-1, 1:] - Ey[:, :-1, :-1]) / dz
    )

    Hy[:-1, :, :-1] -= dt * (
        (Ex[:-1, :, 1:] - Ex[:-1, :, :-1]) / dz
        - (Ez[1:, :, :-1] - Ez[:-1, :, :-1]) / dx
    )

    Hz[:-1, :-1, :] -= dt * (
        (Ey[1:, :-1, :] - Ey[:-1, :-1, :]) / dx
        - (Ex[:-1, 1:, :] - Ex[:-1, :-1, :]) / dy
    )


def update_E(Ex, Ey, Ez, Hx, Hy, Hz, dt, dx, dy, dz):
    Ex[:, 1:, 1:] += dt * (
        (Hz[:, 1:, 1:] - Hz[:, :-1, 1:]) / dy
        - (Hy[:, 1:, 1:] - Hy[:, 1:, :-1]) / dz
    )

    Ey[1:, :, 1:] += dt * (
        (Hx[1:, :, 1:] - Hx[1:, :, :-1]) / dz
        - (Hz[1:, :, 1:] - Hz[:-1, :, 1:]) / dx
    )

    Ez[1:, 1:, :] += dt * (
        (Hy[1:, 1:, :] - Hy[:-1, 1:, :]) / dx
        - (Hx[1:, 1:, :] - Hx[1:, :-1, :]) / dy
    )


def save_ez_slice(Ez, step, outdir="numpy_aligned"):
    os.makedirs(outdir, exist_ok=True)

    z_mid = Ez.shape[2] // 2
    slice_2d = Ez[:, :, z_mid]

    np.save(f"{outdir}/ez_step_{step:03d}.npy", slice_2d)

    plt.figure()
    plt.imshow(slice_2d, origin="lower")
    plt.colorbar()
    plt.title(f"Numpy Ez slice step {step}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/ez_step_{step:03d}.png", dpi=120)
    plt.close()


def main():
    resolution = 8
    cell_x = cell_y = cell_z = 4.0

    nx = int(cell_x * resolution)
    ny = int(cell_y * resolution)
    nz = int(cell_z * resolution)

    dx = dy = dz = 1.0 / resolution
    dt = 0.5 / resolution   # match Meep Courant/resolution

    total_steps = 20
    save_interval = 2

    Ex, Ey, Ez, Hx, Hy, Hz = initialize_fields(nx, ny, nz, dx, dy, dz)

    save_ez_slice(Ez, 0)

    for step in range(1, total_steps + 1):
        update_H(Ex, Ey, Ez, Hx, Hy, Hz, dt, dx, dy, dz)
        update_E(Ex, Ey, Ez, Hx, Hy, Hz, dt, dx, dy, dz)

        if step % save_interval == 0:
            save_ez_slice(Ez, step)
            print(f"step {step:03d}: Ez min={Ez.min():.6e}, max={Ez.max():.6e}")

    print("Done. Output in ./numpy_aligned/")


if __name__ == "__main__":
    main()
