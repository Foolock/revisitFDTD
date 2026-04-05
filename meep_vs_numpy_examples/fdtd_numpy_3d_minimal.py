import numpy as np


def initialize_fields(nx, ny, nz, dtype=np.float64):
    Ex = np.zeros((nx, ny, nz), dtype=dtype)
    Ey = np.zeros((nx, ny, nz), dtype=dtype)
    Ez = np.zeros((nx, ny, nz), dtype=dtype)

    Hx = np.zeros((nx, ny, nz), dtype=dtype)
    Hy = np.zeros((nx, ny, nz), dtype=dtype)
    Hz = np.zeros((nx, ny, nz), dtype=dtype)

    return Ex, Ey, Ez, Hx, Hy, Hz


def add_simple_source(Ez, t, amplitude=1.0, omega=0.2):
    cx = Ez.shape[0] // 2
    cy = Ez.shape[1] // 2
    cz = Ez.shape[2] // 2
    Ez[cx, cy, cz] += amplitude * np.sin(omega * t)


def update_H(Ex, Ey, Ez, Hx, Hy, Hz, dt, dx, dy, dz):
    # Ignore boundary cells for now
    # This is the simplest controlled version.

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


def print_stats(name, arr):
    cx = arr.shape[0] // 2
    cy = arr.shape[1] // 2
    cz = arr.shape[2] // 2
    print(f"{name}: min={arr.min():.6e}, max={arr.max():.6e}, mean={arr.mean():.6e}, center={arr[cx, cy, cz]:.6e}")


def main():
    # Grid size
    nx, ny, nz = 32, 32, 32

    # Space step
    dx = dy = dz = 1.0

    # Time step
    # For now keep it small and safe.
    dt = 0.1

    # Number of time steps
    nsteps = 50

    Ex, Ey, Ez, Hx, Hy, Hz = initialize_fields(nx, ny, nz)

    for t in range(nsteps):
        update_H(Ex, Ey, Ez, Hx, Hy, Hz, dt, dx, dy, dz)
        update_E(Ex, Ey, Ez, Hx, Hy, Hz, dt, dx, dy, dz)
        add_simple_source(Ez, t)

        if t % 10 == 0:
            print("=" * 60)
            print(f"step {t}")
            print_stats("Ex", Ex)
            print_stats("Ey", Ey)
            print_stats("Ez", Ez)
            print_stats("Hx", Hx)
            print_stats("Hy", Hy)
            print_stats("Hz", Hz)

    np.save("Ex.npy", Ex)
    np.save("Ey.npy", Ey)
    np.save("Ez.npy", Ez)
    np.save("Hx.npy", Hx)
    np.save("Hy.npy", Hy)
    np.save("Hz.npy", Hz)

    print("\nSaved final fields to Ex.npy, Ey.npy, Ez.npy, Hx.npy, Hy.npy, Hz.npy")


if __name__ == "__main__":
    main()
