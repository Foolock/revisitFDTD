import meep as mp
import numpy as np


def print_field_stats(name, arr):
    print(f"{name} shape: {arr.shape}")
    print(f"{name} min : {arr.min():.6e}")
    print(f"{name} max : {arr.max():.6e}")
    print(f"{name} mean: {arr.mean():.6e}")

    cx = arr.shape[0] // 2
    cy = arr.shape[1] // 2
    cz = arr.shape[2] // 2
    print(f"{name} center value: {arr[cx, cy, cz]:.6e}")
    print()


def main():
    # --------------------------------------------------
    # 1. Basic setup
    # --------------------------------------------------
    resolution = 8
    cell = mp.Vector3(4, 4, 4)

    # Use a Gaussian source instead of a continuous source.
    # This is often nicer for debugging because the field does not just keep growing forever.
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency=1.0, fwidth=0.4),
            component=mp.Ez,
            center=mp.Vector3(0, 0, 0)
        )
    ]

    # PML is included to reduce boundary reflections.
    # For a practical reference setup, this is usually easier.
    pml_layers = [mp.PML(0.5)]

    sim = mp.Simulation(
        cell_size=cell,
        resolution=resolution,
        boundary_layers=pml_layers,
        sources=sources,
        dimensions=3
    )

    # --------------------------------------------------
    # 2. Run in small chunks and inspect fields
    # --------------------------------------------------
    sample_times = [2, 4, 6, 8, 10]

    for t in sample_times:
        sim.run(until=t)

        ez = sim.get_array(component=mp.Ez)
        ex = sim.get_array(component=mp.Ex)
        ey = sim.get_array(component=mp.Ey)
        hx = sim.get_array(component=mp.Hx)
        hy = sim.get_array(component=mp.Hy)
        hz = sim.get_array(component=mp.Hz)

        print("=" * 60)
        print(f"After run(until={t})")
        print_field_stats("Ez", ez)
        print_field_stats("Ex", ex)
        print_field_stats("Ey", ey)
        print_field_stats("Hx", hx)
        print_field_stats("Hy", hy)
        print_field_stats("Hz", hz)

    # --------------------------------------------------
    # 3. Save final arrays for later comparison
    # --------------------------------------------------
    np.save("ez_final.npy", ez)
    np.save("ex_final.npy", ex)
    np.save("ey_final.npy", ey)
    np.save("hx_final.npy", hx)
    np.save("hy_final.npy", hy)
    np.save("hz_final.npy", hz)

    print("Saved final field arrays to .npy files.")


if __name__ == "__main__":
    main()
