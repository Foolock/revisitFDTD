import os
import meep as mp
import numpy as np
import matplotlib.pyplot as plt


def gaussian_ic(p, sigma=0.25):
    r2 = p.x**2 + p.y**2 + p.z**2
    return np.exp(-r2 / (2 * sigma * sigma))


def save_ez_slice(sim, step, cell, outdir="meep_aligned"):
    os.makedirs(outdir, exist_ok=True)

    ez = sim.get_array(
        component=mp.Ez,
        center=mp.Vector3(),
        size=cell
    )

    z_mid = ez.shape[2] // 2
    slice_2d = ez[:, :, z_mid]

    np.save(f"{outdir}/ez_step_{step:03d}.npy", slice_2d)

    plt.figure()
    plt.imshow(slice_2d, origin="lower")
    plt.colorbar()
    plt.title(f"Meep Ez slice step {step}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/ez_step_{step:03d}.png", dpi=120)
    plt.close()


def main():
    resolution = 8
    cell = mp.Vector3(4, 4, 4)

    # Explicitly set Courant so the timestep is unambiguous
    courant = 0.5
    dt = courant / resolution

    sim = mp.Simulation(
        cell_size=cell,
        resolution=resolution,
        dimensions=3,
        sources=[],
        boundary_layers=[],
        Courant=courant
    )

    # Initialize internal structures first, then set initial field
    sim.init_sim()

    # If sim.initialize_field(...) works in your install, you can use that too.
    sim.fields.initialize_field(mp.Ez, gaussian_ic)

    total_steps = 20
    save_interval = 2

    current_time = 0.0

    # Save step 0 too, useful for debugging
    save_ez_slice(sim, 0, cell)

    for step in range(1, total_steps + 1):
        sim.run(until=current_time + dt)
        current_time += dt

        if step % save_interval == 0:
            save_ez_slice(sim, step, cell)
            ez = sim.get_array(component=mp.Ez, center=mp.Vector3(), size=cell)
            print(f"step {step:03d}: Ez min={ez.min():.6e}, max={ez.max():.6e}")

    print("Done. Output in ./meep_aligned/")


if __name__ == "__main__":
    main()
