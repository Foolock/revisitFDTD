import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import os


def save_ez_slice(ez, step, outdir="slices"):
    os.makedirs(outdir, exist_ok=True)

    z_mid = ez.shape[2] // 2
    slice_2d = ez[:, :, z_mid]

    # 保存 numpy
    np.save(f"{outdir}/ez_step_{step:03d}.npy", slice_2d)

    # 保存图片（SSH可用）
    plt.figure()
    plt.imshow(slice_2d, origin="lower")
    plt.colorbar()
    plt.title(f"Ez slice at step {step}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/ez_step_{step:03d}.png", dpi=120)
    plt.close()


def main():
    # ----------------------------
    # 1. Basic setup
    # ----------------------------
    resolution = 8
    cell = mp.Vector3(4, 4, 4)

    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency=1.0, fwidth=0.4),
            component=mp.Ez,
            center=mp.Vector3()
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        resolution=resolution,
        boundary_layers=[mp.PML(0.5)],
        sources=sources,
        dimensions=3
    )

    # ----------------------------
    # 2. Time stepping control
    # ----------------------------
    total_steps = 20
    save_interval = 2

    current_time = 0

    for step in range(1, total_steps + 1):
        # 每次推进 1 timestep
        sim.run(until=current_time + 1)
        current_time += 1

        if step % save_interval == 0:
            ez = sim.get_array(component=mp.Ez)

            print("=" * 50)
            print(f"Step {step}")
            print("Ez min/max:", ez.min(), ez.max())

            save_ez_slice(ez, step)

    print("\nAll slices saved in ./slices/")


if __name__ == "__main__":
    main()
