import meep as mp
import numpy as np
import matplotlib.pyplot as plt


def main():
    # ----------------------------
    # 1. Basic simulation settings
    # ----------------------------
    resolution = 10            # grid points per distance unit
    cell_size = mp.Vector3(4, 4, 4)

    # ----------------------------
    # 2. Simple source
    # ----------------------------
    # A continuous source placed at the center.
    # We excite Ez only, to keep things simple.
    sources = [
        mp.Source(
            src=mp.ContinuousSource(frequency=1.0),
            component=mp.Ez,
            center=mp.Vector3(0, 0, 0)
        )
    ]

    # ----------------------------
    # 3. Boundary layers
    # ----------------------------
    # Add PML so waves do not reflect too much from the boundary.
    # This is the simplest practical setup.
    pml_layers = [mp.PML(0.5)]

    # ----------------------------
    # 4. Build simulation
    # ----------------------------
    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        boundary_layers=pml_layers,
        sources=sources,
        dimensions=3
    )

    # ----------------------------
    # 5. Run simulation
    # ----------------------------
    sim.run(until=20)

    # ----------------------------
    # 6. Extract field data
    # ----------------------------
    ez = sim.get_array(component=mp.Ez)

    print("Ez shape:", ez.shape)
    print("Ez min:", np.min(ez))
    print("Ez max:", np.max(ez))
    print("Ez mean:", np.mean(ez))

    # Print one z-slice index in the middle
    z_mid = ez.shape[2] // 2
    print(f"Middle z-slice index: {z_mid}")
    print("Ez middle z-slice:")
    print(ez[:, :, z_mid])

    # ----------------------------
    # 7. Save a 2D slice image
    # ----------------------------
    plt.figure(figsize=(6, 5))
    plt.imshow(ez[:, :, z_mid], origin="lower")
    plt.colorbar(label="Ez")
    plt.title("Ez middle z-slice")
    plt.xlabel("x index")
    plt.ylabel("y index")
    plt.tight_layout()
    plt.savefig("ez_middle_slice.png", dpi=150)

    # ----------------------------
    # 8. Optionally save raw data
    # ----------------------------
    np.save("ez.npy", ez)

    print("Saved figure to ez_middle_slice.png")
    print("Saved raw Ez data to ez.npy")


if __name__ == "__main__":
    main()
