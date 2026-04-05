import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def extract_steps(directory):
    files = glob.glob(f"{directory}/ez_step_*.npy")
    steps = sorted([
        int(f.split("_")[-1].split(".")[0])
        for f in files
    ])
    return steps


def compare_all(meep_dir="meep_aligned",
                numpy_dir="numpy_aligned",
                outdir="compare_all"):

    os.makedirs(outdir, exist_ok=True)

    steps = extract_steps(meep_dir)

    l2_list = []
    rel_list = []
    max_list = []

    print("Comparing steps:", steps)
    print()

    for step in steps:
        a = np.load(f"{meep_dir}/ez_step_{step:03d}.npy")
        b = np.load(f"{numpy_dir}/ez_step_{step:03d}.npy")

        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch at step {step}")

        diff = a - b

        l2 = np.linalg.norm(diff)
        rel = l2 / (np.linalg.norm(a) + 1e-12)
        max_abs = np.max(np.abs(diff))

        l2_list.append(l2)
        rel_list.append(rel)
        max_list.append(max_abs)

        print(f"step {step:03d} | "
              f"L2={l2:.3e} | rel={rel:.3e} | max={max_abs:.3e}")

    # ----------------------------
    # Plot error vs step
    # ----------------------------
    plt.figure()

    plt.plot(steps, l2_list, label="L2 error")
    plt.plot(steps, rel_list, label="Relative error")
    plt.plot(steps, max_list, label="Max abs error")

    plt.xlabel("Time step")
    plt.ylabel("Error")
    plt.title("Error vs Time Step")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{outdir}/error_curve.png", dpi=150)
    plt.close()

    print(f"\nSaved error curve to {outdir}/error_curve.png")

    # ----------------------------
    # Optional: save raw numbers
    # ----------------------------
    np.save(f"{outdir}/steps.npy", np.array(steps))
    np.save(f"{outdir}/l2.npy", np.array(l2_list))
    np.save(f"{outdir}/rel.npy", np.array(rel_list))
    np.save(f"{outdir}/max.npy", np.array(max_list))


if __name__ == "__main__":
    compare_all()
