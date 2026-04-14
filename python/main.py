from sys import argv
import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from imageio import v2 as imageio

PI = 3.1415926


def create_animation(path, title, fps=10):
    images = []
    image_files = sorted(glob.glob(os.path.join(path, "*.png")))

    if not image_files:
        print("No images found to animate.")
        return

    output_path = f"{path}_{"_".join(title.split(" "))}.mp4"

    with imageio.get_writer(output_path, fps=fps) as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"Animation saved to: {output_path}")


def plot_ellipsoids(
    centers, directions, path, a, c, rve_side_len=1.0, title=None, iteration=0
):
    fig = plt.figure(figsize=(10.08, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Create base sphere coordinates
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = a * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))

    for center, target_dir in zip(centers, directions):
        target_dir = target_dir / np.linalg.norm(target_dir)

        z_axis = np.array([0, 0, 1])
        v_cross = np.cross(z_axis, target_dir)
        s = np.linalg.norm(v_cross)
        cos_c = np.dot(z_axis, target_dir)

        if s == 0:
            R = np.eye(3) if cos_c > 0 else -np.eye(3)
        else:
            vx = np.array(
                [
                    [0, -v_cross[2], v_cross[1]],
                    [v_cross[2], 0, -v_cross[0]],
                    [-v_cross[1], v_cross[0], 0],
                ]
            )
            R = np.eye(3) + vx + (vx @ vx) * ((1 - cos_c) / (s**2))
        points = np.stack([x.flatten(), y.flatten(), z.flatten()])
        rotated_points = R @ points

        x_p = rotated_points[0, :].reshape(x.shape) + center[0]
        y_p = rotated_points[1, :].reshape(y.shape) + center[1]
        z_p = rotated_points[2, :].reshape(z.shape) + center[2]

        ax.plot_surface(
            x_p, y_p, z_p, alpha=1.0, color="blue", edgecolors=None, linewidth=None
        )
        start = center + c * target_dir
        end = center + a * target_dir
        ax.plot((start[0], end[0]), (start[1], end[1]), (start[2], end[2]), color="red")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-0.5 * rve_side_len, 0.5 * rve_side_len)
    ax.set_ylim(-0.5 * rve_side_len, 0.5 * rve_side_len)
    ax.set_zlim(-0.5 * rve_side_len, 0.5 * rve_side_len)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.suptitle(title)
    plt.savefig(f"{path}/{iteration:0>8}.png", dpi=100)
    plt.close(fig)


PATH = "out"


def run_on_path(path):
    with open(path + "/config.json") as file:
        config = json.load(file)
    print("\nPlotting run:", config["name"])
    frames = sorted(
        [
            int(p.removesuffix("_pos.npy").removeprefix(path + "/"))
            for p in glob.glob(os.path.join(path, "*_pos.npy"))
        ]
    )

    for i in frames:
        frame_n = f"{i:0>5}"
        p_file = path + "/" + frame_n + "_pos.npy"
        d_file = path + "/" + frame_n + "_dir.npy"
        if os.path.exists(d_file):
            centers = np.load(p_file)
            directions = np.load(d_file)
            plot_ellipsoids(
                centers,
                directions,
                path,
                a=config["big_saxis"],
                c=config["small_saxis"],
                rve_side_len=config["rve_side_len"],
                title=f"{config["name"]} {config["duration"] * config["log_frames"] * i:.1f} ms",
                iteration=i,
            )
    print("\nMaking animation: ", config["name"])
    create_animation(path, title = config["name"], fps=10)


def main():
    if len(argv) > 1:
        for dir in argv[1:]:
            run_on_path(dir)
    else:
        last_dir = sorted(
            [dir for dir in os.listdir(PATH) if os.path.isdir(f"{PATH}/{dir}")]
        )[-1]
        run_on_path(f"{PATH}/{last_dir}")


if __name__ == "__main__":
    main()
