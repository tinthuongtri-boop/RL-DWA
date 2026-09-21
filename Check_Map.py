import matplotlib.pyplot as plt
import numpy as np

def debug_plot_grid(raw_env, grid_map, traj=None, obstacles=None):
    plt.figure(figsize=(8, 8))

    # Vẽ grid (flip lại cho đúng orientation)
    plt.imshow(grid_map, origin="lower", cmap="gray")

    # Vẽ trajectory
    if traj is not None:
        traj = np.array(traj)
        gx, gy = [], []
        for x, y, *_ in traj:
            g = raw_env.world_to_grid(x, y, resolution=0.1)
            gx.append(g[0])
            gy.append(g[1])
        plt.plot(gx, gy, 'b-', label="trajectory")

    # Vẽ obstacle từ lidar
    if obstacles is not None and len(obstacles) > 0:
        obs = np.array(obstacles)
        gx, gy = [], []
        for x, y in obs:
            g = raw_env.world_to_grid(x, y, resolution=0.1)
            gx.append(g[0])
            gy.append(g[1])
        plt.scatter(gx, gy, s=5, c='red', label="lidar hits")

    plt.legend()
    plt.title("Grid vs World check")
    plt.show()