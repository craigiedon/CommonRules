import numpy as np
import matplotlib.pyplot as plt


def ray_intersect(ray_origin: np.ndarray, ray_direction: np.ndarray, corners: np.ndarray):
    assert np.isclose(np.linalg.norm(ray_direction), 1.0)

    c_dirs = np.diff(corners, axis=0, append=[corners[0]])
    # cd_norms = np.linalg.norm(c_dirs, axis=1).reshape(-1, 1)
    # c_dirs = c_dirs / cd_norms

    # p = ray_origin
    # r = ray_direction

    # q = corner
    # s = corner_dir

    ts = []
    us = []
    for c, cd in zip(corners, c_dirs):
        r_x_s = np.cross(ray_direction, cd)
        q_neg_p = c - ray_origin

        if r_x_s != 0:
            t = np.cross(q_neg_p, cd) / r_x_s
            u = np.cross(q_neg_p, ray_direction) / r_x_s
            ts.append(t)
            us.append(u)
        else:
            ts.append(-1000)
            us.append(-1000)

    intersections = []
    for t, u in zip(ts,us):
        if t > 0 and (0 <= u <= 1):
            intersections.append(ray_origin + t * ray_direction)
        else:
            intersections.append(np.array([np.inf, np.inf]))

    return np.array(intersections)


def run():
    ray_origin = np.array([0.0, 0.0])
    ray_direction = np.array([0.5, 0.5])
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    corners = np.array([
        [2.0, -30.0],
        [2.0, 30.0],
        [4.0, 30.0],
        [4.0, -30.0]
    ])

    intersection_points = ray_intersect(ray_origin, ray_direction, corners)
    valid_intersection_points = intersection_points[intersection_points[:, 0] != np.inf]

    plt.plot(corners[:, 0], corners[:, 1], c='b')

    plt.scatter(ray_origin[0], ray_origin[1], c='g')
    plt.plot([ray_origin[0], ray_origin[0] + 20 * ray_direction[0]], [ray_origin[1], ray_origin[1] + 20 * ray_direction[1]], c='g')

    plt.scatter(valid_intersection_points[:, 0], valid_intersection_points[:, 1], c='r')
    plt.show()

if __name__ == "__main__":
    run()