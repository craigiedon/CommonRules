from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def ray_intersect(ray_origins: np.ndarray, ray_directions: np.ndarray, corners: np.ndarray):
    assert np.isclose(np.linalg.norm(ray_directions, axis=1), 1.0).all()
    # p = ray_origin, r = ray_direction, q = corner, s = corner_dir

    c_dirs = np.diff(corners, axis=0, append=[corners[0]])

    r_x_ss = np.cross(ray_directions[:, None, :], c_dirs[None, :, :])
    q_neg_ps = corners[None, :, :] - ray_origins[:, None, :]

    tss = np.divide(np.cross(q_neg_ps, c_dirs[None, :, :]), r_x_ss, out=-1 * np.ones((len(ray_origins), len(corners))),
                    where=r_x_ss != 0)
    uss = np.divide(np.cross(q_neg_ps, ray_directions[:, None, :]), r_x_ss,
                    out=-1 * np.ones((len(ray_origins), len(corners))), where=r_x_ss != 0)

    hit_conds = (tss > 0) & (uss >= 0) & (uss <= 1)
    min_ts = np.amin(tss, axis=1, where=hit_conds, initial=np.inf)

    rays_intersections = ray_origins + min_ts[:, None] * ray_directions

    return rays_intersections, min_ts


def calc_occlusions(ray_origins: np.ndarray, ray_directions: np.ndarray, target_corners: np.ndarray,
                    occluders: List[np.ndarray]) -> Tuple[float, np.ndarray, np.ndarray, List[np.ndarray]]:
    target_hits, target_ls = ray_intersect(ray_origins, ray_directions, target_corners)

    free_hits_mask = np.isfinite(target_ls)
    visible_hits_mask = free_hits_mask

    occ_hits_masks = []
    for occ_corners in occluders:
        occ_hits, occ_ls = ray_intersect(ray_origins, ray_directions, occ_corners)
        visible_hits_mask = visible_hits_mask & (np.isinf(occ_ls) | (target_ls < occ_ls))

        occ_hits_mask = free_hits_mask & (np.isfinite(occ_ls) & (occ_ls < target_ls))
        occ_hits_masks.append(occ_hits_mask)

    free_hits = target_ls[free_hits_mask]
    viz_hits = target_ls[visible_hits_mask]

    if len(free_hits) == 0:
        viz_prop = 0.0
    else:
        viz_prop = len(viz_hits) / len(free_hits)

    return viz_prop, free_hits_mask, visible_hits_mask, occ_hits_masks


def run():
    n_rays = 120
    ray_origins = np.tile(np.array([0.0, 0.0]), (n_rays, 1))

    angles = np.linspace(0, np.pi * 2.0, num=n_rays)
    ray_directions = np.column_stack((np.cos(angles), np.sin(angles)))

    ray_norms = np.linalg.norm(ray_directions, axis=1).reshape(-1, 1)
    ray_directions = ray_directions / ray_norms

    occluder_shape = np.array([
        [2.0, -1.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [3.0, -1.0]
    ])

    occluders = [occluder_shape, occluder_shape + np.array([0.0, 4.0])]
    # occluders = [occluder_shape]

    target_shape = np.array([
        [2.0, -3.0],
        [2.0, 3.0],
        [4.0, 3.0],
        [4.0, -3.0]
    ]) + np.array([3.0, 4.0])

    # TODO: This is the bug region
    occ_intersects, occ_rls = ray_intersect(ray_origins, ray_directions, occluder_shape)
    target_intersects, target_rls = ray_intersect(ray_origins, ray_directions, target_shape)

    viz_prop, free_hits_mask, visible_hits_mask, occ_hits_masks = calc_occlusions(ray_origins, ray_directions, target_shape, occluders)

    print(viz_prop)

    for occ in occluders:
        plt.gca().add_patch(Polygon(occ, edgecolor='b', fill=False))
    plt.gca().add_patch(Polygon(target_shape, edgecolor='b', fill=False))
    # plt.plot(corners[:, 0], corners[:, 1], c='b')
    # plt.plot(corners[-1, 0], corners[0, 1], c='b')

    # Plot the occluded rays
    for occluded_hits_mask in occ_hits_masks:
        for ro, rd, ip, l in zip(ray_origins[occluded_hits_mask, :], ray_directions[occluded_hits_mask, :],
                                 occ_intersects[occluded_hits_mask, :], occ_rls[occluded_hits_mask]):
            # plt.scatter(ro[0], ro[1], c='g')

            plt.plot([ro[0], ro[0] + l * rd[0]], [ro[1], ro[1] + l * rd[1]], c='y')
            plt.scatter(ip[0], ip[1], c='r')

    # Plot the visible rays
    for ro, rd, ip, l in zip(ray_origins[visible_hits_mask, :], ray_directions[visible_hits_mask, :],
                             target_intersects[visible_hits_mask, :], target_rls[visible_hits_mask]):
        plt.scatter(ro[0], ro[1], c='g')

        # if np.isfinite(l):
        plt.plot([ro[0], ro[0] + l * rd[0]], [ro[1], ro[1] + l * rd[1]], c='g')
        plt.scatter(ip[0], ip[1], c='r')
        # else:
        #     small_l = 1
        #     plt.plot([ro[0], ro[0] + small_l * rd[0]], [ro[1], ro[1] + small_l * rd[1]], c='g')

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()


if __name__ == "__main__":
    run()
