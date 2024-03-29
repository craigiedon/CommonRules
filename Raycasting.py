from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


@dataclass
class RayBatch:
    origins: np.ndarray
    dirs: np.ndarray


@dataclass
class RaycastsResult:
    origins: np.ndarray
    dirs: np.ndarray
    intersection_points: np.ndarray
    lengths: np.ndarray
    hit_mask: np.ndarray  # A boolean mask indicating which rays intersected with the target, and which did not


def radial_ray_batch(origin: np.ndarray, n_rays: int) -> RayBatch:
    ray_origins = np.tile(origin, (n_rays, 1))

    angles = np.linspace(0.0, np.pi * 2.0, num=n_rays)

    ray_directions = np.column_stack((np.cos(angles), np.sin(angles)))
    ray_norms = np.linalg.norm(ray_directions, axis=1).reshape(-1, 1)
    ray_directions = ray_directions / ray_norms

    return RayBatch(ray_origins, ray_directions)


def ray_intersect(ray_origins: np.ndarray, ray_directions: np.ndarray, corners: np.ndarray) -> RaycastsResult:
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

    return RaycastsResult(ray_origins, ray_directions, rays_intersections, min_ts, np.isfinite(min_ts))


def occlusions_from_ray_hits(target_res: RaycastsResult, occs_res: List[RaycastsResult]) -> Tuple[float, np.ndarray, List[np.ndarray]]:
    # target_hits, target_ls = ray_intersect(ray_origins, ray_directions, target_corners)

    # free_hits_mask = np.isfinite(target_ls)
    visible_hits_mask = target_res.hit_mask

    occ_hits_masks = []
    for o_res in occs_res:
        # occ_hits, occ_ls = ray_intersect(ray_origins, ray_directions, occ_corners)
        visible_hits_mask = visible_hits_mask & (~o_res.hit_mask | (target_res.lengths < o_res.lengths))

        occ_hits_mask = target_res.hit_mask & (o_res.hit_mask & (o_res.lengths < target_res.lengths))
        occ_hits_masks.append(occ_hits_mask)

    free_hits = target_res.lengths[target_res.hit_mask]
    viz_hits = target_res.lengths[visible_hits_mask]

    if len(free_hits) == 0:
        viz_prop = 0.0
    else:
        viz_prop = len(viz_hits) / len(free_hits)

    return viz_prop, visible_hits_mask, occ_hits_masks


def run():
    rays = radial_ray_batch(np.array([0.0, 0.0]), 120)

    occluder_shape = np.array([
        [2.0, -1.0],
        [2.0, 1.0],
        [3.0, 1.0],
        [3.0, -1.0]
    ])

    occluders = [occluder_shape, occluder_shape + np.array([0.0, 4.0])]

    target_shape = np.array([
        [2.0, -3.0],
        [2.0, 3.0],
        [4.0, 3.0],
        [4.0, -3.0]
    ]) + np.array([3.0, 4.0])

    target_res = ray_intersect(rays.origins, rays.dirs, target_shape)

    occs_res = [ray_intersect(rays.origins, rays.dirs, occ) for occ in occluders]
    viz_prop, visible_hits_mask, occ_hits_masks = occlusions_from_ray_hits(target_res, occs_res)

    print(viz_prop)

    for occ in occluders:
        plt.gca().add_patch(Polygon(occ, edgecolor='b', fill=False))
    plt.gca().add_patch(Polygon(target_shape, edgecolor='b', fill=False))

    # Plot the occluded rays
    for o_res, occ_hit_mask in zip(occs_res, occ_hits_masks):
        for ro, rd, ip, l in zip(rays.origins[occ_hit_mask, :], rays.dirs[occ_hit_mask, :],
                                 o_res.intersection_points[occ_hit_mask, :], o_res.lengths[occ_hit_mask]):
            plt.plot([ro[0], ro[0] + l * rd[0]], [ro[1], ro[1] + l * rd[1]], c='y')
            plt.scatter(ip[0], ip[1], c='r')

    # Plot the visible rays
    for ro, rd, ip, l in zip(rays.origins[visible_hits_mask, :], rays.dirs[visible_hits_mask, :],
                             target_res.intersection_points[visible_hits_mask, :],
                             target_res.lengths[visible_hits_mask]):
        plt.scatter(ro[0], ro[1], c='g')

        plt.plot([ro[0], ro[0] + l * rd[0]], [ro[1], ro[1] + l * rd[1]], c='g')
        plt.scatter(ip[0], ip[1], c='r')

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()


if __name__ == "__main__":
    run()
