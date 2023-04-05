import pickle
import time
from typing import List, Mapping, Any, Tuple, Optional

import numpy as np

from bbox_funcs import bbox_iou_numpy, bbox_iou
from hungarianMatching import hungarian_matching, Matching
from utils import rot_mat, angle_diff


def filter_classes(class_names: List[str], d_set: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    if "annos" in d_set:
        annos = d_set["annos"]
    else:
        annos = d_set

    total_objs = len(annos["name"])
    matching_ids = [i for i in range(total_objs) if annos["name"][i] in class_names]

    # If nothing matches, just return empty annotations
    if len(matching_ids) == 0:
        return {k: np.array([]) for k in annos.keys()}

    # Filter out unexpected classes from every numpy field in annoation dict
    return {k: v[matching_ids] for k, v in annos.items() if isinstance(v, np.ndarray) and v.shape[0] == total_objs}


def bounds_from_kitti_locs(loc: np.ndarray, dims: np.ndarray, rot_y: float) -> np.ndarray:
    """

    :param loc: 3-D Array: Right, Down, Forward ("x", "y", "z")
    :param dims: 3-D Array: "Length", "Width", "Height"
    :return: x-min, y-min, x-max, y-max (From a "top-down" view)
    """

    td_dims = dims[[0, 2]]
    td_loc = loc[[0, 2]]

    # Negate the rotation because, in KITTI, the y-axis is pointed *downwards* in camera view
    ry_mat = rot_mat(-rot_y)

    corner_offset = 0.5 * td_dims * (ry_mat @ np.array([1, 0]))
    c1 = td_loc + corner_offset
    c2 = td_loc - corner_offset

    x_min = min(c1[0], c2[0])
    y_min = min(c1[1], c2[1])
    x_max = max(c1[0], c2[0])
    y_max = max(c1[1], c2[1])

    return np.array([x_min, y_min, x_max, y_max])


def filter_matches(raw_matches: List[Tuple[Optional[int], Optional[int]]], model_bbs: np.ndarray, tru_bbs: np.ndarray,
                   iou_thresh: float) -> Matching:
    filtered_matches = []
    for mi, ti in raw_matches:
        if mi is None:
            filtered_matches.append((mi, ti))
        elif ti is not None:
            iou = bbox_iou_numpy(model_bbs[[mi]], tru_bbs[[ti]])
            if iou > iou_thresh:
                filtered_matches.append((mi, ti))
            else:
                filtered_matches.append((None, ti))
    return filtered_matches


def run():
    pp_fp = "/media/cinnes/Craig Files/OpenPCDet/output/kitti_models/pointpillar/default/eval/epoch_7728/val/default/result.pkl"
    val_ds_fp = "/media/cinnes/Craig Files/OpenPCDet/data/kitti/kitti_infos_val.pkl"

    # Load the OpenDetPC KITTI PointPillar Results
    with open(pp_fp, "rb") as f:
        pp_kitti_res = pickle.load(f)

    # Load the KITTI Validation Set itself!
    with open(val_ds_fp, 'rb') as f:
        kitti_val_infos = pickle.load(f)

    start_time = time.time()

    salient_inputs = []
    salient_labels = []
    for pp_res, kv_info in zip(pp_kitti_res, kitti_val_infos):
        tru_cars = filter_classes(["Car"], kv_info)
        # mod_cars = filter_classes(["Car"], pp_res)
        mod_cars = pp_res

        mod_locs = mod_cars["location"]
        mod_dims = mod_cars["dimensions"]
        mod_ry = mod_cars["rotation_y"]

        tru_locs = tru_cars["location"]
        tru_dims = tru_cars["dimensions"]
        tru_ry = tru_cars["rotation_y"]
        tru_occs = tru_cars["occluded"]

        # x_min, y_min, x_max, y_max
        mod_bbs = np.array([bounds_from_kitti_locs(l, d, ry) for l, d, ry in zip(mod_locs, mod_dims, mod_ry)])
        tru_bbs = np.array([bounds_from_kitti_locs(l, d, ry) for l, d, ry in zip(tru_locs, tru_dims, tru_ry)])

        raw_matches = hungarian_matching(mod_bbs, tru_bbs)
        filtered_matches = filter_matches(raw_matches, mod_bbs, tru_bbs, 0.01)

        for mi, ti in filtered_matches:
            # Filter out unknown occlusions
            if tru_occs[ti] == 3:
                continue

            # Ins: tru_loc_x, tru_loc_z, tru_rot_y, tru_dim_l, tru_dim_w, occlusion
            # Outs: detected, err_loc_x, err_loc_z, err_rot_y (careful with identities!)
            s_in = [tru_locs[ti][0], tru_locs[ti][2], tru_ry[ti], tru_dims[ti][0], tru_dims[ti][2], tru_occs[ti]]

            if mi is not None:
                s_label = [1, tru_locs[ti][0] - mod_locs[mi][0], tru_locs[ti][2] - mod_locs[mi][2],
                           angle_diff(tru_ry[ti], mod_ry[mi])]
            else:
                s_label = [0, -1000, -1000, -1000]

            salient_inputs.append(s_in)
            salient_labels.append(s_label)

        assert len(salient_inputs) == len(salient_labels)

        # print("Len salient inputs: ", len(salient_inputs))

    print("Total length: ", len(salient_inputs))
    print(len(salient_inputs))
    print(len(salient_labels))

    np.savetxt("data/salient_inputs.txt", salient_inputs,
               fmt=['%.3f', "%.3f", "%.3f", "%.3f", "%.3f", "%.0f"],
               header="Format: <loc_x> <loc_z> <rot_y> <dim_l> <dim_w> <occlusion>")

    np.savetxt("data/salient_labels.txt", salient_labels, fmt=['%.0f',"%.3f", "%.3f", "%.3f" ],
               header="Format: <detected> <err_loc_x> <err_loc_z> <err_rot_y>")

    # TODO: Dig up the sanity check code to see that it was done correctly?
    # TODO: Plot the difference between the "true" locations and the "actual" locations?
    # TODO: Save it all in some nice "input / label" format for a clean ML Step...
    print(f"Done. Took : {time.time() - start_time}")


if __name__ == "__main__":
    run()
