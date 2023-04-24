import os
import pickle
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt
from nuscenes.nuscenes import NuScenes

from KITTIPEMDataCreation import filter_matches
from hungarianMatching import hungarian_matching
from utils import angle_diff, rot_mat
from tqdm import tqdm


@dataclass
class NuAnno:
    class_name: str
    loc: np.ndarray  # x - Right, y - Forward, z - Up(?)
    dims: np.ndarray  # <Length, Width, Height>
    rot: float  # Start with the front facing along x-axis, then rotate as normal
    viz: int  # Visibility rating from 1 (0-40% visible) to 4 (80-100% visible)


@dataclass
class ModelRes:
    class_name: str
    score: float
    loc: np.ndarray  # x - Right, y - Forward, z - Up(?)
    dims: np.ndarray  # <Length, Width, Height>
    rot: float  # Start with the front facing along x-axis, then rotate as normal


def extract_relevant_info(nusc: NuScenes, nu_info: Dict) -> List[NuAnno]:
    sample = nusc.get('sample', nu_info['token'])

    num_objs = len(nu_info["gt_names"])

    gt_boxes = nu_info["gt_boxes"]
    obj_classes = nu_info["gt_names"]
    visibilities = [nusc.get('sample_annotation', a)['visibility_token'] for a in sample['anns']]

    annos = []
    for i in range(num_objs):
        gt_box = gt_boxes[i]
        assert (len(gt_box) == 9)
        annos.append(NuAnno(
            class_name=obj_classes[i],
            loc=gt_boxes[i, 0:3],
            dims=gt_boxes[i, 3:6],
            rot=gt_boxes[i, 6],
            viz=int(visibilities[i])
        ))

    # nusc.render_sample(nu_info['token'])
    # plt.show()

    return annos


def convert_to_mod_res(mod_info: Dict) -> List[ModelRes]:
    mrs = []
    for i in range(len(mod_info['name'])):
        mrs.append(ModelRes(mod_info['name'][i],
                            mod_info['score'][i],
                            mod_info['boxes_lidar'][i][0:3],
                            mod_info['boxes_lidar'][i][3:6],
                            mod_info['boxes_lidar'][i][6]))
    return mrs


def bounds_from_nusc(loc: np.ndarray, dims: np.ndarray, rot: float) -> np.ndarray:
    """
    :param loc:  3-D Array: Right, Forward, Up (xyz)
    :param dims: 3-D Array: "Length, Width, Height"
    :param rot: Radians
    :return:  x-min, y-min, x-max, y-max
    """

    td_dims = dims[[0, 1]]
    td_loc = loc[[0, 1]]

    r_mat = rot_mat(rot)

    corner_offset = 0.5 * td_dims * (r_mat @ np.array([1, 0]))

    c1 = td_loc + corner_offset
    c2 = td_loc - corner_offset

    x_min = min(c1[0], c2[0])
    y_min = min(c1[1], c2[1])
    x_max = max(c1[0], c2[0])
    y_max = max(c1[1], c2[1])

    return np.array([x_min, y_min, x_max, y_max])


# def filter_nuscenes_info_class(class_names: List[str], nu_info: Dict) -> Dict:
#     obj_classes = nu_info["gt_names"]
#     num_objs = len(obj_classes)
#     matching_ids = [i for i in range(num_objs) if obj_classes[i] in class_names]
#
#     keys_to_extract = {'gt_boxes', }
#
#     if len(matching_ids) == 0:
#         raise NotImplementedError


def run():
    # Full NuScenes Config
    data_root="/media/cinnes/Extreme Pro/OpenPCDet/data/nuscenes/v1.0-trainval"
    version = "v1.0-trainval"
    pp_fp = "/media/cinnes/Extreme Pro/OpenPCDet/output/nuscenes_models/cbgs_pp_multihead/default/eval/epoch_5823/val/default/result.pkl"

    # Mini NuScenes Config
    # data_root = "/media/cinnes/Craig Files/OpenPCDet/data/nuscenes/v1.0-mini"
    # version = "v1.0-mini"
    # pp_fp = "/media/cinnes/Craig Files/OpenPCDet/output/nuscenes_models/cbgs_pp_multihead/default/eval/epoch_5823/val/default/result.pkl"

    val_ds_fp = os.path.join(data_root, "nuscenes_infos_10sweeps_val.pkl")

    # Load the OpenDetPC KITTI PointPillar Results
    with open(pp_fp, "rb") as f:
        pp_nu_res = pickle.load(f)

    # Load the Validation Set itself!
    with open(val_ds_fp, 'rb') as f:
        nu_val_infos = pickle.load(f)

    assert len(pp_nu_res) == len(nu_val_infos)

    nusc = NuScenes(version=version, dataroot=data_root,
                    verbose=True)

    salient_inputs = []
    salient_labels = []
    for mod_info, tru_info in tqdm(zip(pp_nu_res, nu_val_infos), total=len(pp_nu_res)):
        # Filter out all the tru infos that aren't cars
        tru_labels = [l for l in extract_relevant_info(nusc, tru_info) if l.class_name=="car"]
        mod_res = convert_to_mod_res(mod_info)

        # test_sample = nusc.get('sample', tru_info['token'])
        # nusc.render_annotation(test_sample['anns'][5])
        plt.show()

        # Ignore samples with no cars in them
        if len(tru_labels) == 0:
            continue

        mod_bbs = np.array([bounds_from_nusc(x.loc, x.dims, x.rot) for x in mod_res])
        tru_bbs = np.array([bounds_from_nusc(x.loc, x.dims, x.rot) for x in tru_labels])

        raw_matches = hungarian_matching(mod_bbs, tru_bbs)
        filtered_matches = filter_matches(raw_matches, mod_bbs, tru_bbs, 0.01)

        for mi, ti in filtered_matches:
            # Ins: tru_loc_x, tru_loc_y, tru_rot, tru_dim_l, tru_dim_w, tru_dim_h, viz
            # Outs: detected, err_loc_x, err_loc_y, err_rot
            tru_label = tru_labels[ti]
            s_in = [tru_label.loc[0], tru_label.loc[1], tru_label.rot, *tru_label.dims, tru_label.viz]

            if mi is not None:
                s_label = [1,
                           tru_label.loc[0] - mod_res[mi].loc[0],
                           tru_label.loc[1] - mod_res[mi].loc[1],
                           angle_diff(tru_label.rot, mod_res[mi].rot)]
            else:
                s_label = [0, -1000, -1000, -1000]
            salient_inputs.append(s_in)
            salient_labels.append(s_label)

        # Extract the learned bounding boxes from the model

    # Ins: tru_loc_x, tru_loc_y, tru_rot, tru_dim_l, tru_dim_w, tru_dim_h, viz
    # Outs: detected, err_loc_x, err_loc_y, err_rot
    np.savetxt("data/nuscenes/salient_inputs.txt", salient_inputs,
               fmt=['%.3f', "%.3f", "%.3f", "%.3f", "%.3f", "%.3f", "%.0f"],
               header="Format: <loc_x> <loc_y> <rot> <dim_l> <dim_w> <dim_h> <visibility>")

    np.savetxt("data/nuscenes/salient_labels.txt", salient_labels, fmt=['%.0f',"%.3f", "%.3f", "%.3f" ],
               header="Format: <detected> <err_loc_x> <err_loc_y> <err_rot>")
    # val_sample = nusc.get('sample', nu_val_infos[0]['token'])
    # val_anno = nusc.get('sample_annotation', val_sample['anns'][0])

    print("Done")


if __name__ == "__main__":
    run()
