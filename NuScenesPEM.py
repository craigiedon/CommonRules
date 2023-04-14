import pickle
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt
from nuscenes.nuscenes import NuScenes


@dataclass
class NuAnno:
    class_name: str
    loc: np.ndarray  # x - Right, y - Forward, z - Up(?)
    dims: np.ndarray  # <Length, Width, Height>
    rot: float # Start with the front faceing along x-axis, then rotate as normal
    viz: int  # Visibility rating from 1 (0-40% visible) to 4 (80-100% visible)


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
            viz=visibilities[i]
        ))

    # nusc.render_sample(nu_info['token'])
    # plt.show()

    return annos


def filter_nuscenes_info_class(class_names: List[str], nu_info: Dict) -> Dict:
    obj_classes = nu_info["gt_names"]
    num_objs = len(obj_classes)
    matching_ids = [i for i in range(num_objs) if obj_classes[i] in class_names]

    keys_to_extract = {'gt_boxes', }

    if len(matching_ids) == 0:
        raise NotImplementedError


def run():
    pp_fp = "/media/cinnes/Craig Files/OpenPCDet/output/nuscenes_models/cbgs_pp_multihead/default/eval/epoch_5823/val/default/result.pkl"
    val_ds_fp = "/media/cinnes/Craig Files/OpenPCDet/data/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_val.pkl"

    # Load the OpenDetPC KITTI PointPillar Results
    with open(pp_fp, "rb") as f:
        pp_nu_res = pickle.load(f)

    # Load the KITTI Validation Set itself!
    with open(val_ds_fp, 'rb') as f:
        nu_val_infos = pickle.load(f)

    assert len(pp_nu_res) == len(nu_val_infos)

    nusc = NuScenes(version='v1.0-mini', dataroot="/media/cinnes/Craig Files/OpenPCDet/data/nuscenes/v1.0-mini",
                    verbose=True)

    for mod_res, tru_info in zip(pp_nu_res, nu_val_infos):
        # Filter out all the tru infos that aren't cars
        # tru_labels = [l for l in extract_relevant_info(nusc, tru_info) if l.class_name=="car"]
        tru_labels = [l for l in extract_relevant_info(nusc, tru_info)]

        test_sample = nusc.get('sample', tru_info['token'])
        # nusc.render_annotation(test_sample['anns'][5])
        plt.show()

        # Ignore samples with no cars in them
        if len(tru_labels) == 0:
            continue

        # TODO: Fill in the rest of the conversion parts
        tru_bbs = None

        # Extract the learned bounding boxes from the model

    val_sample = nusc.get('sample', nu_val_infos[0]['token'])
    val_anno = nusc.get('sample_annotation', val_sample['anns'][0])

    print(pp_nu_res)


if __name__ == "__main__":
    run()
