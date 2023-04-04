from typing import List, Tuple, Optional, TypeAlias

import numpy as np
import torch

from bbox_funcs import bbox_iou, bbox_iou_numpy
from ortools.linear_solver import pywraplp

Matching: TypeAlias = List[Tuple[Optional[int], Optional[int]]]


def hungarian_matching(model_detections: np.ndarray, tru_detections: np.ndarray) -> Matching:
    """

    :param model_detections: [[x_min, y_min, x_max, y_max]]
    :param tru_detections: [[x_min, y_min, x_max, y_max]]
    :return: [(model_index, tru_det_index)]

    """

    n_model_dets = len(model_detections)
    n_tru_dets = len(tru_detections)

    # No true detections --- every model detection is a false positive
    if len(tru_detections) == 0:
        return [(i, None) for i, _ in enumerate(model_detections)]

    # No model detections --- every tru detection is as a false negative
    if len(model_detections) == 0:
        return [(None, i) for i, _ in enumerate(tru_detections)]

    iou_matrix = bbox_iou_numpy(model_detections, tru_detections)
    if n_model_dets <= n_tru_dets:
        m_t_pairings: Matching = hungarian_matching_ilp(iou_matrix)

        # Add in false negatives
        tru_det_assigns = {tdi: mdi for mdi, tdi in m_t_pairings}
        m_t_pairings.extend([(None, j) for j in range(n_tru_dets) if j not in tru_det_assigns])

    else:
        t_m_pairings = hungarian_matching_ilp(iou_matrix.T)
        m_t_pairings = [(i, j) for j, i in t_m_pairings]

        # Add in false positives
        mod_det_assigns = {mdi: tdi for mdi, tdi in m_t_pairings}
        m_t_pairings.extend([(i, None) for i in range(n_model_dets) if i not in mod_det_assigns])

    return m_t_pairings


def hungarian_matching_ilp(score_matrix) -> Matching:
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Variables
    x = {}
    n_rows = len(score_matrix)
    n_cols = len(score_matrix[0])

    assert n_rows <= n_cols, f"Number of rows must be less than or equal to num cols: n_rows: {n_rows}, n_cols: {n_cols}"

    for i in range(n_rows):
        for j in range(n_cols):
            x[i, j] = solver.IntVar(0, 1, '')

    # Constraints
    # Each model detection must be assigned to exactly one true detection
    for i in range(n_rows):
        solver.Add(solver.Sum([x[i, j] for j in range(n_cols)]) == 1)

    # Each tru detection must have at most 1 model detection assigned to it
    for j in range(n_cols):
        solver.Add(solver.Sum([x[i, j] for i in range(n_rows)]) <= 1)

    # Objective
    objective_terms = []
    for i in range(n_rows):
        for j in range(n_cols):
            objective_terms.append(score_matrix[i][j] * x[i, j])
    solver.Maximize(solver.Sum(objective_terms))

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        assignments = []
        # print(f'Total cost = {solver.Objective().Value()}\n')
        for i in range(n_rows):
            for j in range(n_cols):
                # Test if x[i, j] is 1 (with tolerance for floating point arithmetic).
                if x[i, j].solution_value() >= 0.5:
                    # print(f"Model det: {i} assigned to task {j}. Score = {score_matrix[i][j]}")
                    assignments.append((i, j))
        return assignments
    raise ValueError("Feasible solution not found")


def run():
    model_detections = np.array([[1, 1, 5, 5],
                                 [300, 200, 350, 250],
                                 [750, 750, 800, 800]
                                 ])
    tru_detections = np.array([[300, 200, 360, 260],
                               [1.1, 0.8, 4.5, 4.5],
                               [755, 740, 800, 800]
                               ])
    matches = hungarian_matching(model_detections, tru_detections)
    print(matches)


if __name__ == "__main__":
    run()
