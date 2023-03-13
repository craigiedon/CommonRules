from typing import Dict, List

import commonroad.scenario.state
import numpy as np
from commonroad.scenario.obstacle import Obstacle
from commonroad.scenario.state import State, KSState

from stl import G, GEQ0, stl_rob, F, U, STLExp, LEQ0, And, Or, Neg, Implies, O, H


def in_same_lane(c1: Obstacle, c2: Obstacle, lane_centres: List[float], lane_widths: float) -> STLExp:
    return Or([And([in_lane(c1, l, lane_widths), in_lane(c2, l, lane_widths)]) for l in lane_centres])


def in_lane(car: Obstacle, lane_centre: float, lane_width: float) -> STLExp:
    def f(s: Dict[int, KSState]) -> float:
        car_ys = np.array([
            s[car.obstacle_id].position[1],
            front(car, s)[1],
            rear(car, s)[1],
            left(car, s)[1],
            right(car, s)[1]
        ])
        dist = np.min(np.abs(car_ys - lane_centre))
        return dist - lane_width / 2.0

    return LEQ0(f)


def rot_mat(r: float) -> np.ndarray:
    return np.array([
        [np.cos(r), -np.sin(r)],
        [np.sin(r), np.cos(r)]]
    )


def rear(car: Obstacle, s: Dict[int, KSState]) -> np.ndarray:
    pos = s[car.obstacle_id].position
    rot = s[car.obstacle_id].orientation
    offset = rot_mat(rot) @ np.array([car.obstacle_shape.length / 2.0, 0.0])
    return pos - offset


def front(car: Obstacle, s: Dict[int, KSState]) -> np.ndarray:
    pos = s[car.obstacle_id].position
    rot = s[car.obstacle_id].orientation
    offset = rot_mat(rot) @ np.array([car.obstacle_shape.length / 2.0, 0.0])
    return pos + offset


def left(car: Obstacle, s: Dict[int, KSState]) -> np.ndarray:
    pos = s[car.obstacle_id].position
    rot = s[car.obstacle_id].orientation
    offset = rot_mat(rot) @ np.array([0.0, car.obstacle_shape.width / 2.0])
    return pos + offset


def right(car: Obstacle, s: Dict[int, KSState]) -> np.ndarray:
    pos = s[car.obstacle_id].position
    rot = s[car.obstacle_id].orientation
    offset = rot_mat(rot) @ np.array([0.0, car.obstacle_shape.width / 2.0])
    return pos - offset


# Is Car A in front of Car B
def in_front_of(car_a: Obstacle, car_b: Obstacle) -> STLExp:
    def f(s: Dict[int, KSState]) -> float:
        a_rear_x = rear(car_a, s)[0]
        b_front_x = front(car_b, s)[0]

        return a_rear_x - b_front_x

    return GEQ0(f)


def turning_left(car: Obstacle) -> STLExp:
    def f(s: Dict[int, KSState]) -> float:
        return s[car.obstacle_id].orientation

    return GEQ0(f)


def turning_right(car: Obstacle) -> STLExp:
    def f(s: Dict[int, KSState]) -> float:
        return s[car.obstacle_id].orientation

    return LEQ0(f)


def single_lane(car: Obstacle, lane_centres: List[float], lane_widths: float) -> STLExp:
    single_lane_options = []
    for chosen_lane_i in range(len(lane_centres)):
        lane_choices = []
        for j in range(len(lane_centres)):
            if j == chosen_lane_i:
                lane_choices.append(in_lane(car, lane_centres[j], lane_widths))
            else:
                lane_choices.append(Neg(in_lane(car, lane_centres[j], lane_widths)))

        single_lane_options.append(And(lane_choices))

    return Or(single_lane_options)


def cut_in(behind_car: Obstacle, cutter_car: Obstacle, lane_centres: List[float], lane_widths: float) -> STLExp:
    def y_diff(s: Dict[int, KSState]) -> float:
        behind_y = s[behind_car.obstacle_id].position[1]
        cut_y = s[cutter_car.obstacle_id].position[1]
        return behind_y - cut_y

    return And([Neg(single_lane(cutter_car, lane_centres, lane_widths)),
                Or([
                    And([GEQ0(y_diff), turning_left(cutter_car)]),
                    And([LEQ0(y_diff), turning_right(cutter_car)])
                ]),
                in_same_lane(behind_car, cutter_car, lane_centres, lane_widths)])


def keeps_safe_distance_prec(behind_car: Obstacle, front_car: Obstacle, acc_min: float, reaction_time: float) -> STLExp:
    def f(s: Dict[int, KSState]) -> float:
        front_v = s[front_car.obstacle_id].velocity
        behind_v = s[behind_car.obstacle_id].velocity

        behind_pot = ((behind_v ** 2) / (-2 * np.abs(acc_min)))
        front_pot = ((front_v ** 2) / (-2 * np.abs(acc_min)))
        safe_dist = behind_pot - front_pot + front_v * reaction_time

        dist = rear(front_car, s)[0] - front(behind_car, s)[0]

        return dist - safe_dist

    return GEQ0(f)


def unnecessary_braking() -> float:
    return 0.0


def keeps_lane_speed_limit() -> float:
    return 0.0


def keeps_type_speed_limit() -> float:
    return 0.0


def keeps_fov_speed_limit() -> float:
    return 0.0


def keeps_braking_speed_limit() -> float:
    return 0.0


def slow_leading_vehicle() -> float:
    return 0.0


def preserves_flow() -> float:
    return 0.0


def in_congestion() -> float:
    return 0.0


def exist_standing_leading_vehicle() -> float:
    return 0.0


def in_standstill() -> float:
    return 0.0


def left_of() -> float:
    return 0.0


def drives_faster() -> float:
    return 0.0


def in_vehicle_queue():
    return


def in_slow_moving_traffic():
    return


def slightly_higher_speed():
    return


def left_of_broad_marking():
    return


def right_of_broad_marking():
    return


def on_access_ramp():
    return


def on_main_carriageway():
    return


def main_carriageway_right_lain():
    return


def safe_dist_rule(ego_car: Obstacle, other_car: Obstacle, lane_centres: List[float], lane_widths: float,
                   acc_min: float, reaction_time: float, t_cut_in: int) -> STLExp:
    positioning = And([in_same_lane(ego_car, other_car, lane_centres, lane_widths),
                       in_front_of(other_car, ego_car)])
    cutting_behaviour = Neg(O(
        And([cut_in(ego_car, other_car, lane_centres, lane_widths),
             H(Neg(cut_in(ego_car, other_car, lane_centres, lane_widths)), 1, 1)
             ]), 0, t_cut_in))
    lhs = And([positioning, cutting_behaviour])

    rhs = keeps_safe_distance_prec(ego_car, other_car, acc_min, reaction_time)
    return G(Implies(lhs, rhs), 0, 1000)


def run():
    # TODO: Stick the parameters from the paper here
    rg_1 = safe_dist_rule(ego_car, other_car, lane_centres, lane_widths, acc_min, reaction_time, t_cut_in)
    rg_2 = None
    rg_3 = None
    rg_4 = None
    rg_5 = None

    ri_1 = None
    ri_2 = None
    ri_4 = None
    ri_5 = None


if __name__ == "__main__":
    run()
