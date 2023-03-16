from functools import partial
from itertools import combinations
from typing import Dict, List, Callable

import commonroad.scenario.state
import math
import numpy as np
from commonroad.scenario.obstacle import Obstacle
from commonroad.scenario.state import State, KSState, CustomState

from stl import G, GEQ0, stl_rob, F, U, STLExp, LEQ0, And, Or, Neg, Implies, O, H, LEQc, GEQc, LEQ, GEQ, Tru


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


# Is car A to the left of car B
def left_of(car_a: Obstacle, car_b: Obstacle) -> STLExp:
    def x(f):
        return lambda s: f(s)[0]

    def y(f):
        return lambda s: f(s)[1]

    left_b = y(partial(left, car_b))
    right_a = y(partial(right, car_a))

    front_a = x(partial(front, car_a))
    front_b = x(partial(front, car_b))

    rear_a = x(partial(rear, car_a))
    rear_b = x(partial(rear, car_b))

    # Rear of car_a is in between the front & rear of car_b
    a_rear_between = And([
        LEQ(rear_b, rear_a),
        LEQ(rear_a, front_b)
    ])

    # Front of car_a is ahead of car_b, and back of car_a is behind_car_b
    bigger_car = And([
        LEQ(front_b, front_a),
        LEQ(rear_a, rear_b)
    ])

    # Front of car_a is in between the front & rear of car_b
    a_front_between = And([
        LEQ(rear_b, front_a),
        LEQ(front_a, front_b)
    ])

    return And([
        LEQ(left_b, right_a),
        Or([a_rear_between, bigger_car, a_front_between])
    ])


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


def unnecessary_braking(ego_car: Obstacle, other_cars: List[Obstacle], lane_centres: List[float], lane_widths: float,
                        a_abrupt: float, acc_min: float, reaction_time: float) -> STLExp:
    def ego_acc(s: Dict[int, CustomState]) -> float:
        return s[ego_car.obstacle_id].acceleration

    # lhs = LEQ0(ego_acc)

    front_lane_checks = []
    for other_car in other_cars:
        front_lane_checks.append(
            And([
                in_front_of(other_car, ego_car),
                in_same_lane(ego_car, other_car, lane_centres, lane_widths),
            ])
        )
    nothing_up_front = Neg(Or(front_lane_checks))

    def abrupt_difference(behind_car: Obstacle, front_car: Obstacle) -> STLExp:
        def acc_diff(s: Dict[int, CustomState]) -> float:
            b_acc = s[behind_car.obstacle_id].acceleration
            f_acc = s[front_car.obstacle_id].acceleration
            diff = b_acc - f_acc
            return diff

        return LEQc(acc_diff, a_abrupt)

    follow_abruptness = []
    for other_car in other_cars:
        follow_abruptness.append(And([
            keeps_safe_distance_prec(ego_car, other_car, acc_min, reaction_time),
            in_front_of(other_car, ego_car),
            in_same_lane(ego_car, other_car, lane_centres, lane_widths),
            abrupt_difference(ego_car, other_car)
        ]))

    abruptness = Or([
        And([nothing_up_front, LEQc(ego_acc, a_abrupt)]),
        Or(follow_abruptness)
    ])

    return And([LEQ0(ego_acc), abruptness])


# def keeps_lane_speed_limit() -> float:
#     return 0.0
#
#
# def keeps_type_speed_limit() -> float:
#     return 0.0
#
#
# def keeps_fov_speed_limit() -> float:
#     return 0.0
#
#
# def keeps_braking_speed_limit() -> float:
#     return 0.0

def car_v(c: Obstacle) -> Callable:
    def f(s: Dict[int, KSState]) -> float:
        return s[c.obstacle_id].velocity

    return f


def slow_leading_vehicle(ego_car: Obstacle, other_cars: List[Obstacle], lane_centres: List[float], lane_widths: float,
                         speed_limit: float, slow_delta: float) -> STLExp:
    slow_lvs = []
    for other_car in other_cars:
        slow_lvs.append(And([
            LEQc(car_v(other_car), speed_limit - slow_delta),
            in_same_lane(ego_car, other_car, lane_centres, lane_widths),
            in_front_of(other_car, ego_car)
        ]))
    return Or(slow_lvs)


def preserves_flow(car: Obstacle, speed_limit: float, slow_delta: float) -> STLExp:
    return GEQc(car_v(car), speed_limit - slow_delta)


def in_congestion(ego_car: Obstacle, other_cars: List[Obstacle], lane_centres: List[float], lane_widths: float,
                  congestion_vel: float, congestion_size: int) -> STLExp:
    assert congestion_size > 0

    # Congestion is impossible if there are less total cars than congestion threshold
    if congestion_size > len(other_cars):
        return Neg(Tru())

    con_contrib = []
    for other_car in other_cars:
        con_contrib.append(And([
            in_same_lane(ego_car, other_car, lane_centres, lane_widths),
            in_front_of(other_car, ego_car),
            LEQc(car_v(other_car), congestion_vel)
        ]))

    congestion_combs = list(combinations(con_contrib, congestion_size))
    assert len(congestion_combs) == math.comb(len(other_cars), congestion_size)

    return Or([And(list(cong_comb)) for cong_comb in congestion_combs])


def in_slow_moving_traffic(ego_car: Obstacle, other_cars: List[Obstacle], lane_centres: List[float], lane_widths: float,
                           slow_traff_vel: float, traffic_size: int) -> STLExp:
    return in_congestion(ego_car, other_cars, lane_centres, lane_widths, slow_traff_vel, traffic_size)


def in_vehicle_queue(ego_car: Obstacle, other_cars: List[Obstacle], lane_centres: List[float], lane_widths: float,
                     queue_vel: float, queue_size: int) -> STLExp:
    return in_congestion(ego_car, other_cars, lane_centres, lane_widths, queue_vel, queue_size)


def exist_standing_leading_vehicle(ego_car: Obstacle, other_cars: List[Obstacle], lane_centres: List[float],
                                   lane_widths: float, error_bound: float) -> STLExp:
    standing_leads = []
    for other_car in other_cars:
        standing_leads.append(And([
            in_same_lane(ego_car, other_car, lane_centres, lane_widths),
            in_front_of(other_car, ego_car),
            in_standstill(other_car, error_bound)
        ]))
    return Or(standing_leads)


def in_standstill(car: Obstacle, error_bound: float) -> STLExp:
    def f(s: Dict[int, KSState]) -> float:
        return abs(s[car.obstacle_id].velocity)

    return LEQc(f, error_bound)


def drives_faster(car_a: Obstacle, car_b: Obstacle) -> STLExp:
    return GEQ(car_v(car_a), car_v(car_b))


def slightly_higher_speed(car_a: Obstacle, car_b: Obstacle, diff_thresh: float):
    def f(s: Dict[int, KSState]) -> float:
        return s[car_a.obstacle_id].velocity - s[car_b.obstacle_id].velocity

    return And([
        GEQ0(f),
        LEQc(f, diff_thresh)
    ])


# def left_of_broad_marking():
#     return
#
#
# def right_of_broad_marking():
#     return


def on_access_ramp(car: Obstacle, access_cs: List[float], lane_widths: float) -> STLExp:
    return on_main_carriageway(car, access_cs, lane_widths)


def on_main_carriageway(car: Obstacle, main_cw_cs: List[float], lane_widths: float) -> STLExp:
    return Or([in_lane(car, l, lane_widths) for l in main_cw_cs])


# def main_carriageway_right_lane():
#     return


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


def no_unnecessary_braking_rule(ego_car: Obstacle, other_cars: List[Obstacle], lane_centres: List[float],
                                lane_widths: float,
                                a_abrupt: float, acc_min: float, reaction_time: float) -> STLExp:
    return G(Neg(unnecessary_braking(ego_car, other_cars, lane_centres, lane_widths, a_abrupt, acc_min, reaction_time)),
             0, 1000)


def keeps_speed_limit_rule(ego_car: Obstacle, max_vel: float) -> STLExp:
    def f(s: Dict[int, KSState]) -> float:
        return s[ego_car.obstacle_id].velocity

    return G(LEQc(f, max_vel), 0, 1000)


def traffic_flow_rule(ego_car: Obstacle, other_cars: List[Obstacle], lane_centres, lane_widths, speed_limit: float,
                      slow_delta: float) -> STLExp:
    return G(Implies(Neg(slow_leading_vehicle(ego_car, other_cars, lane_centres, lane_widths, speed_limit, slow_delta)),
                     preserves_flow(ego_car, speed_limit, slow_delta)), 0, 1000)


def interstate_stopping_rule(ego_car: Obstacle, other_cars: List[Obstacle], lane_centres, lane_widths, error_bound):
    return G(Implies(
        Neg(exist_standing_leading_vehicle(ego_car, other_cars, lane_centres, lane_widths, error_bound)),
        Neg(in_standstill(ego_car, error_bound))), 0, 1000)


def faster_than_left_rule(ego_car: Obstacle, other_cars, main_cw_cs: List[float], access_cs: List[float],
                          lane_widths: float, congestion_vel: float, congestion_size: int, queue_vel: float,
                          queue_size: int, slow_traff_vel: float, traffic_size: int, diff_thresh: float) -> STLExp:
    lane_centres = [*main_cw_cs, *access_cs]
    faster_left_cars = []
    for other_car in other_cars:
        lhs = And([left_of(other_car, ego_car), drives_faster(ego_car, other_car)])
        remaining_cars = [o for o in other_cars if o != other_car]
        blocked = Or([
            in_congestion(other_car, remaining_cars, lane_centres, lane_widths, congestion_vel, congestion_size),
            in_vehicle_queue(other_car, remaining_cars, lane_centres, lane_widths, queue_vel, queue_size),
            in_slow_moving_traffic(other_car, remaining_cars, lane_centres, lane_widths, slow_traff_vel, traffic_size)
        ])
        faster_than_blocked = And([blocked, slightly_higher_speed(ego_car, other_car, diff_thresh)])

        access_ramp_exempt = And([
            on_access_ramp(ego_car, access_cs, lane_widths),
            on_main_carriageway(other_car, main_cw_cs, lane_widths),
            Neg(blocked)
        ])
        faster_left_cars.append(Implies(lhs, Or([faster_than_blocked, access_ramp_exempt])))

    return G(And(faster_left_cars), 0, 1000)


# def consider_entering_vehicles_rule_old(ego_car: Obstacle, other_cars: List[Obstacle], main_cw_cs: List[float],
#                                     access_cs: List[float], lane_widths: float):
#     access_considerations = []
#     for other_car in other_cars:
#         lhs = And([
#             on_main_carriageway(ego_car, main_cw_cs, lane_widths),
#             in_front_of(other_car, ego_car),
#             on_access_ramp(other_car, access_cs, lane_widths),
#             F(on_main_carriageway(other_car, main_cw_cs, lane_widths), 0, 1000)])
#
#         rhs = Neg(And([
#             Neg(in_lane(ego_car, min(main_cw_cs), lane_widths)),
#             F(in_lane(ego_car, min(main_cw_cs), lane_widths), 1, 1000)
#         ]))
#
#         access_considerations.append(Implies(lhs, rhs))
#     return G(And(access_considerations), 0, 1000)


def consider_entering_vehicles_rule(ego_car: Obstacle, other_cars: List[Obstacle], main_cw_cs: List[float],
                                    access_cs: List[float], lane_widths: float):
    access_considerations = []
    rh_lane = min(main_cw_cs)
    for other_car in other_cars:
        lhs = And([
            on_main_carriageway(ego_car, main_cw_cs, lane_widths),
            Neg(in_lane(ego_car, rh_lane, lane_widths)),
            in_front_of(other_car, ego_car),
            on_access_ramp(other_car, access_cs, lane_widths),
            F(And([on_main_carriageway(other_car, main_cw_cs, lane_widths),
                    Neg(on_access_ramp(other_car, access_cs, lane_widths))]), 0, 1000)
            ])

        rhs = U(Neg(in_lane(ego_car, rh_lane, lane_widths)),
                And([
                    on_main_carriageway(other_car, main_cw_cs, lane_widths),
                    Neg(on_access_ramp(other_car, access_cs, lane_widths))
                ]), 0, 1000)

        access_considerations.append(Implies(lhs, rhs))
    return G(And(access_considerations), 0, 1000)


def run():
    pass


if __name__ == "__main__":
    run()
