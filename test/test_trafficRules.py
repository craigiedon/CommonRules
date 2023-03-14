from typing import List, Dict

import numpy as np
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType, DynamicObstacle, Obstacle
from commonroad.scenario.state import KSState, InitialState, CustomState
from pytest import approx

from stl import stl_rob
from trafficRules import front, rear, left, right, in_lane, in_same_lane, in_front_of, turning_left, single_lane, \
    cut_in, keeps_safe_distance_prec, safe_dist_rule, unnecessary_braking


def test_front():
    pos = np.array([1.0, 2.0])
    v = 0.0
    r = 0.0
    car = StaticObstacle(0, ObstacleType.CAR, Rectangle(length=4.0, width=1.5),
                         InitialState(position=pos, velocity=v, orientation=r))
    s = {0: KSState(0, position=pos, velocity=v, orientation=r)}

    f = front(car, s)
    assert f[0] == approx(3.0) and f[1] == approx(2.0)


def test_front_rot():
    pos = np.array([1.0, 2.0])
    v = 0.0
    r = np.pi / 2.0
    car = StaticObstacle(0, ObstacleType.CAR, Rectangle(length=4.0, width=1.5),
                         InitialState(position=pos, velocity=v, orientation=r))
    s = {0: KSState(0, position=pos, velocity=v, orientation=r)}

    f = front(car, s)
    assert f[0] == approx(1.0) and f[1] == approx(4.0)


def test_rear():
    pos = np.array([1.0, 2.0])
    v = 0.0
    r = 0.0
    car = StaticObstacle(0, ObstacleType.CAR, Rectangle(length=4.0, width=1.5),
                         InitialState(position=pos, velocity=v, orientation=r))
    s = {0: KSState(0, position=pos, velocity=v, orientation=r)}

    f = rear(car, s)
    assert f[0] == approx(-1.0) and f[1] == approx(2.0)


def test_left():
    pos = np.array([1.0, 2.0])
    v = 0.0
    r = 0.0
    car = StaticObstacle(0, ObstacleType.CAR, Rectangle(length=4.0, width=1.5),
                         InitialState(position=pos, velocity=v, orientation=r))
    s = {0: KSState(0, position=pos, velocity=v, orientation=r)}

    f = left(car, s)
    assert f[0] == approx(1.0) and f[1] == approx(2.75)


def test_right():
    pos = np.array([1.0, 2.0])
    v = 0.0
    r = 0.0
    car = StaticObstacle(0, ObstacleType.CAR, Rectangle(length=4.0, width=1.5),
                         InitialState(position=pos, velocity=v, orientation=r))
    s = {0: KSState(0, position=pos, velocity=v, orientation=r)}

    f = right(car, s)
    assert f[0] == approx(1.0) and f[1] == approx(1.25)


def test_in_lane_single():
    pos = np.array([1.0, 1.75])
    v = 0.0
    r = 0.0
    car = DynamicObstacle(0,
                          ObstacleType.CAR,
                          Rectangle(length=4.0, width=1.5),
                          InitialState(position=pos, velocity=v, orientation=r))
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5

    e = in_lane(car, lane_centres[0], lane_widths)
    xs: List[Dict[int, KSState]] = [
        {0: KSState(0, position=pos, velocity=v, orientation=r)}
    ]

    result = stl_rob(e, xs, 0)
    assert result == approx(1.75)


def test_in_lane_not():
    pos = np.array([1.0, 1.75])
    v = 0.0
    r = 0.0
    car = DynamicObstacle(0,
                          ObstacleType.CAR,
                          Rectangle(length=4.0, width=1.5),
                          InitialState(position=pos, velocity=v, orientation=r))
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5

    e = in_lane(car, lane_centres[1], lane_widths)
    xs: List[Dict[int, KSState]] = [
        {0: KSState(0, position=pos, velocity=v, orientation=r)}
    ]

    result = stl_rob(e, xs, 0)
    assert result == approx(-1.0)
    return


def gen_cars(num_cars: int) -> List[Obstacle]:
    cars = [DynamicObstacle(i, ObstacleType.CAR, Rectangle(length=4.0, width=1.5),
                            InitialState(position=np.array([0.0, 0.0]), velocity=0.0, orientation=0.0))
            for i in range(num_cars)]
    return cars


def test_in_same_lane_not():
    c1, c2 = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5

    e = in_same_lane(c1, c2, lane_centres, lane_widths)

    xs = [
        {0: KSState(0, position=np.array([0.0, lane_centres[0]]), orientation=0.0),
         1: KSState(0, position=np.array([0.0, lane_centres[1]]), orientation=0.0)}
    ]

    result = stl_rob(e, xs, 0)
    assert result == approx(-1.0)


def test_in_same_lane_single_same():
    c1, c2 = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5

    e = in_same_lane(c1, c2, lane_centres, lane_widths)

    xs = [
        {0: KSState(0, position=np.array([0.0, lane_centres[0]]), orientation=0.0),
         1: KSState(0, position=np.array([0.0, lane_centres[0]]), orientation=0.0)}
    ]

    result = stl_rob(e, xs, 0)
    assert result == approx(1.75)


def test_in_same_lane_straddling():
    c1, c2 = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5

    e = in_same_lane(c1, c2, lane_centres, lane_widths)

    xs = [
        {0: KSState(0, position=np.array([0.0, lane_centres[0]]), orientation=0.0),
         1: KSState(0, position=np.array([0.0, (lane_centres[0] + lane_centres[1]) / 2.0]), orientation=0.0)}
    ]

    result = stl_rob(e, xs, 0)
    assert result == approx(0.75)


def test_in_front_of_true():
    c1, c2 = gen_cars(2)
    lane_centres = [1.75, 5.25]

    e = in_front_of(c1, c2)
    xs = [
        {0: KSState(0, position=np.array([10.0, lane_centres[0]]), orientation=0.0),
         1: KSState(0, position=np.array([0.0, lane_centres[0]]), orientation=0.0)}
    ]

    result = stl_rob(e, xs, 0)
    assert result == approx(6)


def test_in_front_of_false():
    c1, c2 = gen_cars(2)
    lane_centres = [1.75, 5.25]

    e = in_front_of(c1, c2)
    xs = [
        {0: KSState(0, position=np.array([3.0, lane_centres[0]]), orientation=0.0),
         1: KSState(0, position=np.array([0.0, lane_centres[0]]), orientation=0.0)}
    ]

    result = stl_rob(e, xs, 0)
    assert result == approx(-1)


def test_turning_left_true():
    c1 = gen_cars(1)[0]
    e = turning_left(c1)
    xs = [
        {0: KSState(0, position=np.array([3.0, 1.75]), orientation=np.pi / 4.0)}
    ]

    result = stl_rob(e, xs, 0)
    assert result == approx(np.pi / 4.0)


def test_single_lane_single():
    c1 = gen_cars(1)[0]
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5

    e = single_lane(c1, lane_centres, lane_widths)
    xs = [
        {0: KSState(0, position=np.array([3.0, lane_centres[0]]), orientation=0.0)}
    ]

    result = stl_rob(e, xs, 0)
    assert result == approx(1.0)


def test_single_lane_many():
    c1 = gen_cars(1)[0]
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5

    e = single_lane(c1, lane_centres, lane_widths)
    xs = [
        {0: KSState(0, position=np.array([3.0, (lane_centres[0] + lane_centres[1]) / 2.0]), orientation=0.0)}
    ]

    result = stl_rob(e, xs, 0)
    assert result == approx(-0.75)


def test_cut_in_cutting_in():
    behind_car, cutter_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5

    e = cut_in(behind_car, cutter_car, lane_centres, lane_widths)
    xs = [
        {0: KSState(0, position=np.array([0.0, lane_centres[1]]), orientation=0.0),
         1: KSState(0, position=np.array([5.0, (lane_centres[0] + lane_centres[1]) / 2.0]), orientation=np.pi / 2.0)}
    ]
    result = stl_rob(e, xs, 0)
    assert result == approx(1.5)


def test_cut_in_turning_out():
    behind_car, cutter_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5

    e = cut_in(behind_car, cutter_car, lane_centres, lane_widths)
    xs = [
        {0: KSState(0, position=np.array([0.0, lane_centres[1]]), orientation=0.0),
         1: KSState(0, position=np.array([5.0, (lane_centres[0] + lane_centres[1]) / 2.0]), orientation=-np.pi / 2.0)}
    ]
    result = stl_rob(e, xs, 0)
    assert result == approx(-np.pi / 2.0)


def test_cut_in_different_lanes():
    behind_car, cutter_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5

    e = cut_in(behind_car, cutter_car, lane_centres, lane_widths)
    xs = [
        {0: KSState(0, position=np.array([0.0, lane_centres[1]]), orientation=0.0),
         1: KSState(0, position=np.array([5.0, lane_centres[0]]), orientation=0.0)}
    ]
    result = stl_rob(e, xs, 0)
    assert result == approx(-1.0)


def test_cut_in_same_lane_single_lane():
    behind_car, cutter_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5

    e = cut_in(behind_car, cutter_car, lane_centres, lane_widths)
    xs = [
        {0: KSState(0, position=np.array([0.0, lane_centres[1]]), orientation=0.0),
         1: KSState(0, position=np.array([5.0, lane_centres[1] - 0.001]), orientation=np.pi / 8)}
    ]
    result = stl_rob(e, xs, 0)
    assert result < 0


def test_keeps_safe_distance_true():
    behind_car, cutter_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5
    dt = 0.1

    e = keeps_safe_distance_prec(behind_car, cutter_car, acc_min=-10.5 * dt, reaction_time=0.3)

    bc_start_x = 0.0
    vels = 25.0 * dt

    xs = []
    for i in range(3):
        bc_pos = np.array([bc_start_x + vels * i, lane_centres[0]])
        cc_pos = bc_pos + np.array([10.0, 0.0])
        xs.append(
            {0: KSState(0, position=bc_pos, velocity=vels, orientation=0.0),
             1: KSState(0, position=cc_pos, velocity=vels, orientation=0.0)},
        )

    result = stl_rob(e, xs, 0)

    assert result > 0


def test_keeps_safe_distance_false():
    behind_car, cutter_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5
    dt = 0.1

    e = keeps_safe_distance_prec(behind_car, cutter_car, acc_min=-10.5 * dt, reaction_time=0.3)

    bc_start_x = 0.0
    vels = 25.0 * dt

    xs = []
    for i in range(3):
        bc_pos = np.array([bc_start_x + vels * i, lane_centres[0]])
        cc_pos = bc_pos + np.array([2.0, 0.0])
        xs.append(
            {0: KSState(0, position=bc_pos, velocity=vels, orientation=0.0),
             1: KSState(0, position=cc_pos, velocity=vels, orientation=0.0)},
        )

    result = stl_rob(e, xs, 0)

    assert result < 0


def test_safe_dist_rule_obeys_rhs():
    behind_car, cutter_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5
    dt = 0.1

    e = safe_dist_rule(behind_car, cutter_car, lane_centres, lane_widths, acc_min=-10.5 * dt, reaction_time=0.3,
                       t_cut_in=int(np.round(3.0 / dt)))

    bc_start_x = 0.0
    vels = 25.0 * dt

    xs = []
    for i in range(3):
        bc_pos = np.array([bc_start_x + vels * i, lane_centres[0]])
        cc_pos = bc_pos + np.array([10.0, 0.0])
        xs.append(
            {0: KSState(0, position=bc_pos, velocity=vels, orientation=0.0),
             1: KSState(0, position=cc_pos, velocity=vels, orientation=0.0)},
        )

    result = stl_rob(e, xs, 0)

    assert result > 0


def test_safe_dist_rule_disobeys_rhs():
    behind_car, cutter_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5
    dt = 0.1

    e = safe_dist_rule(behind_car, cutter_car, lane_centres, lane_widths, acc_min=-10.5 * dt, reaction_time=0.3,
                       t_cut_in=int(np.round(3.0 / dt)))

    bc_start_x = 0.0
    vels = 25.0 * dt

    xs = []
    for i in range(3):
        bc_pos = np.array([bc_start_x + vels * i, lane_centres[0]])
        cc_pos = bc_pos + np.array([4.5, 0.0])
        xs.append(
            {0: KSState(0, position=bc_pos, velocity=vels, orientation=0.0),
             1: KSState(0, position=cc_pos, velocity=vels, orientation=0.0)},
        )

    result = stl_rob(e, xs, 0)

    assert result < 0


def test_safe_dist_rule_not_positioned_lane():
    behind_car, cutter_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5
    dt = 0.1

    e = safe_dist_rule(behind_car, cutter_car, lane_centres, lane_widths, acc_min=-10.5 * dt, reaction_time=0.3,
                       t_cut_in=int(np.round(3.0 / dt)))

    bc_start_x = 0.0
    vels = 25.0 * dt

    xs = []
    for i in range(3):
        bc_pos = np.array([bc_start_x + vels * i, lane_centres[0]])
        cc_pos = bc_pos + np.array([4.5, 3.5])
        xs.append(
            {0: KSState(0, position=bc_pos, velocity=vels, orientation=0.0),
             1: KSState(0, position=cc_pos, velocity=vels, orientation=0.0)},
        )

    result = stl_rob(e, xs, 0)

    assert result > 0


def test_safe_dist_rule_not_positioned_front():
    behind_car, cutter_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5
    dt = 0.1

    e = safe_dist_rule(behind_car, cutter_car, lane_centres, lane_widths, acc_min=-10.5 * dt, reaction_time=0.3,
                       t_cut_in=int(np.round(3.0 / dt)))

    bc_start_x = 0.0
    vels = 25.0 * dt

    xs = []
    for i in range(3):
        bc_pos = np.array([bc_start_x + vels * i, lane_centres[0]])
        cc_pos = bc_pos + np.array([-10.0, 0.0])
        xs.append(
            {0: KSState(0, position=bc_pos, velocity=vels, orientation=0.0),
             1: KSState(0, position=cc_pos, velocity=vels, orientation=0.0)},
        )

    result = stl_rob(e, xs, 0)

    assert result > 0


def test_safe_dist_rule_recent_cut_in():
    behind_car, cutter_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5
    dt = 0.1

    e = safe_dist_rule(behind_car, cutter_car, lane_centres, lane_widths, acc_min=-10.5 * dt, reaction_time=0.3,
                       t_cut_in=int(np.round(3.0 / dt)))

    bc_start_x = 0.0
    vels = 25.0 * dt

    xs = []
    for i in range(3):
        bc_pos = np.array([bc_start_x + vels * i, lane_centres[1]])
        cc_pos = bc_pos + np.array([4.5, -1.75])
        xs.append(
            {0: KSState(0, position=bc_pos, velocity=vels, orientation=0.0),
             1: KSState(0, position=cc_pos, velocity=vels, orientation=np.pi / 4.0)},
        )

    result = stl_rob(e, xs, 0)

    assert result > 0


def test_unnecessary_braking_positive_acc():
    ego_car, other_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5
    dt = 0.1

    a_abrupt = -2 * dt
    acc_min = -10.5 * dt
    reaction_time = 0.3

    e = unnecessary_braking(ego_car, [other_car], lane_centres, lane_widths, a_abrupt, acc_min, reaction_time)

    xs = [
        {
            0: CustomState(position=[0.0, lane_centres[0]], velocity=1.0, acceleration=5.0, orientation=0.0, time_step=0),
            1: CustomState(position=[100.0, lane_centres[1]], velocity=1.0, acceleration=0.0, orientation=0.0, time_step=0),
        }
    ]

    result = stl_rob(e, xs, 0)

    assert result == approx(-5.0 + a_abrupt)


def test_unnecessary_braking_abrupt_on_clear_road():
    ego_car, other_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5
    dt = 0.1

    a_abrupt = -2 * dt
    acc_min = -10.5 * dt
    reaction_time = 0.3

    e = unnecessary_braking(ego_car, [other_car], lane_centres, lane_widths, a_abrupt, acc_min, reaction_time)

    xs = [
        {
            0: CustomState(position=[0.0, lane_centres[0]], velocity=1.0, acceleration=-1.0, orientation=0.0, time_step=0),
            1: CustomState(position=[100.0, lane_centres[1]], velocity=1.0, acceleration=0.0, orientation=0.0, time_step=0),
        }
    ]

    result = stl_rob(e, xs, 0)

    assert result == approx(1 + a_abrupt)


def test_unnecessary_braking_follower_abruptness():
    ego_car, other_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5
    dt = 0.1

    a_abrupt = -2 * dt
    acc_min = -10.5 * dt
    reaction_time = 0.3

    e = unnecessary_braking(ego_car, [other_car], lane_centres, lane_widths, a_abrupt, acc_min, reaction_time)

    xs = [
        {
            0: CustomState(position=[0.0, lane_centres[0]], velocity=1.0, acceleration=-10.3, orientation=0.0, time_step=0),
            1: CustomState(position=[10.0, lane_centres[0]], velocity=1.0, acceleration=-10.0, orientation=0.0, time_step=0),
        }
    ]

    result = stl_rob(e, xs, 0)

    assert result == approx(0.1)


def test_unnecessary_braking_clear_but_not_abrupt():
    ego_car, other_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5
    dt = 0.1

    a_abrupt = -2 * dt
    acc_min = -10.5 * dt
    reaction_time = 0.3

    e = unnecessary_braking(ego_car, [other_car], lane_centres, lane_widths, a_abrupt, acc_min, reaction_time)

    xs = [
        {
            0: CustomState(position=[0.0, lane_centres[0]], velocity=1.0, acceleration=-0.05, orientation=0.0, time_step=0),
            1: CustomState(position=[100.0, lane_centres[1]], velocity=1.0, acceleration=0.0, orientation=0.0, time_step=0),
        }
    ]

    result = stl_rob(e, xs, 0)

    assert result == approx(-0.15)


def test_unncessary_braking_follow_but_not_abrupt():
    ego_car, other_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5
    dt = 0.1

    a_abrupt = -2 * dt
    acc_min = -10.5 * dt
    reaction_time = 0.3

    e = unnecessary_braking(ego_car, [other_car], lane_centres, lane_widths, a_abrupt, acc_min, reaction_time)

    xs = [
        {
            0: CustomState(position=[0.0, lane_centres[0]], velocity=1.0, acceleration=-10.1, orientation=0.0, time_step=0),
            1: CustomState(position=[10.0, lane_centres[0]], velocity=1.0, acceleration=-10.0, orientation=0.0, time_step=0),
        }
    ]

    result = stl_rob(e, xs, 0)

    assert result == approx(-0.1)


def test_unnecessary_braking_follow_but_not_safe_distance_prec():
    ego_car, other_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5
    dt = 0.1

    a_abrupt = -2 * dt
    acc_min = -10.5 * dt
    reaction_time = 0.3

    e = unnecessary_braking(ego_car, [other_car], lane_centres, lane_widths, a_abrupt, acc_min, reaction_time)

    xs = [
        {
            0: CustomState(position=[0.0, lane_centres[0]], velocity=1.0, acceleration=-11.0, orientation=0.0, time_step=0),
            1: CustomState(position=[4.1, lane_centres[0]], velocity=1.0, acceleration=-10.0, orientation=0.0, time_step=0),
        }
    ]

    result = stl_rob(e, xs, 0)

    assert result < 0
