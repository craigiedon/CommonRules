# TODO: Test on simple example (no timings)
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import Obstacle, ObstacleType, DynamicObstacle
from commonroad.scenario.state import KSState, InitialState, CustomState

from stl import *
import numpy as np
from onlineMonitor import online_run, online_max_lemire

from trafficRules import keeps_safe_distance_prec, safe_dist_rule, unnecessary_braking, interstate_stopping_rule, \
    faster_than_left_rule, consider_entering_vehicles_rule


def gen_cars(num_cars: int) -> List[Obstacle]:
    cars = [DynamicObstacle(i, ObstacleType.CAR, Rectangle(length=4.0, width=1.5),
                            InitialState(position=np.array([0.0, 0.0]), velocity=0.0, orientation=0.0))
            for i in range(num_cars)]
    return cars


def test_notimings():
    spec = GEQ0(lambda x: x)
    xs = np.array([-1, 0, 2, -5])
    v_res, res_map = online_run(spec, xs)

    assert np.all(v_res == np.array([-1, -1, -1, -1]))


def test_or():
    xs = [(1, -1), (-2, 5), (3, -4), (-1, 5)]
    or_spec = Or((GEQ0(lambda x: x[0]), GEQ0(lambda x: x[1])))
    spec = G(or_spec, 0, 3)
    v_res, final_res_map = online_run(spec, xs)

    assert np.all(v_res == [1, 1, 1, 1])
    assert np.all(v_res == [1, 1, 1, 1])

    assert np.all(final_res_map[or_spec].vs == [1, 5, 3, 5])
    assert np.all(final_res_map[spec].vs == [1, 3, 3, 5])

    print(final_res_map)


def test_unnestedG():
    spec = G(GEQ0(lambda x: x), 0, 3)
    xs = np.array([-1, 0, 2, -5])
    v_res, res_map = online_run(spec, xs)

    assert np.all(v_res == np.array([-1, -1, -1, -5]))


def test_online_max_lemire_one_item():
    xs = [5.0]
    ts = [0]
    width = 1
    fill_v = np.inf
    result = online_max_lemire(xs, ts, width, fill_v)
    assert np.all(result == np.array([5.0]))


def test_online_max_lemire_two_item_one_width():
    xs = [5.0, 10.0]
    ts = [0, 1]
    width = 1
    fill_v = np.inf
    result = online_max_lemire(xs, ts, width, fill_v)
    assert np.all(result == np.array([5.0, 10.0]))


def test_online_max_lemire_two_item_two_width():
    xs = [5.0, 10.0]
    ts = [0, 1]
    width = 2
    fill_v = np.inf
    result = online_max_lemire(xs, ts, width, fill_v)
    assert np.all(result == np.array([10.0, 10.0]))


def test_unnestedF():
    spec = F(GEQ0(lambda x: x), 0, 3)
    xs = np.array([-1, 0, 2, -5])
    v_res, res_map = online_run(spec, xs)

    assert np.all(v_res == np.array([-1, 0, 2, 2]))


def test_g_nonzero_start_inf_end():
    spec = G(GEQ0(lambda x: x), 2, np.inf)
    xs = np.array([-1, 0, 2, -5])
    v_res, res_map = online_run(spec, xs)

    assert np.all(v_res == [-np.inf, -np.inf, 2, -5])


# TODO: Test 0, inf outer, finite nested offest
# TODO: Test 0, inf outer, infinite nested offset
# TODO: Test offset inf outer, offset infinite nested
# TODO: Test 0, inf outer, offset "1-width" nested
# TODO: Test until hasn't broken
# TODO: Alter the time stamps to subtract on the temporal stuff

def test_unnestedF_unbounded():
    spec = F(GEQ0(lambda x: x), 1, np.inf)
    xs = np.array([-1, 0, 2, -5])
    v_res, res_map = online_run(spec, xs)

    assert np.all(v_res == np.array([-np.inf, 0, 2, 2]))


def test_nested_timings():
    comp_spec = GEQ0(lambda x: x)
    eventually_spec = F(comp_spec, 1, 2)
    spec = G(eventually_spec, 0, 3)
    xs = [-5, 1, 2, 3, 10, 20]
    v_res, res_map = online_run(spec, xs)

    assert len(res_map[comp_spec].ts) == 5
    assert len(res_map[eventually_spec].ts) == 5
    assert len(res_map[spec].ts) == 5

    assert np.all(res_map[comp_spec].ts == np.array([1, 2, 3, 4, 5]))
    assert np.all(res_map[eventually_spec].ts == np.array([0, 1, 2, 3, 4]))


def test_or_uneven_timings():
    always_spec = G(GEQ0(lambda x: x[0]), 0, np.inf)
    eventually_spec = F(GEQ0(lambda x: x[1]), 1, 3)
    full_spec = Or((always_spec, eventually_spec))

    xs = [(1, 500), (-0.5, -2), (-10, -1), (-100, 5)]

    v_res, res_map = online_run(full_spec, xs)
    assert np.all(v_res == np.array([1, -0.5, -1, 5]))


def test_and_uneven_timings():
    always_spec = G(GEQ0(lambda x: x[0]), 0, np.inf)
    eventually_spec = F(GEQ0(lambda x: x[1]), 1, 3)
    full_spec = And((always_spec, eventually_spec))

    xs = [(1, 500), (-0.5, -2), (-10, -1), (-100, 5)]

    v_res, res_map = online_run(full_spec, xs)
    assert np.all(v_res == np.array([1, -2, -10, -100]))


def test_once_no_nesting_no_offset():
    spec = O(GEQ0(lambda x: x), 0, 3)
    xs = np.array([-1, 0, 2, -5])
    v_res, res_map = online_run(spec, xs)
    assert np.all(v_res == np.array([-1, -1, -1, -1]))


def test_once_no_nesting_offset():
    spec = O(GEQ0(lambda x: x), 1, 3)
    xs = np.array([-1, 0, 2, -5])
    v_res, res_map = online_run(spec, xs)
    assert np.all(v_res == np.array([-np.inf, -np.inf, -np.inf, -np.inf]))


def test_once_nesting_no_offset():
    spec = G(O(GEQ0(lambda x: x), 0, 3), 0, np.inf)
    xs = np.array([5, 0, 0, 0, -3])
    v_res, res_map = online_run(spec, xs)
    assert np.all(v_res == np.array([5, 5, 5, 5, 0]))


def test_once_nesting_offset():
    spec = G(O(GEQ0(lambda x: x), 1, 3), 0, np.inf)
    xs = np.array([5, 0, 0, 0, -3])
    v_res, res_map = online_run(spec, xs)
    assert np.all(v_res == np.array([5, 5, 5, 0, 0]))


def test_until_passesEventually():
    spec = U(GEQ0(lambda x: x[0]), GEQ0(lambda x: x[1]), 0, 3)
    xs = np.array([[5, -5], [3, -6], [2, 10], [-100, -8]])
    v_res, res_map = online_run(spec, xs)

    assert np.all(v_res == np.array([-5, -5, 2, 2]))


def test_until_left_violationEventually():
    spec = U(GEQ0(lambda x: x[0]), GEQ0(lambda x: x[1]), 0, 3)
    xs = np.array([[5, -5], [3, -6], [-1, -10], [2, 10]])
    v_res, res_map = online_run(spec, xs)

    assert np.all(v_res == np.array([-5, -5, -5, -1]))


def test_until_offset_oob():
    spec = U(GEQ0(lambda x: x[0]), GEQ0(lambda x: x[1]), 1, np.inf)
    xs = np.array([[5, -5], [3, -6], [2, 10], [-100, -8]])
    v_res, res_map = online_run(spec, xs)

    assert np.all(v_res == np.array([-np.inf, -6, 2, 2]))


def test_until_right_violation():
    spec = U(GEQ0(lambda x: x[0]), GEQ0(lambda x: x[1]), 0, 1)
    xs = np.array([[5, -5], [3, -6], [2, 10], [-100, -8]])
    v_res, res_map = online_run(spec, xs)

    assert np.all(v_res == np.array([-5, -5, -5, -5]))


def test_on_off_safe_dist_fail():
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

    online_res_hist, online_map = online_run(e, xs)

    offline_res_hist = np.array([stl_rob(e, xs[:i + 1], 0) for i in range(len(xs))])

    assert offline_res_hist[-1] < 0
    assert np.all(offline_res_hist == online_res_hist)


def test_on_off_safe_dist_pass():
    behind_car, cutter_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5
    dt = 0.1
    t_cut_in = int(np.round(3.0 / dt))

    e = safe_dist_rule(behind_car, cutter_car, lane_centres, lane_widths, acc_min=-10.5 * dt, reaction_time=0.3,
                       t_cut_in=t_cut_in)

    bc_start_x = 0.0
    vels = 25.0 * dt

    bc_ps = np.array([[bc_start_x + vels * i, lane_centres[1]] for i in range(3)])
    cc_ps = bc_ps + np.array([[4.5, -5.75],
                              [4.5, -1.75],
                              [4.5, 0]])

    xs = [
        {0: KSState(i, position=bc_pos, velocity=vels, orientation=0.0),
         1: KSState(i, position=cc_pos, velocity=vels, orientation=np.pi / 4.0)}
        for i, (bc_pos, cc_pos) in enumerate(zip(bc_ps, cc_ps))
    ]

    offline_res_hist = np.array([stl_rob(e, xs[:i + 1], 0) for i in range(len(xs))])
    online_res, online_map = online_run(e, xs)

    assert offline_res_hist[-1] > 0
    assert np.all(offline_res_hist == online_res)


def test_on_off_no_unnecessary_braking():
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
            0: CustomState(position=[0.0, lane_centres[0]], velocity=1.0, acceleration=5.0, orientation=0.0,
                           time_step=0),
            1: CustomState(position=[100.0, lane_centres[1]], velocity=1.0, acceleration=0.0, orientation=0.0,
                           time_step=0),
        }
    ]

    offline_res_hist = np.array([stl_rob(e, xs[:i + 1], 0) for i in range(len(xs))])
    online_res, online_map = online_run(e, xs)

    assert np.all(offline_res_hist == online_res)


def test_on_off_interstate_stopping():
    ego_car, other_car = gen_cars(2)
    lane_centres = [1.75, 5.25]
    lane_widths = 3.5
    dt = 0.1
    v_err = 0.01 * dt

    e = interstate_stopping_rule(ego_car, [other_car], lane_centres, lane_widths, v_err)

    xs = [{
        0: CustomState(position=[0.0, lane_centres[0]], velocity=10.0, orientation=0.0, time_step=0),
        1: CustomState(position=[15.0, lane_centres[0]], velocity=10.0, orientation=0.0, time_step=0)}]

    offline_res_hist = np.array([stl_rob(e, xs[:i + 1], 0) for i in range(len(xs))])
    online_res, online_map = online_run(e, xs)

    assert np.all(offline_res_hist == online_res)


def test_on_off_faster_than_left():
    ego_car, *other_cars = gen_cars(3)

    lane_centres = [-1.75, 1.75, 5.25]
    lane_widths = 3.5
    dt = 0.1

    congestion_vel = 2.78 * dt
    slow_traff_vel = 8.33 * dt
    queue_vel = 16.67 * dt

    diff_thresh = 1.0 * dt

    congestion_size = queue_size = traffic_size = 1

    e = faster_than_left_rule(ego_car, other_cars, lane_centres[1:],
                              lane_centres[:1], lane_widths, congestion_vel, congestion_size,
                              queue_vel, queue_size,
                              slow_traff_vel,
                              traffic_size,
                              diff_thresh)

    xs = [{
        0: CustomState(position=[0.0, lane_centres[0]], velocity=1.0, orientation=0.0, time_step=0),
        1: CustomState(position=[0.0, lane_centres[1]], velocity=1.5, orientation=0.0, time_step=0),
        2: CustomState(position=[10.0, lane_centres[1]], velocity=1.5, orientation=0.0, time_step=0)}]

    offline_res_hist = np.array([stl_rob(e, xs[:i + 1], 0) for i in range(len(xs))])
    online_res, online_map = online_run(e, xs)

    assert np.all(offline_res_hist == online_res)


def test_on_off_consider_entering():
    ego_car, other_car = gen_cars(2)

    lane_centres = [-1.75, 1.75, 5.25]
    lane_widths = 3.5

    e = consider_entering_vehicles_rule(ego_car, [other_car], lane_centres[1:], lane_centres[:1], lane_widths)

    xs = [
        {0: CustomState(position=[10.0, lane_centres[2]], velocity=10.0, orientation=0.0, time_step=0),
         1: CustomState(position=[0.0, lane_centres[0]], velocity=10.0, orientation=0.0, time_step=0)},

        {0: CustomState(position=[10.0, lane_centres[2]], velocity=10.0, orientation=0.0, time_step=0),
         1: CustomState(position=[0.0, lane_centres[0]], velocity=10.0, orientation=0.0, time_step=0)}
    ]

    offline_res_hist = np.array([stl_rob(e, xs[:i + 1], 0) for i in range(len(xs))])
    online_res, online_map = online_run(e, xs)

    assert np.all(offline_res_hist == online_res)

