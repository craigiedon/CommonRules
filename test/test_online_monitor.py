# TODO: Test on simple example (no timings)
from stl import *
import numpy as np
from onlineMonitor import online_run, online_max_lemire


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


def test_offline_identical():
    return

# TODO: Test that the values for above are identical to the batch counterpart at each stage
# TODO: Test for all 5 of the traffic rules in the traffic STL (+ identical to offline version?)
