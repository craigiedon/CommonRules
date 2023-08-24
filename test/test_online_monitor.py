# TODO: Test on simple example (no timings)
from stl import *
import numpy as np
from onlineMonitor import online_run


def test_notimings():
    spec = GEQ0(lambda x: x)
    xs = np.array([-1, 0, 2, -5])
    lb_res, ub_res, res_map = online_run(spec, xs)

    assert np.all(lb_res == np.array([-1, -1, -1, -1]))
    assert np.all(ub_res == np.array([-1, -1, -1, -1]))


def test_or():
    xs = [(1, -1), (-2, 5), (3, -4), (-1, 5)]
    or_spec = Or((GEQ0(lambda x: x[0]), GEQ0(lambda x: x[1])))
    spec = G(or_spec, 0, 3)
    lb_res, ub_res, final_res_map = online_run(spec, xs)

    assert np.all(final_res_map[or_spec].lbs == [1, 5, 3, 5])
    assert np.all(final_res_map[or_spec].ubs == [1, 5, 3, 5])
    assert np.all(final_res_map[spec].lbs == [1, -np.inf, -np.inf, -np.inf])
    assert np.all(final_res_map[spec].ubs == [1, 3, 3, 5])
    print(final_res_map)


def test_unnestedG():
    spec = G(GEQ0(lambda x: x), 0, 3)
    xs = np.array([-1, 0, 2, -5])
    lb_res, ub_res, res_map = online_run(spec, xs)

    assert np.all(lb_res == np.array([-np.inf, -np.inf, -np.inf, -5]))
    assert np.all(ub_res == np.array([-1, -1, -1, -5]))


def test_unnestedF():
    spec = F(GEQ0(lambda x: x), 0, 3)
    xs = np.array([-1, 0, 2, -5])
    lb_res, ub_res, res_map = online_run(spec, xs)

    assert np.all(lb_res == np.array([-1, 0, 2, 2]))
    assert np.all(ub_res == np.array([np.inf, np.inf, np.inf, 2]))


def test_g_nonzero_start_inf_end():
    spec = G(GEQ0(lambda x: x), 2, np.inf)
    xs = np.array([-1, 0, 2, -5])
    lb_res, ub_res, res_map = online_run(spec, xs)

    assert np.all(lb_res == [-np.inf, -np.inf, 2, -5])
    assert np.all(ub_res == [np.inf, np.inf, 2, -5])


# TODO: Test 0, inf outer, finite nested offest
# TODO: Test 0, inf outer, infinite nested offset
# TODO: Test offset inf outer, offset infinite nested
# TODO: Test 0, inf outer, offset "1-width" nested
# TODO: Test until hasn't broken
# TODO: Alter the time stamps to subtract on the temporal stuff

def test_unnestedF_unbounded():
    spec = F(GEQ0(lambda x: x), 1, np.inf)
    xs = np.array([-1, 0, 2, -5])
    lb_res, ub_res, res_map = online_run(spec, xs)

    assert np.all(lb_res == np.array([-np.inf, 0, 2, 2]))
    assert np.all(ub_res == np.array([np.inf, np.inf, np.inf, np.inf]))


def test_nested_timings():
    comp_spec = GEQ0(lambda x: x)
    eventually_spec = F(comp_spec, 1, 2)
    spec = G(eventually_spec, 0, 3)
    xs = [-5, 1, 2, 3, 10, 20]
    lb_res, ub_res, res_map = online_run(spec, xs)

    assert len(res_map[comp_spec].ts) == 5
    assert len(res_map[eventually_spec].ts) == 5
    assert len(res_map[spec].ts) == 5

    assert np.all(res_map[comp_spec].ts == np.array([1, 2, 3, 4, 5]))
    assert np.all(res_map[eventually_spec].ts == np.array([0, 1, 2, 3, 4]))


def test_or_uneven_timings():
    always_spec = G(GEQ0(lambda x: x[0]), 0, np.inf)
    eventually_spec = F(GEQ0(lambda x: x[1]), 1, 3)
    full_spec = Or(always_spec, eventually_spec)

    raise NotImplementedError


def test_and_uneven_timings():
    always_spec = G(GEQ0(lambda x: x[0]), 0, np.inf)
    eventually_spec = F(GEQ0(lambda x: x[1]), 1, 3)
    full_spec = And(always_spec, eventually_spec)

    raise NotImplementedError


def test_once():
    spec = O(GEQ0(lambda x: x), 0, 3)
    xs = np.array([-1, 0, 2, -5])
    lb_res, ub_res, res_map = online_run(spec, xs)
    assert np.all(lb_res == np.array([-1, 0, 2, 2]))
    assert np.all(ub_res == np.array([np.inf, np.inf, np.inf, 2]))


def test_history():
    spec = H(GEQ0(lambda x: x), 0, 4)
    xs = np.array([-1, 0, 2, -5])

    return


def test_until():
    return


def test_since():
    return


def test_offline_identical():
    return

# TODO: Test that the values for above are identical to the batch counterpart at each stage
# TODO: Test for all 5 of the traffic rules in the traffic STL (+ identical to offline version?)
