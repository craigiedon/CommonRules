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


def test_unnestedF_unbounded():
    spec = F(GEQ0(lambda x: x), 1, np.inf)
    xs = np.array([-1, 0, 2, -5])
    lb_res, ub_res, res_map = online_run(spec, xs)

    assert np.all(lb_res == np.array([-np.inf, 0, 2, 2]))
    assert np.all(ub_res == np.array([np.inf, np.inf, np.inf, np.inf]))


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
