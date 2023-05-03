from _pytest.python_api import approx

from stl import *
import numpy as np


def test_and_pass():
    xs = [1.0]
    and_exp = And([GEQ0(lambda x: x), LEQ0(lambda x: x - 3)])
    result = stl_rob(and_exp, xs, 0)
    return result == 1


def test_and_fail():
    xs = [5.0]
    and_exp = And([GEQ0(lambda x: x), LEQ0(lambda x: x - 3)])
    result = stl_rob(and_exp, xs, 0)
    return result == -2


def test_and_emptiness():
    xs = np.arange(1, 6)
    and_part = And([GEQ0(lambda x: x), F(GEQ0(lambda x: x - 1), 2, np.inf)])
    s = G(and_part, 1, np.inf)
    result = stl_rob(s, xs, 0)
    assert result > 0.0


def test_or_emptiness():
    xs = np.arange(1, 6)
    or_part = Or([GEQ0(lambda x: x), F(GEQ0(lambda x: x - 1), 2, np.inf)])
    s = G(or_part, 1, np.inf)
    result = stl_rob(s, xs, 0)
    assert result > 0.0


def test_G_pass():
    xs = np.arange(1, 6)
    s = G(LEQ0(lambda x: x - 8), 0, np.inf)
    result = stl_rob(s, xs, 0)
    assert result == 3


def test_G_single_step():
    xs = [2, 3, 4]
    s = G(GEQ0(lambda x: x), 0, 0)
    result = stl_rob(s, xs, 0)
    assert result == approx(2)


def test_G_single_offset_step():
    xs = [2, 3, 4]
    s = G(GEQ0(lambda x: x), 0, 0)
    result = stl_rob(s, xs, 1)
    assert result == approx(3)


def test_G_single_offset_interval():
    xs = [2, 3, 4]
    s = G(GEQ0(lambda x: x), 1, 1)
    result = stl_rob(s, xs, 1)
    assert result == approx(4)


def test_G_fail():
    xs = np.arange(1, 6)
    s = G(GEQ0(lambda x: x - 4), 0, np.inf)
    result = stl_rob(s, xs, 0)
    assert result == -3


def test_F_G_0_pass():
    xs = np.arange(1, 6)
    s = F(G(GEQ0(lambda x: x - 3), 0, np.inf), 0, np.inf)
    result = stl_rob(s, xs, 0)
    assert result == 2


def test_F_G_0_fail():
    xs = np.arange(1, 6)
    s = F(G(GEQ0(lambda x: x - 9), 0, np.inf), 0, np.inf)
    result = stl_rob(s, xs, 0)
    assert result == -4


def test_F_G_i_pass():
    xs = np.arange(1, 6)
    s = F(G(GEQ0(lambda x: x - 4), 2, np.inf), 0, np.inf)
    result = stl_rob(s, xs, 0)
    assert result == 1


def test_F_G_i_fail():
    xs = [1, 2, 3, 4, 5, 0, 1, 2, 3]
    s = F(G(GEQ0(lambda x: x - 4), 2, np.inf), 0, np.inf)
    result = stl_rob(s, xs, 0)
    assert result == -1


def test_O_pass():
    xs = [-1, -1, 0, 0, 0, 2, -5, -100]
    s = O(GEQ0(lambda x: x), 0, np.inf)
    result = stl_rob(s, xs, len(xs) - 1)
    assert result == 2


def test_O_fail():
    xs = [-1, -1, 0, 0, 0, 2, -5, -100]
    s = O(GEQ0(lambda x: x), 0, np.inf)
    result = stl_rob(s, xs, 1)
    assert result == -1

def test_O_bounded():
    xs = [-1, 70, 0, 0, 0, 2, -5, -100]
    s = O(GEQ0(lambda x: x), 2, 4)
    result = stl_rob(s, xs, 1)
    assert result == -1


def test_H_pass():
    xs = [-1, -1, 0, 0, 0, 2, -5, -100]
    s = H(GEQ0(lambda x: x + 101), 0, np.inf)
    result = stl_rob(s, xs, len(xs) - 1)
    assert result == 1


def test_H_fail():
    xs = [-1, -1, 0, 0, 0, 2, -5, -100]
    s = H(GEQ0(lambda x: x + 6), 0, np.inf)
    result = stl_rob(s, xs, len(xs) - 1)
    assert result == -94


def test_S_pass():
    xs = [(1, 100), (2, 200), (3, 300), (4, 400), (5, 500)]
    s = S(GEQ0(lambda x: x[0] - 2), LEQ0(lambda x: x[1] - 301), 0, np.inf)
    result = stl_rob(s, xs, len(xs) - 1)
    assert result == 1


def test_S_lhs_fail():
    xs = [(1, 100), (2, 200), (3, 300), (4, 400), (5, 500)]
    s = S(GEQ0(lambda x: x[0] - 4), LEQ0(lambda x: x[1] - 301), 0, np.inf)
    result = stl_rob(s, xs, len(xs) - 1)
    assert result == -1


def test_S_rhs_fail():
    xs = [(1, 100), (2, 200), (3, 300), (4, 400), (5, 500)]
    s = S(GEQ0(lambda x: x[0]), LEQ0(lambda x: x[1] - 99), 0, np.inf)
    result = stl_rob(s, xs, len(xs) - 1)
    assert result == -1


def test_U_pass():
    xs = [(1, 100), (2, 200), (3, 300), (4, 400), (5, 500)]
    s = U(LEQ0(lambda x: x[0] - 3), GEQ0(lambda x: x[1] - 199), 0, np.inf)
    result = stl_rob(s, xs, 0)
    assert result == 1


def test_U_fail_lhs():
    xs = [(1, 100), (2, 200), (3, 300), (4, 400), (5, 500)]
    s = U(LEQ0(lambda x: x[0] - 2), GEQ0(lambda x: x[1] - 399), 0, np.inf)
    result = stl_rob(s, xs, 0)
    assert result == -2


def test_U_fail_rhs():
    xs = [(1, 100), (2, 200), (3, 300), (4, 400), (5, 500)]
    s = U(LEQ0(lambda x: x[0] - 10), GEQ0(lambda x: x[1] - 1000), 0, np.inf)
    result = stl_rob(s, xs, 0)
    assert result == -500
