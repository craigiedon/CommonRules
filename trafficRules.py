import numpy as np

from stl import G, GEQ0, stl_rob, F, U


def in_same_lane() -> float:
    return 0.0


def in_front_of() -> float:
    return 0.0


def cut_in() -> float:
    return 0.0


def keeps_safe_distance_prec() -> float:
    return 0.0


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


def run():
    pass


if __name__ == "__main__":
    r_g1 = G(GEQ0(lambda x: x + 2), 0, 49)
    # rv = stl_rob(r_g1, xs, 0)

    T = 50
    xs_1 = np.ones(T) * -2
    xs_1[25:] = 3
    xs_1[48:] = -10

    test_stl_1 = F(G(GEQ0(lambda x: x), 2, 1000), 0, 10000)
    rv_1 = stl_rob(test_stl_1, xs_1, 0)

    print("rv_1:", rv_1)

    xs_2 = np.zeros((T, 2))

    xs_2[:, 0] = 7
    xs_2[:, 1] = -2

    test_stl_2 = U(GEQ0(lambda x: x[0]), GEQ0(lambda x: x[1]), 0, T-1)
    rv_2 = stl_rob(test_stl_2, xs_2, 0)

    print("rv_2: ", rv_2)

    run()
