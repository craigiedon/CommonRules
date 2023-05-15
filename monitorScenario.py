# Load the scenario
import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import time
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario

from stl import stl_rob, STLExp
from trafficRules import safe_dist_rule, no_unnecessary_braking_rule, keeps_speed_limit_rule, traffic_flow_rule, \
    interstate_stopping_rule, faster_than_left_rule, consider_entering_vehicles_rule, safe_dist_rule_multi
from anim_utils import animate_scenario, animate_scenario_and_monitor
import json


@dataclass
class InterstateRulesConfig:
    dt: float
    t_cut_in: float
    reaction_time: float
    acc_min: float
    max_vel: float
    slow_delta: float
    a_abrupt: float
    stop_err_bound: float
    congestion_size: int
    queue_size: int
    traffic_size: int
    congestion_vel: float
    slow_traff_vel: float
    queue_vel: float
    faster_diff_thresh: float


def gen_interstate_rules(ego_id: int, scenario: Scenario, lane_centres: List[float], lane_widths: float,
                         main_cw_cs: List[float], access_cs: List[float], irc: InterstateRulesConfig) -> Dict[
    str, STLExp]:
    ego_car = scenario.obstacle_by_id(ego_id)
    obstacles = [o for o in scenario.obstacles if o.obstacle_id != ego_id]
    # dynamic_obstacles = [o for o in scenario.dynamic_obstacles if o.obstacle_id != ego_id]

    rule_dict = {
        "rg_1": safe_dist_rule_multi(ego_car, obstacles, lane_centres, lane_widths, irc.acc_min, irc.reaction_time,
                                     int(np.round(irc.t_cut_in / irc.dt))),
        "rg_2": no_unnecessary_braking_rule(ego_car, obstacles, lane_centres, lane_widths, irc.a_abrupt, irc.acc_min,
                                            irc.reaction_time),
        "rg_3": keeps_speed_limit_rule(ego_car, irc.max_vel),
        "rg_4": traffic_flow_rule(ego_car, obstacles, lane_centres, lane_widths, irc.max_vel, irc.slow_delta),
        "ri_1": interstate_stopping_rule(ego_car, obstacles, lane_centres, lane_widths, irc.stop_err_bound),
        "ri_2": faster_than_left_rule(ego_car, obstacles, main_cw_cs, access_cs, lane_widths, irc.congestion_vel,
                                      irc.congestion_size, irc.queue_vel, irc.queue_size, irc.slow_traff_vel,
                                      irc.traffic_size,
                                      irc.faster_diff_thresh),
        "ri_5": consider_entering_vehicles_rule(ego_car, obstacles, main_cw_cs, access_cs, lane_widths)}

    # for o in obstacles:
    #     # rule_dict[f"rg_1_{o.obstacle_id}"] = safe_dist_rule(ego_car, o, lane_centres, lane_widths, irc.acc_min, irc.reaction_time, int(np.round(irc.t_cut_in / irc.dt)))
    #     rule_dict[f"ri_2_{o.obstacle_id}"] = faster_than_left_rule(ego_car, [o], main_cw_cs, access_cs, lane_widths, irc.congestion_vel,
    #                                   irc.congestion_size, irc.queue_vel, irc.queue_size, irc.slow_traff_vel,
    #                                   irc.traffic_size,
    #                                   irc.faster_diff_thresh)

    return rule_dict


def run():
    file_path = "scenarios/Complex_Solution.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    T = 60
    ego_id = 100
    state_dict = [scenario.obstacle_states_at_time_step(i) for i in range(T)]

    # Values from "Formalization of Interstate Traffic rules in Temporal Logic
    with open("config/interstate_rule_config.json", 'r') as f:
        irc = InterstateRulesConfig(**json.load(f))

    lane_centres = [-1.75, 1.75, 5.25]
    main_cw_cs = lane_centres[1:]
    access_cs = lane_centres[0:1]
    lane_widths = 3.5

    rule_dict = gen_interstate_rules(ego_id, scenario, lane_centres, lane_widths, main_cw_cs, access_cs, irc)

    start_time = time.time()
    for rule_name, rule in rule_dict.items():
        # rob_vals = [stl_rob(rule, state_dict[:i], 0) for i in range(1, len(state_dict))]
        rob_val = stl_rob(rule, state_dict, 0)
        print(f"{rule_name}: {rob_val}")
    print(f"All Functions Took: {time.time() - start_time} secs")


if __name__ == "__main__":
    run()
