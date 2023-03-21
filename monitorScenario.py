# Load the scenario
import json
from typing import Dict

import numpy as np
import time
from commonroad.common.file_reader import CommonRoadFileReader

from stl import stl_rob, STLExp
from trafficRules import safe_dist_rule, no_unnecessary_braking_rule, keeps_speed_limit_rule, traffic_flow_rule, \
    interstate_stopping_rule, faster_than_left_rule, consider_entering_vehicles_rule
from utils import animate_scenario, animate_scenario_and_monitor

file_path = "scenarios/Complex_Solution.xml"
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

T = 60
ego_id = 100
dt = 0.1
state_dict = [scenario.obstacle_states_at_time_step(i) for i in range(T)]
ego_car = scenario.obstacle_by_id(ego_id)
obstacles = [s for s in scenario.obstacles if s.obstacle_id != ego_id]

# Values from "Formalization of Interstate Traffic rules in Temporal Logic
t_cut_in = int(np.round(3.0 / dt))
reaction_time = 0.3
acc_min = -10.5 * dt
lane_centres = [-1.75, 1.75, 5.25]
main_cw_cs = lane_centres[1:]
access_cs = lane_centres[:1]
lane_widths = 3.5

max_vel = 36.6
slow_delta = 15.0
a_abrupt = -2.0
stop_err_bound = 0.01


congestion_size = queue_size = traffic_size = 3
congestion_vel = 2.78
slow_traff_vel = 8.33
queue_vel = 16.67

faster_diff_thresh = 5.55


# TODO: Redo this as a combined rule?
rg_1s = [safe_dist_rule(ego_car, other_car, lane_centres, lane_widths, acc_min, reaction_time, t_cut_in) for other_car in obstacles]

rule_dict: Dict[str, STLExp] = {
    "rg_2": no_unnecessary_braking_rule(ego_car, obstacles, lane_centres, lane_widths, a_abrupt, acc_min, reaction_time),
    "rg_3": keeps_speed_limit_rule(ego_car, max_vel),
    "rg_4": traffic_flow_rule(ego_car, obstacles, lane_centres, lane_widths, max_vel, slow_delta) ,
    "ri_1": interstate_stopping_rule(ego_car, obstacles, lane_centres, lane_widths, stop_err_bound),
    "ri_2": faster_than_left_rule(ego_car, obstacles, main_cw_cs, access_cs, lane_widths, congestion_vel, congestion_size, queue_vel, queue_size, slow_traff_vel, traffic_size, faster_diff_thresh),
    "ri_5": consider_entering_vehicles_rule(ego_car, obstacles, main_cw_cs, access_cs, lane_widths),
}


for rule_name, rule in rule_dict.items():
    start_time = time.time()
    rob_vals = [stl_rob(rule, state_dict[:i], 0) for i in range(1, len(state_dict))]
    print(f"{rule_name}: {rob_vals[-1]}")
    print(f"Function took: {time.time() - start_time} secs")

    with open(f"results/{rule_name}.json", "w") as f:
        json.dump(rob_vals, f)


# ob_rob_vs = {}
# for r_i, rule in enumerate(dist_rules):
#     rob_vals = [stl_rob(rule, state_dict[:i], 0) for i in range(1, len(state_dict))]
#     # rob_vals = [stl_rob(rule, state_dict[:i], 0) for i in range(1, 10)]
#     ob_rob_vs[obstacles[r_i].obstacle_id] = rob_vals
#     # rob_vals = stl_rob(rule, state_dict, 0)
#     print(f"Obstacle: {obstacles[r_i].obstacle_id}, r: {rob_vals}")

# Save ob_rob_vs to a json file
# with open("results/rg_1_results.json", "w") as f:
#     json.dump(ob_rob_vs, f)

# with open("results/rg_1_results.json", "r") as f:
#     ob_rob_vs = json.load(f)

# animate_scenario_and_monitor(scenario, ob_rob_vs)
# animate_scenario(scenario, planning_problem_set, T)


# Log these robustness values, and chart them
# Sync the car animation with the robustness value animation?
# Save gif and present at group meeting Thursday