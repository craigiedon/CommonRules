# Load the scenario
import json

import numpy as np
import time
from commonroad.common.file_reader import CommonRoadFileReader

from stl import stl_rob
from trafficRules import safe_dist_rule
from utils import animate_scenario, animate_monitor, animate_scenario_and_monitor

file_path = "scenarios/Complex_Solution.xml"
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

T = 60
ego_id = 100
dt = 0.1
state_dict = [scenario.obstacle_states_at_time_step(i) for i in range(T)]
ego_car = scenario.obstacle_by_id(ego_id)
obstacles = [s for s in scenario.obstacles if s.obstacle_id != ego_id]

t_cut_in = int(np.round(3.0 / dt))
reaction_time = 0.3
acc_min = -10.5 * dt
lane_centres = [-1.75, 1.75, 5.25]
lane_widths = 3.5

dist_rules = [safe_dist_rule(ego_car, other_car, lane_centres, lane_widths, acc_min, reaction_time, t_cut_in) for other_car in obstacles]

start_time = time.time()

# ob_rob_vs = {}
# for r_i, rule in enumerate(dist_rules):
#     rob_vals = [stl_rob(rule, state_dict[:i], 0) for i in range(1, len(state_dict))]
#     # rob_vals = [stl_rob(rule, state_dict[:i], 0) for i in range(1, 10)]
#     ob_rob_vs[obstacles[r_i].obstacle_id] = rob_vals
#     # rob_vals = stl_rob(rule, state_dict, 0)
#     print(f"Obstacle: {obstacles[r_i].obstacle_id}, r: {rob_vals}")
# print(f"Function took: {time.time() - start_time} secs")

# Save ob_rob_vs to a json file
# with open("results/rg_1_results.json", "w") as f:
#     json.dump(ob_rob_vs, f)

with open("results/rg_1_results.json", "r") as f:
    ob_rob_vs = json.load(f)

animate_scenario_and_monitor(scenario, ob_rob_vs)
# animate_scenario(scenario, planning_problem_set, T)


# Log these robustness values, and chart them
# Sync the car animation with the robustness value animation?
# Save gif and present at group meeting Thursday