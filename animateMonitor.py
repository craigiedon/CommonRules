import json
import os

from commonroad.common.file_reader import CommonRoadFileReader

from anim_utils import animate_scenario_and_monitor


def run():
    file_path = "scenarios/Complex_Solution.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    T = 60
    ego_id = 100
    dt = 0.1
    state_dict = [scenario.obstacle_states_at_time_step(i) for i in range(T)]
    ego_car = scenario.obstacle_by_id(ego_id)
    obstacles = [s for s in scenario.obstacles if s.obstacle_id != ego_id]

    result_fns = sorted([r for r in os.listdir("results") if ".json" in r])
    for r in result_fns:
        with open(f"results/{r}", "r") as f:
            rob_vs = json.load(f)
        print(f"{r}: {rob_vs[-1]}")

    with open(f"results/rg_2.json", "r") as f:
        rob_vs = json.load(f)

    animate_scenario_and_monitor(scenario, rob_vs)
    # animate_scenario(scenario, planning_problem_set, T)


if __name__ == "__main__":
    run()
