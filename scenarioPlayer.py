from commonroad.common.file_reader import CommonRoadFileReader

from anim_utils import animate_scenario

file_path = "scenarios/Complex_Solution.xml"
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

animate_scenario(scenario, planning_problem_set, 60)
