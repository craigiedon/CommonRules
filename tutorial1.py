#%%
import os
import matplotlib.pyplot as plt
import matplotlib

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

from IPython import display

# Generate path of file to be opened
file_path = "scenarios/ZAM_Tutorial-1_1_T-1.xml"

# read in teh scenario and planning problem set
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

# plot the scenario for 40 time step, here each time step corresponds to 0.1 second
for i in range(0, 40):
    plt.figure(figsize=(25, 10))
    rnd = MPRenderer()
    # plot the scenario at different tiem step
    scenario.draw(rnd, draw_params=({'time_begin': i}))

    # plot the planning problem set
    planning_problem_set.draw(rnd)
    rnd.render()

plt.show()
# %%

import numpy as np

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType
from commonroad.scenario.trajectory import State

# read in the scenario and planning problem set
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

# generate the static obstacle according to the specification, refer to API for detials of input parameters
static_obstacle_id = scenario.generate_object_id()
static_obstacle_type = ObstacleType.PARKED_VEHICLE
static_obstacle_shape = Rectangle(width=2.0, length=4.5)
static_obstacle_initial_state = State(position=np.array([30.0, 3.5]), orientation=0.02, time_step=0)

# feed in the required components to construct a static obstacle
static_obstacle = StaticObstacle(static_obstacle_id, static_obstacle_type, static_obstacle_shape, static_obstacle_initial_state)

scenario.add_objects(static_obstacle)

# plot the scenario for each time step
for i in range(0, 41):
    plt.figure(figsize=(25,10))
    rnd = MPRenderer()
    scenario.draw(rnd, draw_params={'time_begin': i})
    planning_problem_set.draw(rnd)
    rnd.render()

# %%

from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction

# initial state has a time step of 0
dynamic_obstacle_initial_state = State(position = np.array([50.0, 0.0]),
                                       velocity=22,
                                       orientation=0.02,
                                       time_step=0)

# generate states for the obstacle from time steps 1 to 40 by assuming constant velocity
state_list = []
for i in range(1, 41):
    # compute new position
    new_position = np.array([dynamic_obstacle_initial_state.position[0] + scenario.dt * i * 22, 0])

    # create new state
    new_state = State(position = new_position,
                      velocity=22,
                      orientation=0.02,
                      time_step=i)
    state_list.append(new_state)

# create the trajectory of the obstacle, starting at time step 1
dynamic_obstacle_trajectory = Trajectory(1, state_list)

# create the prediction using the trajectory and the shape of the obstacle
dynamic_obstacle_shape = Rectangle(width=1.8, length=4.3)
dynamic_obstacle_prediction = TrajectoryPrediction(dynamic_obstacle_trajectory, dynamic_obstacle_shape)

# generate the dynamic obstacle according to the specification
dynamic_obstacle_id = scenario.generate_object_id()
dynamic_obstacle_type = ObstacleType.CAR
dynamic_obstacle = DynamicObstacle(dynamic_obstacle_id,
                                   dynamic_obstacle_type,
                                   dynamic_obstacle_shape,
                                   dynamic_obstacle_initial_state,
                                   dynamic_obstacle_prediction)
# add dynamic obstacle to the scenario
scenario.add_objects(dynamic_obstacle)

# plot the scenario for each time step
for i in range(0, 41):
    plt.figure(figsize=(25,10))
    rnd = MPRenderer()
    scenario.draw(rnd, draw_params={'time_begin': i})
    planning_problem_set.draw(rnd)
    rnd.render()


# %%

from commonroad.common.file_writer import OverwriteExistingFile, CommonRoadFileWriter
from commonroad.scenario.scenario import Location, Tag

author = 'Craig Innes'
affiliation = 'University of Edinburgh'
source = ''
tags = {Tag.CRITICAL}

# write new scenario
fw = CommonRoadFileWriter(scenario, planning_problem_set, author, affiliation, source, tags)

filename = "ZAM_Tutorial-1_2_T-1.xml"
fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)

#%%

file_path = "ZAM_Tutorial-1_2_T-1.xml"

scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

# plot the scenario for each time step
for i in range(0, 40):
    plt.figure(figsize=(25,10))
    rnd = MPRenderer()
    scenario.draw(rnd, draw_params={'time_begin': i})
    planning_problem_set.draw(rnd)
    rnd.render()
