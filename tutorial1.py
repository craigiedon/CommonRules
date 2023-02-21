import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import vehiclemodels
# from IPython

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.common.util import Interval
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import ObstacleType, DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.visualization.mp_renderer import MPRenderer
import cvxpy
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.utility.visualization import visualize_route
from cvxpy import Variable, quad_form, Minimize, Problem
from matplotlib import animation
from matplotlib.animation import FuncAnimation, PillowWriter
from vehiclemodels import parameters_vehicle3

from utils import animate_scenario

file_path = "scenarios/EmptyRamp.xml"

scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
p1 = list(planning_problem_set.planning_problem_dict.values())[0]

# s_obs = scenario.obstacles
# for ob in s_obs:
#     scenario.remove_obstacle(ob)
#
# p1 = planning_problem_set.find_planning_problem_by_id(1)
# p1.initial_state.velocity = 30.0
# p1.goal.state_list = [State(time_step=Interval(0, 200), position=Rectangle(width=3.5, length=10.0, center=np.array([235.0, 1.75])))]

# fw = CommonRoadFileWriter(scenario, planning_problem_set, "Craig Innes", "University of Edinburgh")
# fw.write_to_file("scenarios/EmptyRamp.xml", OverwriteExistingFile.ALWAYS)

l_cvs = scenario.lanelet_network.find_lanelet_by_id(10).center_vertices
centre_pos = l_cvs[0]

timesteps = 40

dyn_obs_shape = Rectangle(width=1.8, length=4.3)
dyn_obs_init = State(position=centre_pos, velocity = 22, orientation=0.0, time_step=0)

dn_state_list = []
for i in range(1, timesteps + 1):
    new_pos = np.array([dyn_obs_init.position[0] + scenario.dt * i * 22, dyn_obs_init.position[1]])
    dn_state_list.append(State(position=new_pos, velocity = 22, orientation=0.02, time_step=i))

dyn_obs_traj = Trajectory(1, dn_state_list)
dyn_obs_pred = TrajectoryPrediction(dyn_obs_traj, dyn_obs_shape)
dyn_obs = DynamicObstacle(scenario.generate_object_id(),
                          ObstacleType.CAR,
                          dyn_obs_shape,
                          dyn_obs_init,
                          dyn_obs_pred)
# scenario.add_objects(dyn_obs)

route_planner = RoutePlanner(scenario, p1, backend=RoutePlanner.Backend.NETWORKX_REVERSED)
candidate_holder = route_planner.plan_routes()
route =  candidate_holder.retrieve_first_route()

print(route.reference_path)
visualize_route(route, draw_route_lanelets=True, draw_reference_path=True) #, size_x=6)


# animate_scenario(scenario, planning_problem_set, timesteps)


@dataclass
class TIConstraints:
    a_min : float = -8 # Minimum feasible vehicle acceleration
    a_max : float = 15 # Maximum feasible vehicle acceleration
    s_min : float = 0 # Minimum allowed position
    s_max : float = 150 # Maximum allowed position
    v_min : float = 0 # Minimum  velocity
    v_max : float = 35 # Max velocity
    j_min : float = -30 # Minimum allowed jerk
    j_max : float = 30 # Maximum allowed jerk

def plot_state_vector(x: Variable, c: TIConstraints, s_obj = None):
    plt.figure(figsize=(10,10))
    N = x.shape[1]-1
    s_max = np.maximum(150,100+0*np.ceil(np.amax((x.value)[0,:].flatten())*1.1/10)*10)

    # Plot (x_t)_1.
    plt.subplot(4,1,1)
    x1 = x.value[0, :].flatten()
    plt.plot(np.array(range(N+1)),x1,'g')
    if s_obj is not None:
        plt.plot(np.array(range(1,N+1)),s_obj[0],'b')
        plt.plot(np.array(range(1,N+1)),s_obj[1],'r')
    plt.ylabel(r"$s$", fontsize=16)
    plt.yticks(np.linspace(c.s_min, s_max, 3))
    plt.ylim([c.s_min, s_max])
    plt.xticks([])

    # Plot (x_t)_2.
    plt.subplot(4,1,2)
    x2 = x.value[1, :].flatten()
    plt.plot(np.array(range(N+1)),x2,'g')
    plt.yticks(np.linspace(c.v_min,c.v_max,3))
    plt.ylim([c.v_min, c.v_max+2])
    plt.ylabel(r"$v$", fontsize=16)
    plt.xticks([])

    # Plot (x_t)_3.
    plt.subplot(4,1,3)
    x2 = x.value[2, :].flatten()
    plt.plot(np.array(range(N+1)),x2,'g')
    plt.yticks(np.linspace(c.a_min,c.a_max,3))
    plt.ylim([c.a_min, c.a_max+2])
    plt.ylabel(r"$a$", fontsize=16)
    plt.xticks([])

    # Plot (x_t)_4.
    plt.subplot(4,1,4)
    x2 = x.value[3, :].flatten()
    plt.plot(np.array(range(N+1)), x2,'g')
    plt.yticks(np.linspace(c.j_min,c.j_max,3))
    plt.ylim([c.j_min-1, c.j_max+1])
    plt.ylabel(r"$j$", fontsize=16)
    plt.xticks(np.arange(0,N+1,5))
    plt.xlabel(r"$k$", fontsize=16)
    plt.tight_layout()
    plt.show()

# Problem Data
N = 40 # Number Timesteps
n = 4 # State Vector Length
m = 1 # length of input vector
dT = scenario.dt # Time Step

# set up variables
x = Variable(shape=(n,N+1)) # optimization vector x contains n states per time step 
u = Variable(shape=(N)) # optimization vector u contains 1 stateio.dt

c = TIConstraints(-6, 6, 0, 100, 0, 35, -15, 15)

# weights for cost function
w_s = 0
w_v = 8
w_a = 2
w_j = 2
Q = np.eye(n)*np.transpose(np.array([w_s,w_v,w_a,w_j]))
w_u = 1
R = w_u


# Vehicle model: 4th Order Point-Mass model
A = np.array([[1,dT,(dT**2)/2,(dT**3)/6],
               [0,1,dT,(dT**2)/2],
               [0,0,1,dT],
               [0,0,0,1]])

B = np.array([(dT**4)/24,
               (dT**3)/6,
               (dT**2)/2,
               dT]).reshape([n,])

planning_problem = planning_problem_set.find_planning_problem_by_id(100)
initial_state = planning_problem.initial_state

# Initial state vector: (longitudinal pos, velocity, acceleration, jerk)
x_0 = np.array([initial_state.position[0],
                 initial_state.velocity,
                 0.0,
                 0.0]).reshape([n,]) # initial state

# Reference velocity
v_ref = 30.0

states = []
cost = 0


constr = [x[:, 0] == x_0] # Initial state constraint


dyn_obstacles = scenario.dynamic_obstacles

# create constraints for minimum and maximum position
s_min = []
s_max = []

# Note: I think this loop assumes there are only two vehicles, and that one is lead and one is follow. Otherwise the code breaks?
for o in dyn_obstacles:
    prediction = o.prediction.trajectory.state_list
    if o.initial_state.position[0] < x_0[0]:
        print("Following vehicle id={}".format(o.obstacle_id))
        for p in prediction:
            s_min.append(p.position[0] + o.obstacle_shape.length / 2.0 + 2.5)
    else:
        print("Leading vehicle id ={}".format(o.obstacle_id))
        for p in prediction:
            s_max.append(p.position[0] - o.obstacle_shape.length/2.0 - 2.5)

for k in range(N):
    cost += quad_form(x[:, k + 1] - np.array([0, v_ref, 0, 0], ), Q) + R * u[k] ** 2

    # Time variant state/input constraints
    constr.append(x[:, k + 1] == A @ x[:, k] + B * u[k])

    # Obstacle collision constraints
    constr.append(x[0, k + 1] <= s_max[k])
    constr.append(x[0, k + 1] >= s_min[k])

constr.extend([x[1, :] <= c.v_max, x[1, :] >= c.v_min])  # Velocity
constr.extend([x[2, :] <= c.a_max, x[2, :] >= c.a_min])  # acceleration
constr.extend([x[3, :] <= c.j_max, x[3, :] >= c.j_min])  # jerk

prob = Problem(Minimize(cost), constr)

prob.solve(verbose=True)
print("Problem is convex: ", prob.is_dcp())
print("Problem solution is" + prob.status)

# plot_state_vector(x, c, [s_min, s_max])

x_result = x.value
s_ego = x_result[0, :].flatten()
v_ego = x_result[1, :].flatten()

state_list = [initial_state]
for i in range(1, N):
    orientation = initial_state.orientation
    state_list.append(State(**{
        'position': np.array([s_ego[i], 0]),
        'orientation': orientation,
        'time_step': i,
        'velocity': v_ego[i] * np.cos(orientation),
        'velocity_y': v_ego[i] * np.sin(orientation)
    }))

# create the planned trajectory starting at time step 1
ego_vehicle_trajectory = Trajectory(initial_time_step=1, state_list=state_list[1:])

# Create teh prediction using the planned trajectory and the shape of the ego vehicle
vehicle3 = parameters_vehicle3.parameters_vehicle3()
ego_vehicle_shape = Rectangle(length=vehicle3.l, width=vehicle3.w)
ego_vehicle_prediction = TrajectoryPrediction(trajectory=ego_vehicle_trajectory, shape=ego_vehicle_shape)

# The ego vehicle can be visualized by converting it into a Dynamic Obstacle
ego_vehicle_type = ObstacleType.CAR
ego_vehicle = DynamicObstacle(obstacle_id=100, obstacle_type=ego_vehicle_type, obstacle_shape=ego_vehicle_shape, initial_state=initial_state,
                              prediction=ego_vehicle_prediction)

fig, ax = plt.subplots(figsize=(25, 3))
rnd = MPRenderer(ax=ax)
dps = rnd.draw_params
rnd.draw_params["planning_problem"]["initial_state"]["state"].update({"zorder": 20})
is_s = rnd.draw_params["planning_problem"]["initial_state"]["state"]
print(dps)
def animate(i):
    # rnd.draw_params.time_begin = i
    scenario.draw(rnd, draw_params={"time_begin": i})
    ego_vehicle.draw(rnd, draw_params={'time_begin': i, 'dynamic_obstacle': {
        'vehicle_shape': {'occupancy': {'shape': {'rectangle': {
            'facecolor': 'g'}}}}}})
    planning_problem_set.draw(rnd)
    rnd.render()

ani = FuncAnimation(fig, animate, frames=40, interval=32, repeat=True, repeat_delay=200)
# ani.save("tutorial1.gif", animation.PillowWriter(fps=30))
plt.show()
