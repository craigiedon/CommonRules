import copy
import json
import os
import time
from datetime import datetime
from os.path import join
import pickle
from typing import Dict, Tuple, Optional, List

import numpy as np
import pyro
from commonroad.common.file_writer import CommonRoadFileWriter
from commonroad.common.writer.file_writer_interface import OverwriteExistingFile
from commonroad.scenario.obstacle import Obstacle
from commonroad.scenario.state import CustomState, InitialState
from matplotlib import pyplot as plt
from scipy.stats import norm
import torch
import torch.nn.functional as F
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from CarMPC import car_visibilities_raycast, kalman_receding_horizon
from PyroGPClassification import load_gp_classifier
from PyroGPRegression import load_gp_reg
from TaskConfig import TaskConfig, CostWeights
from anim_utils import animate_with_predictions
from immFilter import c_vel_long_model, c_acc_long_model, lat_model
from monitorScenario import gen_interstate_rules, InterstateRulesConfig
from stl import stl_rob
from utils import RecedingHorizonStats, rot_mat, angle_diff, obs_long_lats, mpc_result_to_dyn_obj


class SimpleImportanceSampler(nn.Module):
    def __init__(self, in_dims, h_dims):
        super(SimpleImportanceSampler, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(in_dims, h_dims),
            # nn.Dropout(),
            nn.ReLU(),
            nn.Linear(h_dims, h_dims),
            # nn.Dropout(),
            nn.ReLU(),
            nn.Linear(h_dims, 5)
        )

    def forward(self, x):
        """ Outputs: Detection-Prob-Logit, Long_Noise_Mu, Long_Noise_Log_Var, Lat_Noise_Mu, Lat_Noise_Log_Var"""
        return self.ff(x)


def calc_lp_from_params(states_tensor, pos_det_probs, reg_mus, reg_vars, stats):
    # Collect stats for the likelihood of detections/noises encountered
    ordered_stats = list(stats.values())
    T = len(states_tensor)
    num_obs = len(states_tensor[0])

    det_probs = torch.zeros((T, num_obs), dtype=torch.float)
    long_log_ps = torch.zeros((T, num_obs), dtype=torch.float)
    lat_log_ps = torch.zeros((T, num_obs), dtype=torch.float)

    for t in range(0, T):
        for o_idx in range(0, num_obs):
            if ordered_stats[o_idx].observed_longs[t] is not None:
                # Detection probabilities
                det_probs[t, o_idx] = pos_det_probs[t, o_idx]

                t_long = ordered_stats[o_idx].true_longs[t][0]
                t_lat = ordered_stats[o_idx].true_lats[t][0]

                o_long = ordered_stats[o_idx].observed_longs[t][0]
                o_lat = ordered_stats[o_idx].observed_lats[t]

                # Noise PDFs
                long_noise = torch.tensor(t_long - o_long, dtype=torch.float)
                lat_noise = torch.tensor(t_lat - o_lat, dtype=torch.float)

                long_log_p = -F.gaussian_nll_loss(reg_mus[t, o_idx, 0], long_noise, reg_vars[t, o_idx, 0], full=True)
                lat_log_p = -F.gaussian_nll_loss(reg_mus[t, o_idx, 1], lat_noise, reg_vars[t, o_idx, 1], full=True)

                long_log_ps[t, o_idx] = long_log_p
                lat_log_ps[t, o_idx] = lat_log_p
            else:
                pass
                # Detection probabilities
                # (Noise PDF irrelevant if detection missed)
                det_probs[t, o_idx] = 1 - pos_det_probs[t, o_idx]

    # Combine probabilities
    log_det_probs = torch.log(det_probs)
    return torch.sum(log_det_probs + long_log_ps + lat_log_ps).item()


def log_probs_scenario_traj(states_tensor: torch.FloatTensor, stats: Dict[int, RecedingHorizonStats], pem_det,
                            pem_reg) -> float:
    # Get the PEM detection probabilities and noise means for the states encountered in the trajectories
    pyro.get_param_store().clear()
    pos_det_logits = pem_det(states_tensor.view(-1, states_tensor.shape[2]))[0].view(states_tensor.shape[0:2])
    pos_det_probs = torch.sigmoid(pos_det_logits)

    pyro.get_param_store().clear()
    reg_mus, reg_vars = pem_reg(states_tensor.view(-1, states_tensor.shape[2]))
    reg_mus = reg_mus.T.view(*states_tensor.shape[0:2], 2)
    reg_vars = reg_vars.T.view(*states_tensor.shape[0:2], 2)

    return calc_lp_from_params(states_tensor, pos_det_probs, reg_mus, reg_vars, stats)


def imp_log_probs_scenario_traj(states_tensor: torch.FloatTensor, stats: Dict[int, RecedingHorizonStats],
                                imp_sampler: nn.Module) -> float:

    imp_outs = imp_sampler(states_tensor.view(-1, states_tensor.shape[-1]))

    pos_det_probs = torch.sigmoid(imp_outs[:, 0])

    reg_long_mu = imp_outs[:, 1]
    reg_long_log_var = imp_outs[:, 2]
    reg_lat_mu = imp_outs[:, 3]
    reg_lat_log_var = imp_outs[:, 4]

    reg_mus = reg_mus.T.view(*states_tensor.shape[0:2], 2)
    reg_vars = reg_vars.T.view(*states_tensor.shape[0:2], 2)

    return calc_lp_from_params(states_tensor, pos_det_probs, reg_mus, reg_vars, stats)


def convert_PEM_traj(T: int, ego_id: int, scenario: Scenario, norm_mus, norm_stds) -> torch.FloatTensor:
    # Extract the true lat/longs, then get them relative to
    states_tensor = []
    obstacles = [o for o in scenario.obstacles if o.obstacle_id != ego_id]

    for t in range(T):
        ego_state = scenario.obstacle_by_id(ego_id).state_at_time(t)
        rot_frame = rot_mat(-ego_state.orientation)
        ego_pos = ego_state.position

        visibilities = car_visibilities_raycast(100, ego_state, t, obstacles)
        t_long, t_lat = obs_long_lats(obstacles, 0, T)
        tru_long_states, tru_lat_states = np.array(list(t_long.values())), np.array(list(t_lat.values()))

        obs_dims = torch.tensor([[o.obstacle_shape.length, o.obstacle_shape.width, 1.7] for o in obstacles],
                                dtype=torch.float)
        obs_rots = torch.tensor([o.state_at_time(t).orientation for o in obstacles], dtype=torch.float)

        s_longs, s_lats = rot_frame @ np.array(
            [tru_long_states[:, t, 0] - ego_pos[0], tru_lat_states[:, t, 0] - ego_pos[1]])
        s_rs = angle_diff(obs_rots, ego_state.orientation)

        state_tensor = torch.column_stack([
            torch.tensor(s_longs, dtype=torch.float),
            torch.tensor(s_lats, dtype=torch.float),
            torch.sin(s_rs),
            torch.cos(s_rs),
            obs_dims,
            torch.tensor(visibilities, dtype=torch.float)
        ])

        state_tensor = (state_tensor - norm_mus) / norm_stds
        states_tensor.append(state_tensor)

    states_tensor = torch.stack(states_tensor).cuda()
    assert states_tensor.shape[0] == T and states_tensor.shape[1] == len(obstacles) and states_tensor.shape[2] == 8
    # return states_tensor.view((T * len(obstacles), 8))
    return states_tensor


def sample_from_imp_sampler(obs: List[Obstacle],
                            ego_state: CustomState,
                            t: int,
                            tru_long_states: np.ndarray,
                            tru_lat_states: np.ndarray,
                            imp_sam_pem: nn.Module,
                            norm_mus: torch.Tensor,
                            norm_stds: torch.Tensor,
                            o_viz: np.ndarray) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    rot_frame = rot_mat(-ego_state.orientation)
    ego_pos = ego_state.position

    s_longs, s_lats = rot_frame @ np.array([tru_long_states[:, 0] - ego_pos[0], tru_lat_states[:, 0] - ego_pos[1]])

    obs_rots = np.array([o.state_at_time(t).orientation for o in obs])
    s_rs = torch.tensor(angle_diff(obs_rots, ego_state.orientation), dtype=torch.float)

    s_dims = torch.tensor([[o.obstacle_shape.length, o.obstacle_shape.width, 1.7] for o in obs], dtype=torch.float)

    state_tensor = torch.column_stack([torch.tensor(s_longs, dtype=torch.float),
                                       torch.tensor(s_lats, dtype=torch.float),
                                       torch.sin(s_rs),
                                       torch.cos(s_rs),
                                       s_dims,
                                       torch.tensor(o_viz, dtype=torch.float)])
    state_tensor = (state_tensor - norm_mus) / norm_stds
    state_tensor = state_tensor.cuda()

    imp_sam_pem.eval()
    with torch.no_grad():
        is_output = imp_sam_pem(state_tensor).cpu().detach()
        # det_logit, long_n_mu, long_n_logvar, lat_n_mu, lat_n_logvar = imp_sam_pem(state_tensor)

    det_ps = torch.sigmoid(is_output[:, 0])
    long_n_mu = is_output[:, 1]
    long_n_var = torch.exp(is_output[:, 2])

    lat_n_mu = is_output[:, 3]
    lat_n_var = torch.exp(is_output[:, 4])

    # Longitudinal observation: Position / Velocity
    long_noise = np.random.normal(long_n_mu, np.sqrt(long_n_var))
    observed_long_pos = tru_long_states[:, 0] + long_noise
    observed_long_vel = tru_long_states[:, 1] + long_noise
    observed_long_state = np.array([observed_long_pos, observed_long_vel]).T

    # Latitudinal observation: Position
    observed_lat_state = tru_lat_states[:, 0] + np.random.normal(lat_n_mu, np.sqrt(lat_n_var))

    rands = np.random.rand(len(obs))
    observed_long_state = [s if r < det_p else None for s, r, det_p in zip(observed_long_state, rands, det_ps)]
    observed_lat_state = [s if r < det_p else None for s, r, det_p in zip(observed_lat_state, rands, det_ps)]

    return observed_long_state, observed_lat_state


def pre_trained_imp_sampler(in_dims: int, target_det_prob: float, target_variance: float) -> nn.Module:
    imp_sampler = SimpleImportanceSampler(in_dims, 20).cuda()

    N_toy = 10000
    toy_ins = torch.normal(torch.zeros(N_toy, in_dims), torch.ones(N_toy, in_dims))
    toy_labels = torch.tile(
        torch.tensor([torch.logit(torch.tensor(target_det_prob), eps=1e-6).item(), 0.0, np.log(target_variance), 0.0,
                      np.log(target_variance)],
                     dtype=torch.float),
        (N_toy, 1))
    toy_dataset = TensorDataset(toy_ins.cuda(), toy_labels.cuda())
    toy_loader = DataLoader(toy_dataset, 1024, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(imp_sampler.parameters())

    epochs = 100
    imp_sampler.train()
    avg_losses = []
    for e in range(epochs):
        losses = []
        for toy_X, toy_label in toy_loader:
            imp_pred = imp_sampler(toy_X)
            loss = loss_fn(imp_pred, toy_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        print(f"Epoch {e}: ", avg_loss)
        avg_losses.append(avg_loss)

    return imp_sampler

    # plt.plot(range(len(avg_losses)), avg_losses)
    # plt.show()

    # n_test = 300
    # line_tests = torch.column_stack(
    #     (torch.zeros(n_test, 1), torch.linspace(-4, 4, n_test), torch.zeros((n_test, 6)))).cuda()
    # print(line_tests)
    #
    # imp_sampler.eval()
    # eval_imp = imp_sampler(line_tests)
    # plt.plot(line_tests.cpu().detach()[:, 1], eval_imp.cpu().detach()[:, 1:])
    # plt.show()


def run():
    res_fp = "results/kal_mpc_res_23-05-17-17-47-58"
    file_path = "results/kal_mpc_res_23-05-16-13-22-16/kal_mpc_0.xml"
    scenario, planning_problem_set = CommonRoadFileReader(join(res_fp, "kal_mpc_0.xml")).open()

    ego_id = 100
    with open("config/interstate_rule_config.json", 'r') as f:
        irc = InterstateRulesConfig(**json.load(f))

    lane_cs = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [3, 6, 10]]
    access_cs = lane_cs[0:1]
    ego_lane_cs = lane_cs[1:]
    lane_widths = np.abs((ego_lane_cs[0] - ego_lane_cs[1]) / 2.0)

    T = 40

    with open(join(res_fp, f"prediction_stats_{0}.pkl"), 'rb') as f:
        loaded_stats: Dict[int, RecedingHorizonStats] = pickle.load(f)

    # Load the PEMs here (with normed states)
    det_pem = load_gp_classifier("models/nuscenes/vsgp_class", True)
    det_pem.eval()
    reg_pem = load_gp_reg("models/nuscenes/sgp_reg", True)
    reg_pem.eval()
    norm_mus = torch.load("data/nuscenes/inp_mus.pt")
    norm_stds = torch.load("data/nuscenes/inp_stds.pt")

    # Calculate the probability density function
    pem_states = convert_PEM_traj(T, 100, scenario, norm_mus, norm_stds)
    log_prob = log_probs_scenario_traj(pem_states, loaded_stats, det_pem, reg_pem)
    print(log_prob)

    imp_sampler = pre_trained_imp_sampler(norm_mus.shape[0], 0.5, 0.5)
    # Save importance sampler weights
    torch.save(imp_sampler.state_dict(), "models/imp_toy_0.1.pyt")

    # Load importance sampler weights
    imp_sampler = SimpleImportanceSampler(8, 20).cuda()
    imp_sampler.load_state_dict(torch.load("models/imp_toy_0.1.pyt"))

    # Create an observation function for the importance sampler
    def imp_obs_f(obs, ego_state, t, tlong, tlat, vs):
        return sample_from_imp_sampler(obs, ego_state, t, tlong, tlat, imp_sampler, norm_mus, norm_stds, vs)

    file_path = "scenarios/Complex.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    all_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [3, 6, 10]]
    ego_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [6, 10]]
    lane_widths = np.abs((ego_lane_centres[0] - ego_lane_centres[1]) / 2.0)

    goal_state = planning_problem_set.find_planning_problem_by_id(1).goal.state_list[0].position.center
    end_time = 4.0
    with open("config/task_config.json") as f:
        task_config = TaskConfig(**json.load(f))

    start_state = InitialState(position=np.array([0.0 + task_config.car_length / 2.0, ego_lane_centres[0]]),
                               velocity=task_config.v_goal * 0.8,
                               orientation=0, acceleration=0.0, time_step=0)

    with open("config/cost_weights.json", 'r') as f:
        cws = CostWeights(**json.load(f))

    long_models = [
        c_vel_long_model(task_config.dt, 1.0, 0.1),
        c_acc_long_model(task_config.dt, 1.0, 0.1)
    ]

    lat_models = [
        lat_model(task_config.dt, kd, 7, p_ref, 0.1, 0.1)
        for kd in np.linspace(3.0, 5.0, 3)
        for p_ref in all_lane_centres]

    results_folder_path = f"results/imp_sampler_res_{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}"
    os.mkdir(results_folder_path)
    rep_rob_vals = []
    for i in range(100):
        kalman_time = time.time()
        dn_state_list, prediction_stats = kalman_receding_horizon(end_time, 2.0, start_state, scenario, task_config,
                                                                  long_models, lat_models, imp_obs_f, cws)
        print(f"{i}: {time.time() - kalman_time}")
        solution_scenario = copy.deepcopy(scenario)
        ego_soln_obj = mpc_result_to_dyn_obj(100, dn_state_list, task_config.car_width,
                                             task_config.car_length)
        solution_scenario.add_objects(ego_soln_obj)

        rules = gen_interstate_rules(ego_id, solution_scenario, lane_cs, lane_widths, ego_lane_cs, access_cs, irc)
        solution_state_dict = [solution_scenario.obstacle_states_at_time_step(i) for i in range(len(dn_state_list))]
        rob_vals = []
        for rule_name, rule in rules.items():
            rob_val = stl_rob(rule, solution_state_dict, 0)
            print(rule_name, rob_val)
            rob_vals.append(rob_val)
        rep_rob_vals.append(rob_vals)

        animate_with_predictions(solution_scenario, prediction_stats, int(end_time / task_config.dt), show=True)

        np.savetxt(os.path.join(results_folder_path, "rule_rob_vals.txt"), rep_rob_vals, fmt="%.4f")
        with open(os.path.join(results_folder_path, f"prediction_stats_{i}.pkl"), 'wb') as f:
            pickle.dump(prediction_stats, f)

        scenario_save_path = os.path.join(results_folder_path, f"kal_mpc_{i}.xml")
        fw = CommonRoadFileWriter(solution_scenario, planning_problem_set, "Craig Innes", "University of Edinburgh")
        fw.write_to_file(scenario_save_path, OverwriteExistingFile.ALWAYS)

        # TODO: Log the log-probabilities of sample trajectories under the original PEM
        ep_pem_states = convert_PEM_traj(T, 100, solution_scenario, norm_mus, norm_stds)
        orig_pem_log_prob = log_probs_scenario_traj(ep_pem_states, prediction_stats, det_pem, reg_pem)
        print(f"Orig PEM LP: {orig_pem_log_prob}")

        # TODO: Log the log-probabilities of sampler trajectories under the importance sampler
        imp_sampler_log_prob = None

    # Plot velocity over time
    # plt.plot([s.velocity for s in dn_state_list])
    # plt.ylabel("Velocity")
    # plt.show()
    # Plot acceleration over time
    # plt.plot([s.acceleration for s in dn_state_list])
    # plt.ylabel("Acceleration")
    # plt.show()

    # TODO: Calculate the cross-entropy score here.
    # TODO: Calculate the cross-entropy score here (including the adaptive thresholding filter from previous repo...)
    # TODO: Run in a minibatch loop with movement. Chart failures, and chart log-likelihoods...


if __name__ == "__main__":
    run()
