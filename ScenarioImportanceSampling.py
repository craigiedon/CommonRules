import copy
import json
import os
import time
from datetime import datetime
from math import ceil, floor
from os.path import join
import pickle
from typing import Dict, Tuple, Optional, List, Sequence, Any

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
from torch import nn, FloatTensor, Tensor
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


def calc_lp_from_params(states_tensor, pos_det_probs, reg_mus, reg_vars, det_ind: torch.Tensor,
                        long_noises: torch.Tensor, lat_noises: torch.Tensor) -> torch.Tensor:
    # Collect stats for the likelihood of detections/noises encountered
    # ordered_stats = list(stats.values())
    T = len(states_tensor)
    num_obs = len(states_tensor[0])

    det_probs = det_ind * pos_det_probs + (1 - det_ind) * (1 - pos_det_probs)

    long_log_ps = -F.gaussian_nll_loss(long_noises, reg_mus[:, :, 0], reg_vars[:, :, 0], full=True,
                                       reduction='none') * det_ind
    lat_log_ps = -F.gaussian_nll_loss(lat_noises, reg_mus[:, :, 1], reg_vars[:, :, 1], full=True,
                                      reduction='none') * det_ind

    # Combine probabilities
    log_det_probs = torch.log(det_probs)
    return torch.sum(log_det_probs + long_log_ps + lat_log_ps)


def log_probs_scenario_traj(states_tensor: torch.FloatTensor, det_ind: torch.Tensor, long_noises: torch.Tensor,
                            lat_noises: torch.Tensor, pem_det,
                            pem_reg) -> torch.Tensor:
    # Get the PEM detection probabilities and noise means for the states encountered in the trajectories

    pyro.get_param_store().clear()
    pos_det_logits = pem_det(states_tensor.view(-1, states_tensor.shape[2]))[0].view(states_tensor.shape[0:2])
    pos_det_probs = torch.sigmoid(pos_det_logits)

    pyro.get_param_store().clear()
    reg_mus, reg_vars = pem_reg(states_tensor.view(-1, states_tensor.shape[2]))
    reg_mus = reg_mus.T.view(*states_tensor.shape[0:2], 2)
    reg_vars = reg_vars.T.view(*states_tensor.shape[0:2], 2)

    return calc_lp_from_params(states_tensor, pos_det_probs, reg_mus, reg_vars, det_ind, long_noises, lat_noises)


def imp_log_probs_scenario_traj(states_tensor: torch.FloatTensor, det_ind: torch.Tensor, long_noises: torch.Tensor,
                                lat_noises: torch.Tensor,
                                imp_sampler: nn.Module) -> torch.Tensor:
    imp_outs = imp_sampler(states_tensor)

    pos_det_probs = torch.sigmoid(imp_outs[:, :, 0])

    reg_long_mu = imp_outs[:, :, 1]
    reg_long_log_var = imp_outs[:, :, 2]
    reg_lat_mu = imp_outs[:, :, 3]
    reg_lat_log_var = imp_outs[:, :, 4]

    reg_mus = torch.stack((reg_long_mu, reg_lat_mu), dim=-1)
    reg_vars = torch.exp(torch.stack((reg_long_log_var, reg_lat_log_var), dim=-1))

    return calc_lp_from_params(states_tensor, pos_det_probs, reg_mus, reg_vars, det_ind, long_noises, lat_noises)


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
                            o_viz: np.ndarray,
                            samp_probs_f_name: str) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
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
    state_tensor: Tensor = state_tensor.cuda()

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
    lat_noise = np.random.normal(lat_n_mu, np.sqrt(lat_n_var))
    observed_lat_state = tru_lat_states[:, 0] + lat_noise

    rands = torch.tensor(np.random.rand(len(obs)), dtype=torch.float)
    observed_long_state = [s if r < det_p else None for s, r, det_p in zip(observed_long_state, rands, det_ps)]
    observed_lat_state = [s if r < det_p else None for s, r, det_p in zip(observed_lat_state, rands, det_ps)]

    log_det_prob = torch.log((rands < det_ps) * det_ps + (rands >= det_ps) * (1 - det_ps)).sum()
    long_log_prob = -F.gaussian_nll_loss(torch.tensor(long_noise, dtype=torch.float), long_n_mu, long_n_var,
                                         full=True, reduction='none')
    lat_log_prob = -F.gaussian_nll_loss(torch.tensor(lat_noise, dtype=torch.float), lat_n_mu, lat_n_var, full=True,
                                        reduction='none')
    log_noise_prob = (rands < det_ps) * (long_log_prob + lat_log_prob)
    total_log_prob = log_det_prob + log_noise_prob.sum()

    with open(f"results/{samp_probs_f_name}", 'a') as f:
        np.savetxt(f, total_log_prob.view(1, -1))

    with open(f"results/stateTensors-{samp_probs_f_name}", 'a') as f:
        np.savetxt(f, state_tensor.detach().cpu().numpy())

    with open(f"results/longNoise-{samp_probs_f_name}", 'a') as f:
        np.savetxt(f, long_noise)

    with open(f"results/latNoise-{samp_probs_f_name}", 'a') as f:
        np.savetxt(f, lat_noise)

    torch.save(imp_sam_pem.state_dict(), "frozen_imp_sampler.pyt")

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


def rho_quantile(vals: Sequence[Any], rho: float, min_val: float) -> float:
    descending_safety = sorted(vals, reverse=True)
    rho_q = descending_safety[floor((1.0 - rho) * len(vals))]
    return max(rho_q, min_val)


def ce_score(pem_states, prediction_stats, det_pem, reg_pem, imp_sampler) -> torch.Tensor:
    orig_pem_log_prob = log_probs_scenario_traj(pem_states, prediction_stats, det_pem, reg_pem)
    imp_sampler_log_prob = imp_log_probs_scenario_traj(pem_states, prediction_stats, imp_sampler)
    ll_ratio = (orig_pem_log_prob - imp_sampler_log_prob)
    likelihood_ratio = torch.exp(ll_ratio)
    cross_ent = -likelihood_ratio * imp_sampler_log_prob
    return cross_ent


def ce_score_batch_stable(ep_pem_states, ep_pred_stats, det_pem, reg_pem, imp_sampler) -> torch.Tensor:
    orig_pem_lps = torch.stack([log_probs_scenario_traj(pem_s, pred_s, det_pem, reg_pem) for pem_s, pred_s in
                                zip(ep_pem_states, ep_pred_stats)])
    imp_sampler_lps = torch.stack([imp_log_probs_scenario_traj(pem_s, pred_s, imp_sampler) for pem_s, pred_s in
                                   zip(ep_pem_states, ep_pred_stats)])
    ll_ratios = (orig_pem_lps - imp_sampler_lps)
    # ratio_normalizer = torch.max(ll_ratios).item()

    likelihood_ratios = torch.exp(ll_ratios * 0.1)
    cross_ents = -likelihood_ratios * imp_sampler_lps
    return cross_ents


def ce_one_step(ep_pem_states, ep_det_inds: List[torch.tensor], ep_long_noises: List[torch.Tensor],
                ep_lat_noises: List[torch.Tensor], rep_rob_vals, imp_sampler: nn.Module, det_pem: nn.Module,
                reg_pem: nn.Module):
    rep_rob_vals = torch.tensor(rep_rob_vals)
    rrv_1 = sorted(rep_rob_vals[:, 0], reverse=True)
    ce_thresh = rho_quantile(rrv_1, 0.05, 0.0)
    ce_opt = torch.optim.Adam(imp_sampler.parameters())
    ce_solver_its = 1000

    ce_losses = []
    orig_pem_lps = torch.stack(
        [log_probs_scenario_traj(pem_s, d_ind, long_ns, lat_ns, det_pem, reg_pem) for pem_s, d_ind, long_ns, lat_ns in
         zip(ep_pem_states, ep_det_inds, ep_long_noises, ep_lat_noises)])
    old_imp_sampler_lps = torch.stack(
        [imp_log_probs_scenario_traj(pem_s, d_ind, long_ns, lat_ns, imp_sampler) for pem_s, d_ind, long_ns, lat_ns in
         zip(ep_pem_states, ep_det_inds, ep_long_noises, ep_lat_noises)])
    ll_ratios = (orig_pem_lps - old_imp_sampler_lps).cuda().detach()
    indicator_flag = (rep_rob_vals[:, 0] <= ce_thresh).cuda().detach()

    for ci in range(ce_solver_its):
        # cross_ents = ce_score_batch_stable(ep_pem_states, ep_pred_stats, det_pem, reg_pem, imp_sampler)

        imp_sampler_lps = torch.stack(
            [imp_log_probs_scenario_traj(pem_s, d_ind, long_ns, lat_ns, imp_sampler) for pem_s, d_ind, long_ns, lat_ns
             in
             zip(ep_pem_states, ep_det_inds, ep_long_noises, ep_lat_noises)])

        cross_ents = -(ll_ratios * 0.1).exp() * imp_sampler_lps
        ce_loss = (indicator_flag * cross_ents).sum() / len(ep_pem_states)

        # if not ce_loss.isfinite():
        #     print("Infinite/Nan Value")

        ce_opt.zero_grad()
        ce_loss.backward()
        ce_opt.step()

        if ci % 100 == 0:
            print(f"ci: {ci}: ", ce_loss.item())
        ce_losses.append(ce_loss.item())

    # print("Done with all that then")

    # plt.plot(ce_losses)
    # plt.show()


def imp_obs_f(imp_sampler, norm_mus, norm_stds, fname):
    return lambda obs, ego_state, t, tlong, tlat, vs: sample_from_imp_sampler(obs, ego_state, t, tlong, tlat,
                                                                              imp_sampler, norm_mus, norm_stds, vs,
                                                                              fname)


def dets_and_noise_from_stats(prediction_stats: Dict[int, RecedingHorizonStats]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    long_noises = []
    lat_noises = []
    det_ind = []
    for rhs in prediction_stats.values():
        det_ind.append(torch.tensor([1 if ol is not None else 0 for ol in rhs.observed_longs], device='cuda'))
        long_noises.append(torch.tensor(
            [(ol[0] - tl[0]) if ol is not None else 0.0 for ol, tl in zip(rhs.observed_longs, rhs.true_longs)],
            dtype=torch.float, device='cuda'))
        lat_noises.append(
            torch.tensor([(ol - tl[0]) if ol is not None else 0.0 for ol, tl in zip(rhs.observed_lats, rhs.true_lats)],
                         dtype=torch.float, device='cuda'))

    # Offset by 1 because we assume state is known at T=0
    long_noises = torch.stack(long_noises).T[1:]
    lat_noises = torch.stack(lat_noises).T[1:]
    det_ind = torch.stack(det_ind).T[1:]
    return det_ind, long_noises, lat_noises


def run():
    # res_fp = "results/kal_mpc_res_23-05-17-17-47-58"
    # file_path = "results/kal_mpc_res_23-05-16-13-22-16/kal_mpc_0.xml"
    scenario_fp = "scenarios/Complex.xml"
    scenario, planning_problem_set = CommonRoadFileReader(scenario_fp).open()

    ego_id = 100
    with open("config/interstate_rule_config.json", 'r') as f:
        irc = InterstateRulesConfig(**json.load(f))

    lane_cs = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [3, 6, 10]]
    access_cs = lane_cs[0:1]
    ego_lane_cs = lane_cs[1:]
    lane_widths = np.abs((ego_lane_cs[0] - ego_lane_cs[1]) / 2.0)

    T = 40

    # with open(join(res_fp, f"prediction_stats_{0}.pkl"), 'rb') as f:
    #     loaded_stats: Dict[int, RecedingHorizonStats] = pickle.load(f)

    # Load the PEMs here (with normed states)
    det_pem = load_gp_classifier("models/nuscenes/vsgp_class", True)
    det_pem.eval()
    reg_pem = load_gp_reg("models/nuscenes/sgp_reg", True)
    reg_pem.eval()
    norm_mus = torch.load("data/nuscenes/inp_mus.pt")
    norm_stds = torch.load("data/nuscenes/inp_stds.pt")

    # Calculate the probability density function
    # pem_states = convert_PEM_traj(T, 100, scenario, norm_mus, norm_stds)
    # log_prob = log_probs_scenario_traj(pem_states, loaded_stats, det_pem, reg_pem)
    # print(log_prob)

    # imp_sampler = pre_trained_imp_sampler(norm_mus.shape[0], 0.5, 0.5)
    # Save importance sampler weights
    # torch.save(imp_sampler.state_dict(), "models/imp_toy_0.1.pyt")

    # Load importance sampler weights
    imp_sampler = SimpleImportanceSampler(8, 20).cuda()
    imp_sampler.load_state_dict(torch.load("models/imp_toy_0.1.pyt"))

    # Create an observation function for the importance sampler

    file_path = "scenarios/Complex.xml"
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    all_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [3, 6, 10]]
    ego_lane_centres = [scenario.lanelet_network.find_lanelet_by_id(l).center_vertices[0][1] for l in [6, 10]]
    lane_widths = np.abs((ego_lane_centres[0] - ego_lane_centres[1]) / 2.0)

    # goal_state = planning_problem_set.find_planning_problem_by_id(1).goal.state_list[0].position.center
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
    os.makedirs(results_folder_path, exist_ok=True)

    # TODO: Make this a lil function?
    sims_per_CE = 10
    stages_CE = 10

    lp_means = []
    fail_props = []

    for stage in range(stages_CE):

        rep_rob_vals = []
        ep_pem_states = []
        ep_long_noises = []
        ep_lat_noises = []
        ep_det_inds = []

        for i in range(sims_per_CE):
            kalman_time = time.time()

            with torch.no_grad():
                imp_log_fname = f"{time.time()}-stg-{stage}-sim-{i}.txt"
                dn_state_list, prediction_stats = kalman_receding_horizon(end_time, 2.0, start_state, scenario,
                                                                          task_config,
                                                                          long_models, lat_models,
                                                                          imp_obs_f(imp_sampler, norm_mus, norm_stds,
                                                                                    imp_log_fname), cws)
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
                # print(rule_name, rob_val)
                rob_vals.append(rob_val)
            rep_rob_vals.append(rob_vals)

            # animate_with_predictions(solution_scenario, prediction_stats, int(end_time / task_config.dt), show=True)

            np.savetxt(os.path.join(results_folder_path, "rule_rob_vals.txt"), rep_rob_vals, fmt="%.4f")
            with open(os.path.join(results_folder_path, f"prediction_stats_{i}.pkl"), 'wb') as f:
                pickle.dump(prediction_stats, f)

            scenario_save_path = os.path.join(results_folder_path, f"kal_mpc_{i}.xml")
            fw = CommonRoadFileWriter(solution_scenario, planning_problem_set, "Craig Innes", "University of Edinburgh")
            fw.write_to_file(scenario_save_path, OverwriteExistingFile.ALWAYS)

            # Log the log-probabilities of sample trajectories under the original PEM
            # Offset by one because we assume state is known at T = 0
            # pem_states = convert_PEM_traj(T, 100, solution_scenario, norm_mus, norm_stds)[1:]
            pem_states = convert_PEM_traj(T, 100, solution_scenario, norm_mus, norm_stds)[1:]
            pem_dets, pem_long_noises, pem_lat_noises = dets_and_noise_from_stats(prediction_stats)

            ep_pem_states.append(pem_states)
            ep_det_inds.append(pem_dets)
            ep_long_noises.append(pem_long_noises)
            ep_lat_noises.append(pem_lat_noises)


            # TODO: Compare the loaded state tensors with the converted pem_states
            loaded_states = np.loadtxt(f"results/stateTensors-{imp_log_fname}")

            comparable_states = pem_states.reshape(-1, 8).detach().cpu().numpy()

            assert np.isclose(comparable_states, loaded_states).all()

            orig_pem_log_prob = log_probs_scenario_traj(pem_states, pem_dets, pem_long_noises, pem_lat_noises, det_pem,
                                                        reg_pem)
            # Log the log-probabilities of sampler trajectories under the importance sampler
            imp_sampler_log_prob = imp_log_probs_scenario_traj(pem_states, pem_dets, pem_long_noises, pem_lat_noises,
                                                               imp_sampler)


            frozen_params = torch.load("frozen_imp_sampler.pyt")
            direct_params = imp_sampler.state_dict()
            for fpv, dpv in zip(frozen_params.values(), direct_params.values()):
                assert (fpv == dpv).all()

            print("Imp Log Prob Direct: ", imp_sampler_log_prob.item())

            loaded_imp_lps = np.loadtxt(os.path.join("results", imp_log_fname))

            print("Imp Log Prob Loaded: ", np.sum(loaded_imp_lps))
            if imp_sampler_log_prob < -1500:
                print("Way too unlikely")

            ll_ratio = (orig_pem_log_prob - imp_sampler_log_prob)
            likelihood_ratio = torch.exp(ll_ratio)

        orig_pem_lps = torch.stack(
            [log_probs_scenario_traj(pem_s, d_ind, long_ns, lat_ns, det_pem, reg_pem) for pem_s, d_ind, long_ns, lat_ns
             in
             zip(ep_pem_states, ep_det_inds, ep_long_noises, ep_lat_noises)])
        lp_means.append(orig_pem_lps.mean().item())
        num_failures = torch.sum(torch.tensor(rep_rob_vals) <= 0, axis=0)
        fail_props.append(num_failures[0].item() / sims_per_CE)
        ce_one_step(ep_pem_states, ep_det_inds, ep_long_noises, ep_lat_noises, rep_rob_vals, imp_sampler, det_pem,
                    reg_pem)

    plt.plot(lp_means)
    plt.title("Original Log Probabilities")
    plt.xlabel("CE Stage")
    plt.ylabel("Average Log Probability")
    plt.show()

    plt.plot(fail_props)
    plt.title("Failure Proportion")
    plt.xlabel("CE Stage")
    plt.ylabel("Fail Prop")
    plt.show()


if __name__ == "__main__":
    run()
