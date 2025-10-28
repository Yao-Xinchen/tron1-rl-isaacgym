import math
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import random

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from .wheelfoot_flat_config import BipedCfgWF
from legged_gym.utils.helpers import class_to_dict

class BipedWF(BaseTask):
    def __init__(
        self, cfg: BipedCfgWF, sim_params, physics_engine, sim_device, headless
    ):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None

        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2
        self.group_idx = torch.arange(0, self.cfg.env.num_envs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum(time_out_env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # Initialize command pose with random offset from robot
        offset_x = (self.cfg.commands.command_pose_ranges.init_pos_x[1] -
                    self.cfg.commands.command_pose_ranges.init_pos_x[0]) * torch.rand(len(env_ids), device=self.device) + \
                   self.cfg.commands.command_pose_ranges.init_pos_x[0]
        offset_y = (self.cfg.commands.command_pose_ranges.init_pos_y[1] -
                    self.cfg.commands.command_pose_ranges.init_pos_y[0]) * torch.rand(len(env_ids), device=self.device) + \
                   self.cfg.commands.command_pose_ranges.init_pos_y[0]
        offset_yaw = (self.cfg.commands.command_pose_ranges.init_yaw[1] -
                      self.cfg.commands.command_pose_ranges.init_yaw[0]) * torch.rand(len(env_ids), device=self.device) + \
                     self.cfg.commands.command_pose_ranges.init_yaw[0]

        # Set command pose in world frame
        self.command_pose[env_ids, 0] = self.base_position[env_ids, 0] + offset_x
        self.command_pose[env_ids, 1] = self.base_position[env_ids, 1] + offset_y
        # Extract base yaw from quaternion
        base_yaw = torch.atan2(2.0 * (self.base_quat[env_ids, 3] * self.base_quat[env_ids, 2] +
                                      self.base_quat[env_ids, 0] * self.base_quat[env_ids, 1]),
                               1.0 - 2.0 * (self.base_quat[env_ids, 1]**2 + self.base_quat[env_ids, 2]**2))
        self.command_pose[env_ids, 2] = base_yaw + offset_yaw
        self.command_pose[env_ids, 2] = wrap_to_pi(self.command_pose[env_ids, 2])

        # Initialize SE3 distance tracking based on initial command pose offset
        pos_error = torch.sqrt(offset_x**2 + offset_y**2)
        orient_error = torch.abs(offset_yaw)
        self.se3_distance_ref[env_ids] = 2 * pos_error + orient_error

        self.optim_pos_distance[env_ids] = pos_error
        self.optim_orient_distance[env_ids] = orient_error
        self.pos_improvement[env_ids] = 0.0
        self.orient_improvement[env_ids] = 0.0

        self._resample_commands(env_ids)
        # self._resample_gaits(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history[env_ids] = 0
        obs_buf, _ = self.compute_group_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length)
        self.gait_indices[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["group_terrain_level"] = torch.mean(
                self.terrain_levels[self.group_idx].float()
            )
            self.extras["episode"]["group_terrain_level_stair_up"] = torch.mean(
                self.terrain_levels[self.stair_up_idx].float()
            )
        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, 1].float()
            )
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    def step(self, actions):
        self._action_clip(actions)
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands,
            self.critic_obs_buf # make sure critic_obs update in every for loop
        )
        
    def _action_clip(self, actions):
        self.actions = actions
        
    def _compute_torques(self, actions):
        pos_action = (
            torch.cat(
                (
                    actions[:, 0:3], torch.zeros_like(actions[:, 0]).view(self.num_envs, 1),
                    actions[:, 4:7], torch.zeros_like(actions[:, 0]).view(self.num_envs, 1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_pos
        )
        vel_action = (
            torch.cat(
                (
                    torch.zeros_like(actions[:, 0:3]), actions[:, 3].view(self.num_envs, 1),
                    torch.zeros_like(actions[:, 0:3]), actions[:, 7].view(self.num_envs, 1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_vel
        )
        # pd controller
        torques = self.p_gains * (pos_action + self.default_dof_pos - self.dof_pos) + self.d_gains * (vel_action - self.dof_vel)
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits )  # torque limit is lower than the torque-requiring lower bound
        return torques * self.torques_scale #notice that even send torque at torque limit , real motor may generate bigger torque that limit!!!!!!!!!!

    def post_physics_step(self):
        super().post_physics_step()
        self.wheel_lin_vel = self.foot_velocities[:, 0, :] + self.foot_velocities[:, 1, :]

        # Update command pose and commands tensor
        self._update_command_pose()
        self._update_se3_metrics()
        self._update_commands_tensor()

    def compute_group_observations(self):
        # note that observation noise need to modified accordingly !!!
        dof_list = [0,1,2,4,5,6]
        dof_pos = (self.dof_pos - self.default_dof_pos)[:,dof_list]
        # dof_pos = torch.remainder(dof_pos + self.pi, 2 * self.pi) - self.pi

        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                dof_pos * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        critic_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel, self.obs_buf), dim=-1)
        return obs_buf, critic_obs_buf
    
    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        # self._resample_gaits(env_ids)
        # self._step_contact_targets()

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = 0.1 * wrap_to_pi(self.commands[:, 3] - heading)

        if self.cfg.terrain.measure_heights or self.cfg.terrain.critic_measure_heights:
            self.measured_heights = self._get_heights()

        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

    def _resample_commands(self, env_ids):
        """Randomly sample command velocities for specified environments.

        Args:
            env_ids (List[int]): Environment ids for which new command velocities are needed
        """
        # Sample command velocities uniformly from ranges
        self.command_velocity[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1] -
            self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges["lin_vel_x"][env_ids, 0]

        self.command_velocity[env_ids, 1] = (
            self.command_ranges["lin_vel_y"][env_ids, 1] -
            self.command_ranges["lin_vel_y"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges["lin_vel_y"][env_ids, 0]

        self.command_velocity[env_ids, 2] = (
            self.command_ranges["ang_vel_yaw"][env_ids, 1] -
            self.command_ranges["ang_vel_yaw"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges["ang_vel_yaw"][env_ids, 0]

        # Sample decrease velocity for SE3 reference
        self.se3_decrease_vel[env_ids] = (
            self.cfg.commands.se3_decrease_vel_range[1] -
            self.cfg.commands.se3_decrease_vel_range[0]
        ) * torch.rand(len(env_ids), device=self.device) + \
          self.cfg.commands.se3_decrease_vel_range[0]

        # Reset SE3 distance reference based on current error
        self.position_error = self._compute_position_error()
        self.orientation_error = self._compute_orientation_error()
        self.se3_distance_ref[env_ids] = (
            2 * self.position_error[env_ids] + self.orientation_error[env_ids]
        )

        # Reset optimal tracking
        self.optim_pos_distance[env_ids] = self.position_error[env_ids]
        self.optim_orient_distance[env_ids] = self.orientation_error[env_ids]
        self.pos_improvement[env_ids] = 0.0
        self.orient_improvement[env_ids] = 0.0

        # Update commands tensor with current relative pose and new velocities
        self._update_commands_tensor()

    def _update_command_pose(self):
        """Integrate command velocity to update command pose.
        Velocity is in command pose's own frame, need to transform to world frame.
        """
        # Convert velocity from command pose frame to world frame
        cos_yaw = torch.cos(self.command_pose[:, 2])
        sin_yaw = torch.sin(self.command_pose[:, 2])

        world_vx = cos_yaw * self.command_velocity[:, 0] - sin_yaw * self.command_velocity[:, 1]
        world_vy = sin_yaw * self.command_velocity[:, 0] + cos_yaw * self.command_velocity[:, 1]

        # Integrate position and orientation
        self.command_pose[:, 0] += world_vx * self.dt
        self.command_pose[:, 1] += world_vy * self.dt
        self.command_pose[:, 2] += self.command_velocity[:, 2] * self.dt

        # Wrap yaw to [-pi, pi]
        self.command_pose[:, 2] = wrap_to_pi(self.command_pose[:, 2])

    def _compute_relative_pose(self):
        """Compute command pose relative to base in base frame.
        Returns: (x, y, yaw) in base_link frame
        """
        # Compute relative position in world frame
        rel_pos_world = self.command_pose[:, :2] - self.base_position[:, :2]

        # Extract base yaw from quaternion (z-axis rotation)
        base_yaw = torch.atan2(2.0 * (self.base_quat[:, 3] * self.base_quat[:, 2] +
                                      self.base_quat[:, 0] * self.base_quat[:, 1]),
                               1.0 - 2.0 * (self.base_quat[:, 1]**2 + self.base_quat[:, 2]**2))

        # Rotate position to base frame
        cos_base_yaw = torch.cos(-base_yaw)
        sin_base_yaw = torch.sin(-base_yaw)
        rel_x = cos_base_yaw * rel_pos_world[:, 0] - sin_base_yaw * rel_pos_world[:, 1]
        rel_y = sin_base_yaw * rel_pos_world[:, 0] + cos_base_yaw * rel_pos_world[:, 1]

        # Compute relative yaw
        rel_yaw = wrap_to_pi(self.command_pose[:, 2] - base_yaw)

        return torch.stack([rel_x, rel_y, rel_yaw], dim=-1)

    def _update_commands_tensor(self):
        """Update self.commands with relative pose and command velocity."""
        rel_pose = self._compute_relative_pose()  # (x, y, yaw) in base frame
        self.commands = torch.cat([rel_pose, self.se3_decrease_vel, self.command_velocity], dim=-1)

    def _compute_position_error(self):
        """Compute XY position error between base and command pose (in world frame)."""
        pos_error_xy = self.command_pose[:, :2] - self.base_position[:, :2]
        return torch.norm(pos_error_xy, dim=-1)

    def _compute_orientation_error(self):
        """Compute yaw orientation error between base and command pose."""
        # Extract base yaw from quaternion
        base_yaw = torch.atan2(2.0 * (self.base_quat[:, 3] * self.base_quat[:, 2] +
                                      self.base_quat[:, 0] * self.base_quat[:, 1]),
                               1.0 - 2.0 * (self.base_quat[:, 1]**2 + self.base_quat[:, 2]**2))

        # Compute angular difference
        yaw_error = wrap_to_pi(self.command_pose[:, 2] - base_yaw)
        return torch.abs(yaw_error)

    def _update_se3_metrics(self):
        """Update SE3 distance reference and improvement metrics."""
        # Compute current errors
        self.position_error = self._compute_position_error()
        self.orientation_error = self._compute_orientation_error()

        # Decrease the reference distance over time
        self.se3_distance_ref -= self.se3_decrease_vel * self.dt
        self.se3_distance_ref = torch.clamp(self.se3_distance_ref, min=0.0)

        # Track improvement (how much error decreased since last resample)
        self.pos_improvement = (self.optim_pos_distance - self.position_error).clip(min=0.0)
        self.orient_improvement = (self.optim_orient_distance - self.orientation_error).clip(min=0.0)

        # Update optimal distances (track minimum errors achieved)
        self.optim_pos_distance = torch.minimum(self.position_error, self.optim_pos_distance)
        self.optim_orient_distance = torch.minimum(self.orientation_error, self.optim_orient_distance)

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:12] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[12:20] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[20:] = 0.0  # previous actions
        return noise_vec

    def _init_buffers(self):
        super()._init_buffers()
        self.wheel_lin_vel = torch.zeros_like(self.foot_velocities)
        self.wheel_ang_vel = torch.zeros_like(self.base_ang_vel)

        # Command pose tracking: pose in world frame (x, y, yaw)
        self.command_pose = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        # Command velocity in command pose's own frame (vx, vy, vyaw)
        self.command_velocity = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        # SE3 distance tracking
        self.se3_distance_ref = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.se3_decrease_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # Tracking metrics for position and orientation errors
        self.position_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.orientation_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.optim_pos_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.optim_orient_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.pos_improvement = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.orient_improvement = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    # ------------ reward functions----------------

    def _reward_safety(self):
        # Get foot positions in base frame
        base_quat_expanded = self.base_quat.unsqueeze(1).expand(-1, 2, -1)
        base_position_expanded = self.base_position.unsqueeze(1).expand(-1, 2, -1)

        foot_position_b = quat_rotate_inverse(
            base_quat_expanded,
            self.foot_positions - base_position_expanded
        )

        # Compute base height (from root to mean foot height)
        base_height = (self.base_position[:, 2] -
                       self.foot_positions[:, :, 2].mean(dim=1) +
                       self.cfg.asset.foot_radius)

        # Nominal foot positions in base frame [left, right]
        nominal_foot_x = self.cfg.rewards.nominal_foot_x
        nominal_foot_y_left = self.cfg.rewards.nominal_foot_y
        nominal_foot_y_right = -self.cfg.rewards.nominal_foot_y

        # Compute foot position errors
        # Left foot (index 0), Right foot (index 1)
        foot_error_x = torch.abs(foot_position_b[:, :, 0] - nominal_foot_x)
        foot_error_y_left = foot_position_b[:, 0, 1] - nominal_foot_y_left
        foot_error_y_right = foot_position_b[:, 1, 1] - nominal_foot_y_right

        # Inner-eight condition: feet crossing towards center
        inner_eight_left = foot_error_y_left < 0.0  # Left foot moving right
        inner_eight_right = foot_error_y_right > 0.0  # Right foot moving left

        # Apply different tolerances for inner-eight vs normal
        foot_error_y_left_norm = torch.where(
            inner_eight_left,
            torch.abs(foot_error_y_left) / self.cfg.rewards.inner_eight_tolerance_y,
            torch.abs(foot_error_y_left) / self.cfg.rewards.foot_position_tolerance_y
        )
        foot_error_y_right_norm = torch.where(
            inner_eight_right,
            torch.abs(foot_error_y_right) / self.cfg.rewards.inner_eight_tolerance_y,
            torch.abs(foot_error_y_right) / self.cfg.rewards.foot_position_tolerance_y
        )

        # Normalize X errors
        foot_error_x_norm = foot_error_x / self.cfg.rewards.foot_position_tolerance_x

        # Sum all foot errors
        foot_pos_error_total = (
            foot_error_x_norm.sum(dim=1) +
            foot_error_y_left_norm +
            foot_error_y_right_norm
        )
        foot_pos_error_total = torch.clamp(foot_pos_error_total, max=8.0)

        # Base orientation errors from projected gravity
        base_orient_error_roll = torch.abs(self.projected_gravity[:, 1]) / self.cfg.rewards.roll_tolerance
        base_orient_error_pitch = torch.abs(self.projected_gravity[:, 0]) / self.cfg.rewards.pitch_tolerance

        # Base height error
        base_height_error = ((base_height - self.cfg.rewards.base_height_target) /
                            self.cfg.rewards.height_tolerance) ** 2

        # Compute normalized locomotion error
        normalized_loco_error = (
            foot_pos_error_total / 2.0 +  # Weight: 2.0
            base_orient_error_pitch +     # Weight: 0.5
            base_orient_error_roll +      # Weight: 0.5
            base_height_error * 2         # Weight: 1.0
        ) / 5.0  # Normalize by total weight

        # Compute locomotion safety scale with exponential kernel
        loco_safety_scale = torch.exp(-normalized_loco_error / (self.cfg.rewards.safety_std ** 2))

        # Store for potential use in other rewards (optional)
        self._loco_safety_scale = loco_safety_scale + 0.4

        return loco_safety_scale

    def _reward_track_base_position_exp(self):
        # Position error is already computed in _update_se3_metrics()
        position_error = self.position_error

        # Normal exponential term
        normal = torch.exp(-position_error / (self.cfg.rewards.track_position_std ** 2))

        # Micro enhancement (5x more sensitive for fine control)
        micro_enhancement = torch.exp(-5 * position_error / (self.cfg.rewards.track_position_std ** 2))

        # Combine and scale by safety
        reward = (normal + micro_enhancement) * 0.5 * self._loco_safety_scale

        return reward

    def _reward_track_base_orientation_exp(self):
        # Position and orientation errors already computed in _update_se3_metrics()
        position_error = self.position_error
        orientation_error = self.orientation_error

        # Position scale: reduce orientation reward when position is far
        position_scale = torch.exp(-position_error / (self.cfg.rewards.position_scale_std ** 2))

        # Normal exponential term for orientation
        normal = torch.exp(-orientation_error / (self.cfg.rewards.track_orientation_std ** 2))

        # Micro enhancement (5x more sensitive for fine control)
        micro_enhancement = torch.exp(-5 * orientation_error / (self.cfg.rewards.track_orientation_std ** 2))

        # Combine and scale by position proximity and safety
        reward = (normal + micro_enhancement) * position_scale * 0.5 * self._loco_safety_scale

        return reward

    def _reward_track_base_pb(self):
        # Use optimal distances (best achieved so far) to scale improvements
        position_scale = torch.exp(-self.optim_pos_distance / (0.5 ** 2))
        orient_scale = torch.exp(-self.optim_orient_distance / (0.5 ** 2))

        # Reward improvements (how much closer robot got since last resample)
        # Position weighted 2x more than orientation
        reward = (
            2 * self.pos_improvement * position_scale +
            self.orient_improvement * orient_scale
        ) * self._loco_safety_scale

        return reward

    def _reward_track_base_reference_exp(self):
        # Current actual SE3 error
        current_se3_error = 2 * self.position_error + self.orientation_error

        # How far off from the reference trajectory (with release tolerance)
        track_error = torch.abs(self.se3_distance_ref - current_se3_error) - \
                      self.cfg.rewards.track_reference_release_delta

        # Only penalize positive deviations (beyond release tolerance)
        track_error = torch.clamp(track_error, min=0.0)

        # Exponential reward for staying on trajectory
        reward = torch.exp(-track_error / (self.cfg.rewards.track_reference_std ** 2)) * \
                 0.5 * self._loco_safety_scale

        return reward
