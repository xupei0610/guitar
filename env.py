from typing import Callable, Optional, Union, List, Dict, Any
from collections import namedtuple
import os
from isaacgym import gymapi, gymtorch
import torch

from utils import heading_zup, axang2quat, rotatepoint, quatconj, quatmultiply

def parse_kwarg(kwargs: dict, key: str, default_val: Any):
    return kwargs[key] if key in kwargs else default_val

class DiscriminatorConfig(object):
    def __init__(self,
        key_links: Optional[List[str]]=None, ob_horizon: Optional[int]=None, 
        parent_link: Optional[str]=None, local_pos: Optional[bool]=None,
        replay_speed: Optional[str]=None, motion_file: Optional[str]=None,
        weight:Optional[float]=None
    ):
        self.motion_file = motion_file
        self.key_links = key_links
        self.local_pos = local_pos
        self.parent_link = parent_link
        self.replay_speed = replay_speed
        self.ob_horizon = ob_horizon
        self.weight = weight

DiscriminatorProperty = namedtuple("DiscriminatorProperty",
    "name key_links parent_link local_pos replay_speed ob_horizon id"
)


class Env(object):
    UP_AXIS = 2
    CHARACTER_MODEL = None
    CAMERA_POS= 0, -4.5, 2.0
    CAMERA_FOLLOWING = True

    def __init__(self,
        n_envs: int, fps: int=30, frameskip: int=2,
        episode_length: Optional[Union[Callable, int]] = 300,
        control_mode: str = "position",
        substeps: int = 2,
        compute_device: int = 0,
        graphics_device: Optional[int] = None,
        character_model: Optional[str] = None,
        **kwargs
    ):
        self.viewer = None
        assert(control_mode in ["position", "torque", "free"])
        self.frameskip = frameskip
        self.fps = fps
        self.step_time = 1./self.fps
        self.substeps = substeps
        self.control_mode = control_mode
        self.episode_length = episode_length
        self.device = torch.device(compute_device)
        self.camera_pos = self.CAMERA_POS
        self.camera_following = self.CAMERA_FOLLOWING
        if graphics_device is None:
            graphics_device = compute_device
        self.character_model = self.CHARACTER_MODEL if character_model is None else character_model
        if type(self.character_model) == str:
            self.character_model = [self.character_model]

        sim_params = self.setup_sim_params()
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)
        self.add_ground()
        self.envs, self.actors = self.create_envs(n_envs)
        n_actors_per_env = self.gym.get_actor_count(self.envs[0])
        self.actor_ids = torch.arange(n_actors_per_env * len(self.envs), dtype=torch.int32, device=self.device).view(len(self.envs), -1)
        controllable_actors = []
        for i in range(self.gym.get_actor_count(self.envs[0])):
            dof = self.gym.get_actor_dof_count(self.envs[0], i)
            if dof > 0: controllable_actors.append(i)
        self.actor_ids_having_dofs = \
            n_actors_per_env * torch.arange(len(self.envs), dtype=torch.int32, device=self.device).unsqueeze(-1) + \
            torch.tensor(controllable_actors, dtype=torch.int32, device=self.device).unsqueeze(-2)
        self.setup_action_normalizer()
        self.create_tensors()

        self.gym.prepare_sim(self.sim)

        self.root_tensor.fill_(0)
        self.gym.set_actor_root_state_tensor(self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
        )
        self.joint_tensor.fill_(0)
        self.gym.set_dof_state_tensor(self.sim,
            gymtorch.unwrap_tensor(self.joint_tensor),
        )
        self.refresh_tensors()
        self.train()
        self.viewer_pause = False
        self.viewer_advance = False
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        base_pos = self.root_pos[tar_env].cpu()
        self.cam_target = gymapi.Vec3(*self.vector_up(0.89, [base_pos[0], base_pos[1], base_pos[2]]))

        self.simulation_step = 0
        self.lifetime = torch.zeros(len(self.envs), dtype=torch.int64, device=self.device)
        self.done = torch.ones(len(self.envs), dtype=torch.bool, device=self.device)
        self.info = dict(lifetime=self.lifetime)

        self.act_dim = self.action_scale.size(-1)
        self.ob_dim = self.observe().size(-1)
        self.rew_dim = self.reward().size(-1)

        for i in range(self.gym.get_actor_count(self.envs[0])):
            rigid_body = self.gym.get_actor_rigid_body_dict(self.envs[0], i)
            print("Links", sorted(rigid_body.items(), key=lambda x:x[1]), len(rigid_body))
            dof = self.gym.get_actor_dof_dict(self.envs[0], i)
            print("Joints", sorted(dof.items(), key=lambda x:x[1]), len(dof))


    def __del__(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        if hasattr(self, "sim"):
            self.gym.destroy_sim(self.sim)

    def eval(self):
        self.training = False
        
    def train(self):
        self.training = True

    def vector_up(self, val: float, base_vector=None):
        if base_vector is None:
            base_vector = [0., 0., 0.]
        base_vector[self.UP_AXIS] = val
        return base_vector
    
    def setup_sim_params(self, physx_params=dict()):
        p = gymapi.SimParams()
        p.dt = self.step_time/self.frameskip
        p.substeps = self.substeps
        p.up_axis = gymapi.UP_AXIS_Z if self.UP_AXIS == 2 else gymapi.UP_AXIS_Y
        p.gravity = gymapi.Vec3(*self.vector_up(-9.81))
        p.num_client_threads = 0
        p.physx.num_threads = 4
        p.physx.solver_type = 1
        p.physx.num_subscenes = 4  # works only for CPU 
        p.physx.num_position_iterations = 4
        p.physx.num_velocity_iterations = 0
        p.physx.contact_offset = 0.01
        p.physx.rest_offset = 0.0
        p.physx.bounce_threshold_velocity = 0.2
        p.physx.max_depenetration_velocity = 10.0
        p.physx.default_buffer_size_multiplier = 5.0
        p.physx.max_gpu_contact_pairs = 8*1024*1024
        # p.physx.default_buffer_size_multiplier = 4
        # FIXME IsaacGym Pr4 will provide unreliable results when collecting from all substeps
        p.physx.contact_collection = \
            gymapi.ContactCollection(gymapi.ContactCollection.CC_LAST_SUBSTEP) 
        #gymapi.ContactCollection(gymapi.ContactCollection.CC_ALL_SUBSTEPS)
        for k, v in physx_params.items():
            setattr(p.physx, k, v)
        p.use_gpu_pipeline = True # force to enable GPU
        p.physx.use_gpu = True
        return p

    def add_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(*self.vector_up(1.0))
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

    def add_actor(self, env, group):
        pass

    def create_envs(self, n: int, start_height=0.89, actuate_all_dofs=True, asset_options=dict()):
        if self.control_mode == "position":
            control_mode = gymapi.DOF_MODE_POS
        elif self.control_mode == "torque":
            control_mode = gymapi.DOF_MODE_EFFORT
        else:
            control_mode = gymapi.DOF_MODE_NONE

        envs, actors = [], []
        env_spacing = 3

        actor_asset = []
        actuated_dof = []
        for character_model in self.character_model:
            asset_opt = gymapi.AssetOptions()
            asset_opt.angular_damping = 0.01
            asset_opt.max_angular_velocity = 100.0
            asset_opt.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
            for k, v in asset_options.items():
                setattr(asset_opt, k, v)
            asset = self.gym.load_asset(self.sim,
                os.path.abspath(os.path.dirname(character_model)),
                os.path.basename(character_model),
                asset_opt)
            actor_asset.append(asset)
            if actuate_all_dofs:
                actuated_dof.append([i for i in range(self.gym.get_asset_dof_count(asset))])
            else:
                actuators = []
                for i in range(self.gym.get_asset_actuator_count(asset)):
                    name = self.gym.get_asset_actuator_joint_name(asset, i)
                    actuators.append(self.gym.find_asset_dof_index(asset, name))
                    if actuators[-1] == -1:
                        raise ValueError("Failed to find joint with name {}".format(name))
                actuated_dof.append(sorted(actuators) if len(actuators) else [])

        spacing_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0)
        spacing_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        n_envs_per_row = int(n**0.5)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.vector_up(start_height))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.actuated_dof = []
        for i in range(n):
            env = self.gym.create_env(self.sim, spacing_lower, spacing_upper, n_envs_per_row)
            for aid, (asset, dofs) in enumerate(zip(actor_asset, actuated_dof)):
                actor = self.gym.create_actor(env, asset, start_pose, "actor{}_{}".format(i, aid), i, -1, 0)
                dof_prop = self.gym.get_asset_dof_properties(asset)
                for k in range(len(dof_prop)):
                    if k in dofs:
                        dof_prop[k]["driveMode"] = control_mode
                    else:
                        dof_prop[k]["stiffness"] = 0
                        dof_prop[k]["damping"] = 0
                self.gym.set_actor_dof_properties(env, actor, dof_prop)
                if i == n-1:
                    actors.append(actor)
                    self.actuated_dof.append(dofs)
            self.add_actor(env, i)
            envs.append(env)
        return envs, actors

    def render(self):
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        base_pos = self.root_pos[tar_env].cpu()
        cam_pos = gymapi.Vec3(*self.vector_up(self.camera_pos[2], 
            [base_pos[0]+self.camera_pos[0], base_pos[1]+self.camera_pos[1], base_pos[2]+self.camera_pos[1]]))
        self.gym.viewer_camera_look_at(self.viewer, self.envs[tar_env], cam_pos, self.cam_target)
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "TOGGLE_CAMERA_FOLLOWING")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "TOGGLE_PAUSE")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "SINGLE_STEP_ADVANCE")
    
    def update_viewer(self):
        self.gym.poll_viewer_events(self.viewer)
        for event in self.gym.query_viewer_action_events(self.viewer):
            if event.action == "QUIT" and event.value > 0:
                exit()
            if event.action == "TOGGLE_CAMERA_FOLLOWING" and event.value > 0:
                self.camera_following = not self.camera_following
            if event.action == "TOGGLE_PAUSE" and event.value > 0:
                self.viewer_pause = not self.viewer_pause
            if event.action == "SINGLE_STEP_ADVANCE" and event.value > 0:
                self.viewer_advance = not self.viewer_advance
        if self.camera_following: self.update_camera()
        self.gym.step_graphics(self.sim)

    def update_camera(self):
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, self.envs[tar_env])
        dx, dy = cam_trans.p.x - self.cam_target.x, cam_trans.p.y - self.cam_target.y
        base_pos = self.root_pos[tar_env].cpu()
        cam_pos = gymapi.Vec3(base_pos[0]+dx, base_pos[1]+dy, cam_trans.p.z)
        self.cam_target = gymapi.Vec3(base_pos[0], base_pos[1], 1.0)
        self.gym.viewer_camera_look_at(self.viewer, self.envs[tar_env], cam_pos, self.cam_target)

    def refresh_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def create_tensors(self):
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_tensor = gymtorch.wrap_tensor(root_tensor)
        self.root_tensor = root_tensor.view(len(self.envs), -1, 13)

        num_links = self.gym.get_env_rigid_body_count(self.envs[0])
        link_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        link_tensor = gymtorch.wrap_tensor(link_tensor)
        self.link_tensor = link_tensor.view(len(self.envs), num_links, -1)

        num_dof = self.gym.get_env_dof_count(self.envs[0])
        joint_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        joint_tensor = gymtorch.wrap_tensor(joint_tensor)
        self.joint_tensor = joint_tensor.view(len(self.envs), num_dof, -1)  # n_envs x n_dof x 2

        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self.contact_force_tensor = contact_force_tensor.view(len(self.envs), -1, 3)

        if self.actuated_dof.size(-1) == self.joint_tensor.size(1):
            self.action_tensor = None
        else:
            self.action_tensor = torch.zeros_like(self.joint_tensor[..., 0])

    def setup_action_normalizer(self):
        actuated_dof = []
        dof_cnts = 0
        action_lower, action_upper = [], []
        action_scale = []
        for i, dofs in zip(range(self.gym.get_actor_count(self.envs[0])), self.actuated_dof):
            actor = self.gym.get_actor_handle(self.envs[0], i)
            dof_prop = self.gym.get_actor_dof_properties(self.envs[0], actor)
            if len(dof_prop) < 1: continue
            if self.control_mode == "torque":
                action_lower.extend([-dof_prop["effort"][j] for j in dofs])
                action_upper.extend([dof_prop["effort"][j] for j in dofs])
                action_scale.extend([1]*len(dofs))
            else: # self.control_mode == "position":
                action_lower.extend([min(dof_prop["lower"][j], dof_prop["upper"][j]) for j in dofs])
                action_upper.extend([max(dof_prop["lower"][j], dof_prop["upper"][j]) for j in dofs])
                action_scale.extend([2]*len(dofs))
            for j in dofs:
                actuated_dof.append(dof_cnts+j)
            dof_cnts += len(dof_prop)
        action_offset = 0.5 * np.add(action_upper, action_lower)
        action_scale *= 0.5 * np.subtract(action_upper, action_lower)
        self.action_offset = torch.tensor(action_offset, dtype=torch.float32, device=self.device)
        self.action_scale = torch.tensor(action_scale, dtype=torch.float32, device=self.device)
        self.actuated_dof = torch.tensor(actuated_dof, dtype=torch.int64, device=self.device)

    def process_actions(self, actions):
        a = actions*self.action_scale + self.action_offset
        if self.action_tensor is None:
            return a
        self.action_tensor[:, self.actuated_dof] = a
        return self.action_tensor

    def reset(self):
        self.lifetime.zero_()
        self.done.fill_(True)
        self.info = dict(lifetime=self.lifetime)
        self.request_quit = False
        self.obs = None

    def reset_done(self):
        if not self.viewer_pause:
            env_ids = torch.nonzero(self.done).view(-1)
            if len(env_ids):
                self.reset_envs(env_ids)
                if len(env_ids) == len(self.envs) or self.obs is None:
                    self.obs = self.observe()
                else:
                    self.obs[env_ids] = self.observe(env_ids)
        return self.obs, self.info
    
    def reset_envs(self, env_ids):
        ref_root_tensor, ref_link_tensor, ref_joint_tensor = self.init_state(env_ids)

        self.root_tensor[env_ids] = ref_root_tensor.unsqueeze_(1).repeat(1, self.root_tensor.size(1), 1)
        self.link_tensor[env_ids] = ref_link_tensor
        if self.action_tensor is None:
            self.joint_tensor[env_ids] = ref_joint_tensor
        else:
            self.joint_tensor[env_ids.unsqueeze(-1), self.actuated_dof] = ref_joint_tensor

        actor_ids = self.actor_ids[env_ids].flatten()
        n_actor_ids = len(actor_ids)
        actor_ids = gymtorch.unwrap_tensor(actor_ids)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            actor_ids, n_actor_ids
        )
        actor_ids = self.actor_ids_having_dofs[env_ids].flatten()
        n_actor_ids = len(actor_ids)
        actor_ids = gymtorch.unwrap_tensor(actor_ids)
        self.gym.set_dof_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.joint_tensor),
            actor_ids, n_actor_ids
        )

        self.lifetime[env_ids] = 0

    def do_simulation(self):
        for _ in range(self.frameskip):
            self.gym.simulate(self.sim)
        self.simulation_step += 1

    def step(self, actions):
        if not self.viewer_pause or self.viewer_advance:
            self.apply_actions(actions)
            self.do_simulation()
            self.refresh_tensors()
            self.lifetime += 1
            if self.viewer is not None:
                self.gym.fetch_results(self.sim, True)
                self.viewer_advance = False

        if self.viewer is not None:
            self.update_viewer()
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)    # sync to simulation dt

        rewards = self.reward()
        terminate = self.termination_check()                    # N
        if self.viewer_pause:
            overtime = None
        else:
            overtime = self.overtime_check()
        if torch.is_tensor(overtime):
            self.done = torch.logical_or(overtime, terminate)
        else:
            self.done = terminate
        self.info["terminate"] = terminate
        self.obs = self.observe()
        self.request_quit = False if self.viewer is None else self.gym.query_viewer_has_closed(self.viewer)
        return self.obs, rewards, self.done, self.info

    def apply_actions(self, actions):
        actions = self.process_actions(actions)
        if self.control_mode == "position":
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_position_target_tensor(self.sim, actions)
        elif self.control_mode == "torque":
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_actuation_force_tensor(self.sim, actions)
        else:
            actions = torch.stack((actions, torch.zeros_like(actions)), -1)
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_state_tensor(self.sim, actions)

    def init_state(self, env_ids):
        pass
    
    def observe(self, env_ids=None):
        pass
    
    def overtime_check(self):
        if self.episode_length is None: return None
        if callable(self.episode_length):
            return self.lifetime >= self.episode_length(self.simulation_step).to(self.lifetime.device)
        return self.lifetime >= self.episode_length

    def termination_check(self):
        return torch.zeros(len(self.envs), dtype=torch.bool, device=self.device)

    def reward(self):
        return torch.ones((len(self.envs), 0), dtype=torch.float32, device=self.device)


from ref_motion import ReferenceMotion
import numpy as np


class ICCGANHumanoid(Env):

    CHARACTER_MODEL = "assets/humanoid.xml"
    CONTACTABLE_LINKS = ["right_foot", "left_foot"]
    UP_AXIS = 2

    GOAL_DIM = 0
    GOAL_REWARD_WEIGHT = None
    ENABLE_GOAL_TIMER = False
    GOAL_TENSOR_DIM = None

    OB_HORIZON = 4
    KEY_LINKS = None    # All links
    PARENT_LINK = None  # root link


    def __init__(self, *args,
        motion_file: str,
        discriminators: Dict[str, DiscriminatorConfig],
    **kwargs):
        contactable_links = parse_kwarg(kwargs, "contactable_links", self.CONTACTABLE_LINKS)
        goal_reward_weight = parse_kwarg(kwargs, "goal_reward_weight", self.GOAL_REWARD_WEIGHT)
        self.enable_goal_timer = parse_kwarg(kwargs, "enable_goal_timer", self.ENABLE_GOAL_TIMER)
        self.goal_tensor_dim = parse_kwarg(kwargs, "goal_tensor_dim", self.GOAL_TENSOR_DIM)
        self.ob_horizon = parse_kwarg(kwargs, "ob_horizon", self.OB_HORIZON)
        self.key_links = parse_kwarg(kwargs, "key_links", self.KEY_LINKS)
        self.parent_link = parse_kwarg(kwargs, "parent_link", self.PARENT_LINK)
        super().__init__(*args, **kwargs)

        n_envs = len(self.envs)
        n_links = self.char_link_tensor.size(1)
        n_dofs = self.char_joint_tensor.size(1)
        
        if contactable_links is None:
            self.contactable_links = None
        elif contactable_links:
            contact = np.zeros((n_envs, n_links), dtype=bool)
            for actor in self.actors:
                for link in contactable_links:
                    lid = self.gym.find_actor_rigid_body_handle(self.envs[0], actor, link)
                    # assert(lid >= 0), "Unrecognized contactable link {}".format(link)
                    if lid >= 0:
                        contact[:, lid] = True
            self.contactable_links = torch.tensor(contact).to(self.contact_force_tensor.device)
        else:
            self.contactable_links = False

        if goal_reward_weight is not None:
            reward_weights = torch.empty((self.rew_dim), dtype=torch.float32, device=self.device)
            if not hasattr(goal_reward_weight, "__len__"):
                goal_reward_weight = [goal_reward_weight]
            assert self.rew_dim == len(goal_reward_weight), "{} vs {}".format(self.rew_dim, len(goal_reward_weight))
            for i, w in zip(range(self.rew_dim), goal_reward_weight):
                reward_weights[i] = w
        elif self.rew_dim:
            goal_reward_weight = []
            assert(self.rew_dim == len(goal_reward_weight))

        n_comp = len(discriminators) + self.rew_dim
        if n_comp > 1:
            self.reward_weights = torch.zeros((n_comp,), dtype=torch.float32, device=self.device)
            weights = [disc.weight for _, disc in discriminators.items() if disc.weight is not None]
            total_weights = sum(weights) if weights else 0
            assert(total_weights <= 1), "Discriminator weights must not be greater than 1."
            n_unassigned = len(discriminators) - len(weights)
            rem = 1 - total_weights
            for disc in discriminators.values():
                if disc.weight is None:
                    disc.weight = rem / n_unassigned
                elif n_unassigned == 0:
                    disc.weight /= total_weights
        else:
            self.reward_weights = None

        self.discriminators = dict()
        max_ob_horizon = self.ob_horizon+1
        for i, (id, config) in enumerate(discriminators.items()):
            if config.key_links is None:
                key_links = None
            else:
                key_links = []
                for link in config.key_links:
                    for actor in self.actors:
                        lid = self.gym.find_actor_rigid_body_handle(self.envs[0], actor, link)
                        if lid != -1:
                            key_links.append(lid)
                            break
                    assert lid != -1, "Unfound link {}".format(link)
                key_links = sorted(key_links)
            if config.parent_link is None:
                parent_link = None
            else:
                for j in self.actors:
                    parent_link = self.gym.find_actor_rigid_body_handle(self.envs[0], j, config.parent_link)
                    if parent_link != -1: break

            assert(key_links is None or all(lid >= 0 for lid in key_links))
            assert(parent_link is None or parent_link >= 0)
            
            if config.motion_file is None:
                config.motion_file = motion_file
            if config.ob_horizon is None:
                config.ob_horizon = self.ob_horizon+1
            self.discriminators[id] = DiscriminatorProperty(
                name = id,
                key_links = key_links,
                parent_link = parent_link,
                local_pos = config.local_pos,
                replay_speed = config.replay_speed,
                ob_horizon = config.ob_horizon,
                id=i
            )
            if self.reward_weights is not None:
                self.reward_weights[i] = config.weight
            max_ob_horizon = max(max_ob_horizon, config.ob_horizon)

        if max_ob_horizon != self.state_hist.size(0):
            self.state_hist = torch.zeros((max_ob_horizon, *self.state_hist.shape[1:]),
                dtype=self.root_tensor.dtype, device=self.device)
        if self.rew_dim > 0:
            if self.rew_dim > 1:
                self.reward_weights *= (1-reward_weights.sum(dim=-1, keepdim=True))
            else:
                self.reward_weights *= (1-reward_weights)
            self.reward_weights[-self.rew_dim:] = reward_weights
        else:
            self.reward_weights = None
        

        self.info["ob_seq_lens"] = torch.zeros_like(self.lifetime)  # dummy result
        self.goal_dim = self.GOAL_DIM
        if not hasattr(self.goal_dim, "__getitem__"):
            self.goal_dim = [self.goal_dim, self.goal_dim]
        g = max(self.goal_dim)
        self.state_dim = (self.ob_dim-g)//self.ob_horizon
        if self.discriminators:
            self.info["disc_obs"] = self.observe_disc(self.state_hist)  # dummy result
            self.info["disc_obs_expert"] = self.info["disc_obs"]        # dummy result
            self.disc_dim = {
                name: ob.size(-1)
                for name, ob in self.info["disc_obs"].items()
            }
        else:
            self.disc_dim = {}

        self.ref_motion = ReferenceMotion(motion_file=motion_file, character_model=self.character_model,
            key_links=np.arange(n_links), device=self.device)
        
        self.sampling_workers = []
        self.disc_ref_motion = {}
        import torch.multiprocessing as mp
        mp.set_start_method("spawn")
        manager = mp.Manager()
        seed = np.random.get_state()[1][0]
        for n, config in self.discriminators.items():
            q = manager.Queue(maxsize=1)
            self.disc_ref_motion[n] = q
            key_links_ref = list(range(n_links)) if config.key_links is None else config.key_links
            if config.parent_link is None:
                parent_link = None
                key_links = None
            else:
                if config.parent_link in key_links_ref:
                    key_links = None
                else:
                    key_links = list(range(1, len(key_links_ref)+1))
                    key_links_ref = [config.parent_link] + key_links_ref
                parent_link = key_links_ref.index(config.parent_link)
            p = mp.Process(target=self.__class__.ref_motion_sample, args=(q,
                seed+1+config.id, self.step_time, len(self.envs), config.ob_horizon, key_links, parent_link, config.local_pos, config.replay_speed,
                dict(motion_file=discriminators[n].motion_file, character_model=self.character_model,
                    key_links=key_links_ref, device=self.device
                )
            ))
            p.start()
            self.sampling_workers.append(p)

        self.real_samples = [{n:None for n in self.disc_ref_motion.keys()} for _ in range(128)]
        for n, q in self.disc_ref_motion.items():
            for i, v in enumerate(q.get()):
                self.real_samples[i][n] = v.to(self.device)
    
    def __del__(self):
        if hasattr(self, "sampling_workers"):
            for p in self.sampling_workers:
                p.terminate()
            for p in self.sampling_workers:
                p.join()
        super().__del__()
    
    @staticmethod
    def ref_motion_sample(queue, seed, step_time, n_inst, ob_horizon, key_links, parent_link, local_pos, replay_speed, kwargs):
        np.random.seed(seed)
        torch.set_num_threads(1)
        lib = ReferenceMotion(**kwargs)
        if replay_speed is not None:
            replay_speed = eval(replay_speed)
        while True:
            obs = []
            for _ in range(128):
                if replay_speed is None:
                    dt = step_time
                else:
                    dt = step_time * replay_speed(n_inst)
                motion_ids, motion_times0 = lib.sample(n_inst, truncate_time=dt*(ob_horizon-1))
                motion_ids = np.tile(motion_ids, ob_horizon)
                motion_times = np.concatenate((motion_times0, *[motion_times0+dt*i for i in range(1, ob_horizon)]))
                root_tensor, link_tensor = lib.state(motion_ids, motion_times, with_joint_tensor=False)
                samples = torch.cat((
                    root_tensor, link_tensor.view(root_tensor.size(0), -1)
                ), -1).view(ob_horizon, n_inst, -1)
                ob = observe_iccgan(samples, None, key_links, parent_link, include_velocity=False, local_pos=local_pos)
                obs.append(ob.cpu())
            try:
                queue.put(obs)
            except EOFError:
                break

    def reset_done(self):
        obs, info = super().reset_done()
        info["ob_seq_lens"] = self.ob_seq_lens
        return obs, info
    
    def reset(self):
        if self.goal_tensor is not None:
            self.goal_tensor.zero_()
            if self.goal_timer is not None: self.goal_timer.zero_()
        super().reset()

    def reset_envs(self, env_ids):
        super().reset_envs(env_ids)
        self.reset_goal(env_ids)
        
    def reset_goal(self, env_ids):
        pass
    
    def step(self, actions):
        obs, rews, dones, info = super().step(actions)
        if self.discriminators:
            info["disc_obs"] = self.observe_disc(self.state_hist)
            info["disc_obs_expert"] = self.fetch_real_samples()
        return obs, rews, dones, info

    def overtime_check(self):
        if self.goal_timer is not None:
            self.goal_timer -= 1
            env_ids = torch.nonzero(self.goal_timer <= 0).view(-1)
            if len(env_ids) > 0: self.reset_goal(env_ids)
        return super().overtime_check()

    def termination_check(self):
        if self.contactable_links is None:
            return torch.zeros_like(self.done)
        masked_contact = self.char_contact_force_tensor.clone()
        if self.contactable_links is not False:
            masked_contact[self.contactable_links] = 0          # N x n_links x 3

        contacted = torch.any(masked_contact > 1., dim=-1)  # N x n_links
        too_low = self.link_pos[..., self.UP_AXIS] < 0.15    # N x n_links

        terminate = torch.any(torch.logical_and(contacted, too_low), -1)    # N x
        terminate *= (self.lifetime > 1)
        return terminate

    def init_state(self, env_ids):
        motion_ids, motion_times = self.ref_motion.sample(len(env_ids))
        return self.ref_motion.state(motion_ids, motion_times)

    def create_tensors(self):
        self.blender = []

        super().create_tensors()
        n_dofs = sum([self.gym.get_actor_dof_count(self.envs[0], actor) for actor in self.actors])
        n_links = sum([self.gym.get_actor_rigid_body_count(self.envs[0], actor) for actor in self.actors])
        self.root_pos, self.root_orient = self.root_tensor[:, 0, :3], self.root_tensor[:, 0, 3:7]
        self.root_lin_vel, self.root_ang_vel = self.root_tensor[:, 0, 7:10], self.root_tensor[:, 0, 10:13]
        self.char_root_tensor = self.root_tensor[:, 0]

        if self.link_tensor.size(1) > n_links:
            self.link_pos, self.link_orient = self.link_tensor[:, :n_links, :3], self.link_tensor[:, :n_links, 3:7]
            self.link_lin_vel, self.link_ang_vel = self.link_tensor[:, :n_links, 7:10], self.link_tensor[:, :n_links, 10:13]
            self.char_link_tensor = self.link_tensor[:, :n_links]
        else:
            self.link_pos, self.link_orient = self.link_tensor[..., :3], self.link_tensor[..., 3:7]
            self.link_lin_vel, self.link_ang_vel = self.link_tensor[..., 7:10], self.link_tensor[..., 10:13]
            self.char_link_tensor = self.link_tensor
        if self.joint_tensor.size(1) > n_dofs:
            self.joint_pos, self.joint_vel = self.joint_tensor[:, :n_dofs, 0], self.joint_tensor[:, :n_dofs, 1]
            self.char_joint_tensor = self.joint_tensor[:, :n_dofs]
        else:
            self.joint_pos, self.joint_vel = self.joint_tensor[..., 0], self.joint_tensor[..., 1]
            self.char_joint_tensor = self.joint_tensor
        
        self.char_contact_force_tensor = self.contact_force_tensor[:, :n_links]
    
        self.state_hist = torch.empty((self.ob_horizon+1, len(self.envs), 13 + n_links*13),
            dtype=self.root_tensor.dtype, device=self.device)
        

        if self.key_links is None:
            self.key_links = None
        else:
            key_links = []
            for link in self.key_links:
                for actor in self.actors:
                    lid = self.gym.find_actor_rigid_body_handle(self.envs[0], actor, link)
                    if lid != -1:
                        key_links.append(lid)
                        break
                assert lid != -1, "Unrecognized key link {}".format(link)
            self.key_links = key_links
        if self.parent_link is None:
            self.parent_link = None
        else:
            for actor in self.actors:
                lid = self.gym.find_actor_rigid_body_handle(self.envs[0], actor, self.parent_link)
                if lid != -1:
                    parent_link = lid
                    break
            assert lid != -1, "Unrecognized parent link {}".format(self.parent_link)
            self.parent_link = parent_link
        if self.goal_tensor_dim:
            try:
                self.goal_tensor = [
                    torch.zeros((len(self.envs), dim), dtype=self.root_tensor.dtype, device=self.device)
                    for dim in self.goal_tensor_dim
                ]
            except TypeError:
                self.goal_tensor = torch.zeros((len(self.envs), self.goal_tensor_dim), dtype=self.root_tensor.dtype, device=self.device)
        else:
            self.goal_tensor = None
        self.goal_timer = torch.zeros((len(self.envs), ), dtype=torch.int32, device=self.device) if self.enable_goal_timer else None


    def observe(self, env_ids=None):
        self.ob_seq_lens = self.lifetime+1
        n_envs = len(self.envs)
        if env_ids is None or len(env_ids) == n_envs:
            self.state_hist[:-1] = self.state_hist[1:].clone()
            self.state_hist[-1] = torch.cat((
                self.char_root_tensor, self.char_link_tensor.view(n_envs, -1)
            ), -1)
            env_ids = None
        else:
            n_envs = len(env_ids)
            self.state_hist[:-1, env_ids] = self.state_hist[1:, env_ids].clone()
            self.state_hist[-1, env_ids] = torch.cat((
                self.char_root_tensor[env_ids], self.char_link_tensor[env_ids].view(n_envs, -1)
            ), -1)
        return self._observe(env_ids)
    
    def _observe(self, env_ids):
        if env_ids is None:
            return observe_iccgan(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens, self.key_links, self.parent_link
            ).flatten(start_dim=1)
        else:
            return observe_iccgan(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids], self.key_links, self.parent_link
            ).flatten(start_dim=1)
    
    def observe_disc(self, state):
        seq_len = self.info["ob_seq_lens"]+1
        res = dict()
        if torch.is_tensor(state):
            # fake
            for id, disc in self.discriminators.items():
                res[id] = observe_iccgan(state[-disc.ob_horizon:], seq_len, disc.key_links, disc.parent_link,
                    include_velocity=False, local_pos=disc.local_pos)
            return res
        else:
            # real
            seq_len_ = dict()
            for disc_name, s in state.items():
                disc = self.discriminators[disc_name]
                res[disc_name] = observe_iccgan(s[-disc.ob_horizon:], seq_len, disc.key_links, disc.parent_link,
                    include_velocity=False, local_pos=disc.local_pos)
                seq_len_[disc_name] = seq_len
            return res, seq_len_

    def fetch_real_samples(self):
        if not self.real_samples:
            self.real_samples = [{n: None for n in self.disc_ref_motion.keys()} for _ in range(128)]
            for n, q in self.disc_ref_motion.items():
                for i, v in enumerate(q.get()):
                    self.real_samples[i][n] = v.to(self.device)
        return self.real_samples.pop()

#@torch.jit.script
def observe_iccgan(state_hist: torch.Tensor, seq_len: Optional[torch.Tensor]=None,
    key_links: Optional[List[int]]=None, parent_link: Optional[int]=None,
    include_velocity: bool=True, local_pos: Optional[bool]=None
):
    # state_hist: L x N x (1+N_links) x 13

    UP_AXIS = 2
    n_hist = state_hist.size(0)
    n_inst = state_hist.size(1)

    root_tensor = state_hist[..., :13]
    link_tensor = state_hist[...,13:].view(n_hist, n_inst, -1, 13)
    if key_links is None:
        link_pos, link_orient = link_tensor[...,:3], link_tensor[...,3:7]
    else:
        link_pos, link_orient = link_tensor[:,:,key_links,:3], link_tensor[:,:,key_links,3:7]

    if parent_link is None:
        if local_pos is True:
            origin = root_tensor[:,:, :3]          # N x 3
            orient = root_tensor[:,:,3:7]          # N x 4
        else:
            origin = root_tensor[-1,:, :3]          # N x 3
            orient = root_tensor[-1,:,3:7]          # N x 4

        heading = heading_zup(orient)               # N
        up_dir = torch.zeros_like(origin)
        up_dir[..., UP_AXIS] = 1                    # N x 3
        orient_inv = axang2quat(up_dir, -heading)   # N x 4
        orient_inv = orient_inv.view(1, -1, 1, 4)   # 1 x N x 1 x 4

        origin = origin.clone()
        origin[..., UP_AXIS] = 0                    # N x 3
        origin.unsqueeze_(-2)                       # N x 1 x 3
    else:
        if local_pos is True or local_pos is None:
            origin = link_tensor[:,:, parent_link, :3]  # L x N x 3
            orient = link_tensor[:,:, parent_link,3:7]  # L x N x 4
        else:
            origin = link_tensor[-1,:, parent_link, :3]  # N x 3
            orient = link_tensor[-1,:, parent_link,3:7]  # N x 4
        orient_inv = quatconj(orient)                # L x N x 4 
        orient_inv = orient.view(-1, n_inst, 1, 4)   # L x N x 1 x 4  or 1 x N x 1 x 4
        origin = origin.unsqueeze(-2)                # (L x) N x 1 x 3

    ob_link_pos0 = link_pos - origin                                     # L x N x n_links x 3 
    ob_link_pos = rotatepoint(orient_inv, ob_link_pos0)
    ob_link_orient = quatmultiply(orient_inv, link_orient)              # L x N x n_links x 4

    if include_velocity:
        if key_links is None:
            link_lin_vel, link_ang_vel = link_tensor[...,7:10], link_tensor[...,10:13]
        else:
            link_lin_vel, link_ang_vel = link_tensor[:,:,key_links,7:10], link_tensor[:,:,key_links,10:13]
        ob_link_lin_vel = rotatepoint(orient_inv, link_lin_vel)         # L x N x n_links x 3
        ob_link_ang_vel = rotatepoint(orient_inv, link_ang_vel)         # L x N x n_links x 3
        ob = torch.cat((ob_link_pos, ob_link_orient,
            ob_link_lin_vel, ob_link_ang_vel), -1)                      # L x N x n_links x 13
    else:
        ob = torch.cat((ob_link_pos, ob_link_orient), -1)               # L x N x n_links x 7
    ob = ob.view(n_hist, n_inst, -1)                                    # L x N x (n_links x 7 or 13)

    ob1 = ob.permute(1, 0, 2)                                           # N x L x (n_links x 7 or 13)
    if seq_len is None: return ob1

    ob2 = torch.zeros_like(ob1)
    arange = torch.arange(n_hist, dtype=seq_len.dtype, device=seq_len.device).unsqueeze_(0)
    seq_len_ = seq_len.unsqueeze(1)
    mask1 = arange > (n_hist-1) - seq_len_
    mask2 = arange < seq_len_
    ob2[mask2] = ob1[mask1]
    return ob2


import yaml
import json
from utils import dist2_p2seg, closest_seg2seg

class ICCGANHandBase(ICCGANHumanoid):
    NOTE_FILE = None

    CONTACTABLE_LINKS = [] # no links contactable to the floor
    UP_AXIS = 2

    GOAL_DIM = None
    GOAL_TENSOR_DIM = None
    ENABLE_GOAL_TIMER = True

    # time to resampling the target note score in term of number of notes
    GOAL_SAMPLING_RANGE = (10, 20)

    N_STRINGS = 6
    N_FRETS = 22

    CAMERA_POS = 0, .5, 0.89
    CAMERA_FOLLOWING = False

    OB_HORIZON = 2
    # take n future notes as the goal input
    GOAL_HORIZON = 5
    # number of empty frames before resampling new target notes
    GRACE_PERIOD = 5


    def __init__(self, *args, **kwargs):
        self.n_frets = parse_kwarg(kwargs, "n_frets", self.N_FRETS)
        self.n_strings = parse_kwarg(kwargs, "n_strings", self.N_STRINGS)
        self.note_file = parse_kwarg(kwargs, "note_file", self.NOTE_FILE)
        goal_sampling_range = parse_kwarg(kwargs, "goal_sampling_range", self.GOAL_SAMPLING_RANGE)
        self.goal_sampling_range = min(goal_sampling_range), max(goal_sampling_range)

        self.random_note_sampling = parse_kwarg(kwargs, "random_note_sampling", True)
        self.random_pitch_rate = parse_kwarg(kwargs, "random_pitch_rate", 0.5)
        self.random_bpm_rate = parse_kwarg(kwargs,  "random_bpm_rate", 0.5)
        self.merge_repeated_notes = parse_kwarg(kwargs,  "merge_repeated_notes", True)

        self.goal_horizon = parse_kwarg(kwargs, "goal_horizon", self.GOAL_HORIZON)
        assert self.goal_horizon < min(self.goal_sampling_range)
        self.grace_period = parse_kwarg(kwargs, "grace_period", self.GRACE_PERIOD)

        self.GOAL_DIM = self.get_goal_dim()
        self.GOAL_TENSOR_DIM = self.get_goal_tensor_dim()

        if "fps" not in kwargs: kwargs["fps"] = 60
        if "substeps" not in kwargs: kwargs["substeps"] = 4
        if "frameskip" not in kwargs: kwargs["frameskip"] = 1
        super().__init__(*args, **kwargs)


    def get_goal_dim(self):
        return (self.n_strings+1) * self.goal_horizon

    def get_goal_tensor_dim(self):
        return self.n_strings * self.goal_horizon
    
    def setup_sim_params(self):
        return super().setup_sim_params(physx_params=dict(
            max_depenetration_velocity = 1
        ))

    def create_envs(self, n: int):
        return super().create_envs(n,
            asset_options=dict(
                fix_base_link = True,
                disable_gravity = True,
                thickness = 0.0003
            )
        )

    def add_actor(self, env, i):
        for aid in range(self.gym.get_actor_count(env)):
            actor = self.gym.get_actor_handle(env, aid)
            rb_shape_props = self.gym.get_actor_rigid_shape_properties(env, actor)
            for v in rb_shape_props:
                v.contact_offset = 0.0001
            self.gym.set_actor_rigid_shape_properties(env, actor, rb_shape_props)

    def create_tensors(self):
        super().create_tensors()
        string_names = ["G:string{}".format(i) for i in range(1, self.n_strings+1)]
        string_end_names = ["G:string{}_end".format(i) for i in range(1, self.n_strings+1)]
        strings = []
        strings_end = []

        for string in string_names:
            for actor in self.actors:
                sid = self.gym.find_actor_rigid_body_handle(self.envs[0], actor, string)
                if sid != -1:
                    strings.append(sid)
                    break
            assert sid != -1, string
        for string in string_end_names:
            for actor in self.actors:
                sid = self.gym.find_actor_rigid_body_handle(self.envs[0], actor, string)
                if sid != -1:
                    strings_end.append(sid)
                    break
            assert sid != -1
        self.strings = torch.tensor(strings, dtype=torch.long, device=self.device)
        self.strings_end = torch.tensor(strings_end, dtype=torch.long, device=self.device)
        self.guitar = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], "guitar")
        self.wrist_l = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], "LH:wrist")
        self.palm_l = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], "LH:palm")
        self.wrist_r = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[-1], "RH:wrist")

        self.goal_tensor = self.goal_tensor.to(self.goal_timer.dtype)
        self.goal_cursor = torch.zeros((len(self.envs),), dtype=torch.int32, device=self.device)
        self.goal_pitch_adjust = torch.zeros((len(self.envs), 1), dtype=torch.int32, device=self.device)
        self.goal_t = torch.zeros((len(self.envs), self.goal_horizon), dtype=torch.int32, device=self.device)
        self.goal_t_ = torch.zeros((len(self.envs), ), dtype=torch.int32, device=self.device)

        self.goal_track = torch.zeros((len(self.envs), self.goal_sampling_range[1], self.n_strings), dtype=self.goal_tensor.dtype, device=self.device)
        self.goal_note_t = torch.zeros((len(self.envs), self.goal_track.size(1)), dtype=torch.int32, device=self.device)
            
        if self.note_file:
            self.load_notes(self.note_file)
            if not self.random_note_sampling:
                self.goal_track = torch.zeros((len(self.envs), self.note_track.size(1), self.n_strings), dtype=self.goal_tensor.dtype, device=self.device)
                self.goal_note_t = torch.zeros((len(self.envs), self.note_t.size(1)), dtype=torch.int32, device=self.device)
         
        self.arange_tensor = torch.arange(max(self.n_frets+1, len(self.envs)*self.n_strings, self.goal_track.size(1)), device=self.device)
        self.zero_tensor_int32 = torch.zeros((len(self.envs),), dtype=torch.int32, device=self.device)

    def load_notes(self, note_file):
        left_hand = self.wrist_l != -1
        right_hand = self.wrist_r != -1

        if os.path.splitext(note_file)[1] == ".yaml":
            with open(note_file, 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)
            dirname = os.path.dirname(note_file)
            note_files = []
            note_weights = []
            for item in motion_config['motions']:
                note_weights.append(item['weight'])
                note_files.append(os.path.join(dirname, item['file']))
        else:
            note_files = [note_file]
            note_weights = [1.0]

        effect_names = ["tied", "hammer/pull", "chords"]
        effect_coeff = [0, 0, 0]
        note_level_n = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64])
        note_level = note_level_n/64.
        notes, length, length_ext, weights = [], [], [], []
        t, beat, min_t = [], [], []
        for note_file, curr_weight in zip(note_files, note_weights):
            ct = 0
            if note_file.endswith(".json"):
                with open(note_file, "r") as f:
                    data = json.load(f)
                for track in data:
                    base_t = 240 / track["tempo"]   # time taken by four of 1/4 notes
                    dt = []
                    note = []
                    if not track["notes"] or track["notes"][0] is None:
                        continue
                    if len(track["notes"]) < self.goal_horizon:
                        print("Ignore track with less than {} notes".format(self.goal_horizon))
                        continue
                    for n in track["notes"]:
                        assert len(n["frets"]) == self.n_strings, "Invalid note notation with inconsisent number of strings found {}, {} vs. {}".format(n["frets"], len(n["frets"]), self.n_strings)
                        frets = np.array(n["frets"], dtype=int)
                        if "effects" in n and "chords" in n["effects"]:
                            n["effects"]["chords"] = np.array(n["effects"]["chords"], dtype=int)
                            m = frets == -1
                            frets = np.where(m, n["effects"]["chords"], frets)
                            n["effects"]["chords"] = np.logical_and(m, frets!=-1)
                        if np.any(frets > self.n_frets):
                            frets[frets > self.n_frets] = self.n_frets
                            print("Found frets outside the maximal fret {}. Verified to {}.".format(n["frets"], frets))
                        tar_fret = frets[frets > 0]
                        if len(tar_fret):
                            span = max(tar_fret) - min(tar_fret)
                            if span > 4:
                                span0 = span+1
                                while span > 4:
                                    mean = np.mean(frets[frets > 0])
                                    e = np.abs(frets - mean)
                                    e[frets < 1] = 0
                                    frets[np.argmax(e)] = 0
                                    tar_fret = frets[frets > 0]
                                    span = max(tar_fret) - min(tar_fret)
                                print("Found note spanning {} frets {}. Shrunk to {}".format(span0, n["frets"], frets))
                            if len(np.unique(tar_fret)) > 4:
                                idx, cnts = np.unique(tar_fret, return_counts=True)
                                frets[idx[np.argsort(cnts)[::-1]][4:]] = 0
                                print("Found note with more than four target frets that need to be pressed. {} is modified to {}".format(n["frets"], frets))
                        if type(n["t"]) == str:
                            n_dt = n["t"].split("/")
                            if len(n_dt) == 2:
                                if 64 % int(n_dt[1]) == 0:
                                    if 64 % int(n_dt[1]):
                                        print("Found invalid note interval {}. Only support note interval divisible by 64. (no triplet note support yet).".format(n["t"]))
                                    try:
                                        interval = int(n_dt[0]) * (64 // int(n_dt[1]))
                                    except:
                                        interval = float(n_dt[0]) * (64 // int(n_dt[1]))
                                else:
                                    e = np.abs(int(n_dt[0])*1. / int(n_dt[1]) - note_level)
                                    i = np.max((e == np.min(e)) * np.arange(len(e)))
                                    interval = note_level_n[i.item()]
                                    print("Found invalid note interval {}. Round to {}".format(n["t"], note_level[i]))
                            else:
                                interval = int(n_dt[0]) * 64
                        else:
                            i = np.where(np.isclose(n["t"]/base_t - note_level, 0))[0]
                            assert len(i) != 0, "Illegal beat level ({}/{}, {}) out of {} was found.".format(n["t"], track["tempo"], n["t"]/t, note_level)
                            interval = note_level_n[i.item()]

                        eff = np.zeros((self.n_strings, ), dtype=int)
                        for c, v in n["effects"].items():
                            if c not in effect_names: continue
                            assert len(v) == self.n_strings, "Effects with unsupported length found {} with frets {}".format(n["effects"], n["frets"])
                            try:
                                i = effect_names.index(c)
                            except:
                                continue
                            eff += 100*(2**effect_coeff[i]) * np.array(v)
                        skip = False
                        
                        if left_hand:
                            note_tar = frets.copy()
                            trivial = note_tar == -1   # open strings, does not matter pressing or not, theoretically
                            not_press = note_tar == 0  # should not press
                            note_tar[trivial] = 0
                            note_tar[not_press] = -1

                            if right_hand:
                                sign = np.sign(note_tar)
                                sign[sign == 0] = 1
                                note_tar = sign * (np.abs(note_tar) + eff)
                            elif self.merge_repeated_notes:
                                if len(note) and np.all(note_tar == note[-1]):
                                    # merge consecutive notes for left hand motion training
                                    # NOTE some consecutive notes with the same blend or vibrato effect last quite long (more than 300 frames)
                                    # TODO remove repeated measures
                                    if base_t/64*dt[-1] < 1:
                                        dt[-1] = min(int(np.ceil(1/(base_t/64))), dt[-1]+interval)
                                    skip = True
                        else:
                            note_tar = frets > -1
                            # remove tied
                            if "tiedNote" in n["effects"]:
                                note_tar = np.logical_and(note_tar, (np.array(n["effects"]["tiedNote"]) <= 0))
                            if "hammerOn/pullOff" in n["effects"]:
                                note_tar = np.logical_and(note_tar, (np.array(n["effects"]["hammerOn/pullOff"]) <= 0))                            
                            if "chords" in n["effects"]:
                                note_tar = np.logical_and(note_tar, (np.array(n["effects"]["chords"]) <= 0))

                            # fill in holes between notes for picking
                            tar = np.where(note_tar)[0]
                            if len(tar) > 1:
                                for i in range(np.min(tar), np.max(tar)+1):
                                    note_tar[i] = True

                        if skip: continue
                        note.append(note_tar)
                        dt.append(interval)

                    beat.append(track["tempo"])
                    length.append(len(note))
                    min_dt = min(dt)
                    min_t.append(min_dt)
                    # extend notes for sampling
                    for l in range(self.goal_sampling_range[1]):
                        j = l % length[-1]
                        if j == 0:
                            note.append(np.zeros_like(note[0]))
                            dt.append(min_dt)
                        note.append(note[j])
                        dt.append(dt[j])
                        if len(note) - length[-1] > self.goal_sampling_range[1]:
                            break
                    notes.append(note)
                    t.append(dt)
                    length_ext.append(len(note))
                    ct += 1
                
            else:
                raise NotImplementedError("Unsupported file type {} for music notes".format(note_file.split(".")[-1]))
            if not ct: continue


            if curr_weight is None or curr_weight < 0:
                for l in length[-ct:]:
                    weights.append(None)
            else:
                w = curr_weight / sum(length[-ct:])
                for l in length[-ct:]:
                    weights.append(w*l)
        
        tot_weights_assigned = sum(w for w in weights if w is not None)
        tot_length_with_weights = sum(l for l, w in zip(length, weights) if w is not None and w > 0)
        tot_length_without_weights = sum(l for l, w in zip(length, weights) if w is None)
        if tot_length_with_weights == 0:
            weights = length
        else:
            tot_weights_unassigned = tot_weights_assigned / tot_length_with_weights * tot_length_without_weights
            weights = [l / tot_length_without_weights * tot_weights_unassigned if w is None else w for l, w in zip(length, weights)]

        self.note_weight = torch.tensor(np.divide(weights, sum(weights)), dtype=torch.float32)
        max_length = max(length_ext) + self.goal_track.size(1) - self.goal_sampling_range[0]
        # add one more 1 note for grace period
        self.note_track = torch.tensor(np.array([np.pad(note, ((1, max_length-len(note)), (0,0)), constant_values=(0, 0)) for note in notes]), dtype=torch.int32, device=self.device)
        self.note_t = torch.tensor(np.array([np.pad(d, (1, max_length-len(d)), constant_values=(0,self.grace_period)) for d in t]), dtype=torch.float32, device=self.device)
        self.note_beat = torch.tensor(beat, dtype=torch.int32, device=self.device)
        self.note_length = torch.tensor(length, dtype=torch.int32, device=self.device).add_(1)
        self.note_length_ext = torch.tensor(length_ext, dtype=torch.int32, device=self.device).add_(1)
        self.note_frames_per_tbpm = 60*self.fps/16 # divide BPM to get the frames taken by every 1/64 note
        self.note_t_min = torch.tensor(min_t, dtype=torch.float32, device=self.device)
        if hasattr(self, "note_length"):
            print("Loaded {:d} notes from {:d} files of {:d} tracks".format(torch.sum(self.note_length).item(), len(note_files), len(self.note_length)))

    def reset_goal(self, env_ids=None):
        n_envs = len(self.envs) if env_ids is None else len(env_ids)
        device = self.device
        track_ids = self.note_weight.multinomial(n_envs, replacement=True).to(device)
        bpm = self.note_beat[track_ids]
        fret_adjust = self.zero_tensor_int32[:n_envs]
        
        restart = self.lifetime == 0
        not_restart = ~restart

        if self.random_note_sampling:
            track_len = self.note_length_ext[track_ids]
            arange_envs = self.arange_tensor[:n_envs]
            phase = torch.rand(track_len.shape, device=device)
            if self.goal_sampling_range[0] == self.goal_sampling_range[1]:
                n_samples = torch.full((n_envs,), self.goal_sampling_range[0], dtype=track_len.dtype, device=device)
            else:
                n_samples = torch.randint(self.goal_sampling_range[0], self.goal_sampling_range[1], (n_envs, ),
                    dtype=track_len.dtype, device=device)
            note_idx0 = phase.mul(track_len-n_samples).to(torch.int32)
            n_arange = self.arange_tensor[:self.goal_track.size(1)]
            note_idx = note_idx0.unsqueeze(-1) + n_arange
            valid = n_arange < n_samples.unsqueeze(-1)
            valid[:, 0] = False

            note_t = self.note_t[track_ids][arange_envs.unsqueeze(-1), note_idx]
            goal_track = self.note_track[track_ids][arange_envs.unsqueeze(-1), note_idx]
            note_t = note_t * valid
            goal_track = goal_track * valid.unsqueeze_(-1)
    
            if self.random_pitch_rate:
                act = goal_track > 0
                fret_max = torch.max(goal_track.view(n_envs, -1), -1).values
                fret_min = torch.min((fret_max.view(n_envs, 1, 1)*(~act) + goal_track*act).view(n_envs, -1), -1).values
                adjust = torch.rand(n_envs, device=device).mul_(self.n_frets-fret_max+fret_min).floor_().to(torch.int32) - fret_min + 1
                fret_adjust = torch.where(torch.rand(n_envs, device=device) < self.random_pitch_rate,
                    adjust, fret_adjust)
            
            if self.random_bpm_rate:
                dbpm = torch.randint(-20, 20, size=bpm.size(), dtype=bpm.dtype, device=device)
                mask = (torch.logical_and(bpm < 81,  dbpm >= 0) + \
                        torch.logical_and(bpm > 189, dbpm <= 0) + \
                        torch.logical_and(bpm > 80,  bpm < 190)) > 0
                bpm = torch.where(torch.rand(n_envs, device=device) < self.random_bpm_rate,
                    (dbpm*mask).add_(bpm), bpm
                )
                t_min = torch.min(note_t+(note_t==0)*10000, -1).values
            else:
                t_min = self.note_t_min[track_ids]
            # NOTE We set the minimum time taken by one note to be 5 frames (0.08333s given 60FPS)
            # This is longer than some notes who last, for example, only 0.0176886792452830s (less than 2 frames).
            dt = torch.round((self.note_frames_per_tbpm*t_min/bpm).clip_(min=5)) / t_min

            note_t = torch.round(note_t * dt.view(n_envs, 1)).to(self.goal_note_t.dtype)
            note_t[:, 0] = self.grace_period
            note_t.clip_(max=50) # reduce those notes lasting too long
            valid = n_arange < (n_samples-(self.goal_horizon-1)).unsqueeze(-1)
            timer = torch.sum(note_t * valid, -1, dtype=note_t.dtype)
        else:
            track_len = self.note_length[track_ids]
            goal_track = self.note_track[track_ids]
            note_t = self.note_t[track_ids]
            valid = self.arange_tensor[:note_t.size(1)] < (track_len-self.goal_horizon+1).unsqueeze(-1)
            t_min = self.note_t_min[track_ids]
            dt = torch.round((self.note_frames_per_tbpm*t_min/bpm).clip_(min=5)) / t_min
            note_t = torch.round(note_t * dt.view(n_envs, 1)).to(self.goal_note_t.dtype)
            note_t[:, 0] = self.grace_period
            timer = torch.sum(note_t * valid, -1, dtype=note_t.dtype)
            if self.training:
                note_t.clip_(max=50)  # reduce those notes lasting too long
                valid2 = self.arange_tensor[:note_t.size(1)] < (track_len+2).unsqueeze(-1)
                self.episode_length[env_ids] = torch.sum(note_t * valid2, -1, dtype=note_t.dtype)

        if n_envs == len(self.envs):
            self.goal_track[:] = goal_track
            self.goal_note_t[:] = note_t
            self.goal_timer[:] = torch.sum(self.goal_t[:, 1:], -1, dtype=note_t.dtype)*not_restart + timer
            self.goal_cursor[:] = -1
            self.goal_pitch_adjust[:] = fret_adjust.unsqueeze(-1)
        else:
            self.goal_track[env_ids] = goal_track
            self.goal_note_t[env_ids] = note_t
            self.goal_timer[env_ids] = torch.sum(self.goal_t[env_ids, 1:], -1, dtype=note_t.dtype)*not_restart[env_ids] + timer
            self.goal_cursor[env_ids] = -1
            self.goal_pitch_adjust[env_ids] = fret_adjust.unsqueeze(-1)
        
        if torch.any(restart).item():
            for _ in range(self.goal_horizon - 1):
                self.update_goal_tensor(restart)

    def reset_envs(self, env_ids):
        self.goal_tensor.index_fill_(0, env_ids, False)
        self.goal_t.index_fill_(0, env_ids, 0)
        return super().reset_envs(env_ids)

    def update_goal_tensor(self, env_ids):
        raise NotImplementedError

    def observe_goal(self, env_ids):
        raise NotImplementedError

    def _observe(self, env_ids):
        all_envs = env_ids is None or len(env_ids) == len(self.envs)
        if not self.viewer_pause:
            self.update_goal_tensor(env_ids)
        goal = self.observe_goal(env_ids)
        if all_envs:
            ob = observe_iccgan(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens, self.key_links, self.parent_link
            )
        else:
            ob = observe_iccgan(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids], self.key_links, self.parent_link
            )
        return torch.cat((ob.flatten(start_dim=1), goal), -1)

class ICCGANLeftHand(ICCGANHandBase):
    CHARACTER_MODEL = "assets/left_hand_guitar.xml"
    GOAL_REWARD_WEIGHT = 0.8

    def create_tensors(self):
        super().create_tensors()
        self._create_tensors(self)
        if not hasattr(self, "note_track"):
            self.reset_goal = self.reset_goal_random

    @staticmethod
    def _create_tensors(self):
        fret_names = ["G:nut"] + ["G:fret{}".format(i) for i in range(1, self.n_frets+1)]
        finger_names_l = ["LH:{}{}".format(i, joint) for i in ["index", "middle", "ring", "pinky"] for joint in ["1", "2", "3", "_top"]]
        finger_top_names_l = ["LH:{}{}".format(i, joint) for i in ["index", "middle", "ring", "pinky"] for joint in ["_top"]]
        
        self.frets = torch.tensor([
            self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], n)
            for n in fret_names
        ], dtype=torch.long, device=self.device)
        self.finger_joints_l = torch.tensor([
            self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], n)
            for n in finger_names_l
        ], dtype=torch.long, device=self.device)
        self.finger_top_l = [
            self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], n)
            for n in finger_top_names_l
        ]
        # for termination check
        self.finger_top_offset_l = torch.tensor([[[0, -0.006, 0] for _ in range(len(self.finger_top_l))]], dtype=torch.float32, device=self.device)
        self.thumb_joints_l = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[0], "LH:thumb_top")
        # fret press information
        self.goal_orig_t2 = torch.zeros((len(self.envs)), dtype=torch.float32, device=self.device)
        self.pressed = torch.zeros((len(self.envs), self.n_strings, self.n_frets), dtype=torch.float32, device=self.device)
        self.pressed_fret = torch.zeros((len(self.envs), self.n_strings, 12, self.n_frets), dtype=torch.bool, device=self.device)
        self.finger_not_released = torch.zeros((len(self.envs), 12), dtype=torch.bool, device=self.device)
        self.rew_tar = torch.full((len(self.envs), self.n_strings), 10000, dtype=torch.float32, device=self.device)
        self.press_correct = torch.zeros((len(self.envs), 1), dtype=torch.bool, device=self.device)

    def reset_goal_random(self, env_ids=None):
        raise NotImplementedError("Random goal generation for left hand policy has not been implemented.")

    def update_goal_tensor(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
            env_idsx = self.arange_tensor[:len(self.envs)]
        else:
            env_idsx = env_ids
        self.goal_t[env_ids, 0] -= 1
        self.goal_t_[env_ids] += 1
        next_note = self.goal_t[env_ids, 0] <= 0
        self.goal_cursor[env_ids] += next_note
        cursor = self.goal_cursor[env_ids]

        next_note.unsqueeze_(-1)
        not_next_note = ~next_note
        tar = self.goal_tensor[env_ids, self.n_strings:].clone()
        self.goal_tensor[env_ids, :-self.n_strings] *= not_next_note
        self.goal_tensor[env_ids, :-self.n_strings] += next_note * tar
        tar = self.goal_t[env_ids, 1:].clone()
        self.goal_t[env_ids, :-1] *= not_next_note
        self.goal_t[env_ids, :-1] += next_note * tar
        self.goal_t_[env_ids] *= not_next_note.squeeze_(-1)

        tar = torch.fmod(self.goal_track[env_idsx, cursor], 100) # remove effects
        self.goal_tensor[env_ids, -self.n_strings:] = tar+self.goal_pitch_adjust[env_ids]*(tar>0)
        self.goal_t[env_ids, -1] = self.goal_note_t[env_idsx, cursor]

    def observe_goal(self, env_ids):
        if env_ids is None:
            goal = self.goal_tensor
            t = self.goal_t
        else:
            goal = self.goal_tensor[env_ids]
            t = self.goal_t[env_ids]
        goal = goal.to(torch.float32).view(t.size(0), t.size(1), -1)
        # two special frets, -1 for not press, 0 for not matter 
        goal.div_(self.n_frets+2).mul_(2)
        goal[goal > 1] -= 2
        t = (t/20.).clip_(max=2).sub_(1).unsqueeze_(-1)
        return torch.cat((goal, t), -1).view(goal.size(0), -1)

    def reset_envs(self, env_ids):
        self.rew_tar.index_fill_(0, env_ids, 10000)
        self.press_correct.index_fill_(0, env_ids, 0)
        self.pressed.index_fill_(0, env_ids, 0)
        return super().reset_envs(env_ids)

    def termination_check(self):
        return self._termination_check(self)

    @staticmethod
    def _termination_check(self):
        wrist_pos = self.link_pos[:, self.wrist_l]
        wrist_pos = wrist_pos.unsqueeze(1)
        wrist_pos = rotatepoint(self.guitar_orient_, wrist_pos - self.guitar_pos)
        too_far1 = torch.any(torch.stack((
            wrist_pos[:,0,0] < -0.1, wrist_pos[:,0,0] > 0.3,
            wrist_pos[:,0,1] < -0.3, wrist_pos[:,0,1] > 0.3,
            wrist_pos[:,0,2] < -0.2, wrist_pos[:,0,2] > 0.05
        ), -1), -1)

        l_finger_pos = self.link_pos[:, self.finger_joints_l]
        l_finger_pos = rotatepoint(self.guitar_orient_, l_finger_pos - self.guitar_pos)
        too_far2 = torch.any(l_finger_pos[:, :, 2] < -0.07, 1)

        l_finger_top_pos = rotatepoint(self.link_orient[:, self.finger_top_l], self.finger_top_offset_l)
        l_finger_top_pos += self.link_pos[:, self.finger_top_l]
        l_finger_top_pos = rotatepoint(self.guitar_orient_, l_finger_top_pos - self.guitar_pos)
        l_finger_pos = torch.cat((l_finger_top_pos.unsqueeze_(-2), l_finger_pos.view(-1, 4, 4, 3)[:,:, 1:3]), 2).view(-1, 12, 3)
        under_fretboard = torch.any(torch.logical_and(l_finger_pos[:, :, 0].abs() < 0.028, l_finger_pos[:, :, 2] < 0), 1)

        l_thumb_pos = self.link_pos[:, self.thumb_joints_l].unsqueeze(1)
        l_thumb_pos = rotatepoint(self.guitar_orient_, l_thumb_pos - self.guitar_pos)
        too_far3 = l_thumb_pos[:, 0, 0] > 0.07

        too_far = (too_far1+too_far2+under_fretboard+too_far3) > 0
        return too_far * (self.lifetime > 4)

    @staticmethod
    def _init_obj_tensors(self):
        # fixed guitar pos
        self.guitar_pos = self.link_pos[:, self.guitar].unsqueeze(1) # N x 1 x 3
        guitar_orient = self.link_orient[:, self.guitar]             # N x 4
        self.guitar_orient_ = quatconj(guitar_orient).unsqueeze_(1)  # N x 1 x 4

        string0_pos = self.link_pos[:, self.strings]                  # N x N_strings x 3
        string1_pos = self.link_pos[:, self.strings_end]              # N x N_strings x 3
        self.string0_pos_numpy = string0_pos.cpu().numpy()
        self.string1_pos_numpy = (string0_pos + 0.7*(string1_pos-string0_pos)).cpu().numpy()

        fret_pos = self.link_pos[:, self.frets]                      # N x (N_frets+1) x 3, including frets and nut
        
        # when fret >= 10, t = 0.5
        t = torch.linspace(0.3, 0.02*self.n_frets+0.3, self.n_frets,
            device=fret_pos.device).clip_(max=0.5).unsqueeze_(-1)
        fret_pos = (1-t) * fret_pos[:, 1:] + t * fret_pos[:, :-1]

        string0_pos = rotatepoint(self.guitar_orient_, string0_pos-self.guitar_pos)  # N x N_strings x 3
        string1_pos = rotatepoint(self.guitar_orient_, string1_pos-self.guitar_pos)  # N x N_strings x 3
        fret_pos = rotatepoint(self.guitar_orient_, fret_pos-self.guitar_pos)        # N x N_frets x 3
        self.fret_pos_tar = fret_pos.clone()

        fret_pos.unsqueeze_(1)
        string0_pos.unsqueeze_(2)           # N x N_strings x 1 x 3
        string1_pos.unsqueeze_(2)           # N x N_strings x 1 x 3

        t = (fret_pos[..., 1] - string0_pos[..., 1]) / (string1_pos[..., 1] - string0_pos[..., 1])
        t.unsqueeze_(-1)
        x = string0_pos + t * (string1_pos - string0_pos)
        x[..., 1:] = 0
        fret_pos[..., 0] = 0

        # height is the same with string
        string_fret_pos = fret_pos + x                          # N x N_strings x N_frets x 3
        self.string_fret_pos_numpy = rotatepoint(quatconj(self.guitar_orient_).unsqueeze(1), string_fret_pos)
        self.string_fret_pos_numpy = self.string_fret_pos_numpy.add_(self.guitar_pos.unsqueeze(1)).cpu().numpy()

        self.string_fret_pos = string_fret_pos.unsqueeze_(-2).unsqueeze_(-2)    # N x N_strings x N_frets x 1 x 1 x 3
        self.string0_pos = string0_pos.unsqueeze_(2)            # N x N_strings x 1 x 1 x 3
        self.string1_pos = string1_pos.unsqueeze_(2)            # N x N_strings x 1 x 1 x 3
        self.fret_pos = rotatepoint(self.guitar_orient_, self.link_pos[:, self.frets]-self.guitar_pos) # N x (N_frets+1) x 3

        finger_joints = self.link_pos[:, self.finger_joints_l]              # N x (N_fingers x 4) x 3
        finger_joints = finger_joints.view(finger_joints.size(0), -1, 4, 3) # N x N_fingers x 4 x 3
        finger_joints0 = finger_joints[..., :-1,:]       # N x (N_fingers x 3) x 3
        finger_joints1 = finger_joints[..., 1:, :]       # N x (N_fingers x 3) x 3
        self.finger_len = (finger_joints1 - finger_joints0).square_().sum(-1).sqrt_().view(finger_joints.size(0), -1, 3) # N x N_fingers x 3

    @torch.no_grad()
    def reward(self):
        if self.simulation_step < 2: self._init_obj_tensors(self)
        return self._reward(self)

    @staticmethod
    def _reward(self, statistic=True):
        n_envs = len(self.envs)
        finger_joints = self.link_pos[:, self.finger_joints_l] - self.guitar_pos  # N x (N_fingers x 4) x 3
        finger_joints = rotatepoint(self.guitar_orient_, finger_joints)           # N x (N_fingers x 4) x 3
        finger_joints = finger_joints.view(finger_joints.size(0), -1, 4, 3)       # N x N_fingers x 4 x 3
        finger_joints0 = finger_joints[..., :-1,:]       # N x N_fingers x 3 x 3    # start joint of each finger part
        finger_joints1 = finger_joints[..., 1:, :]       # N x N_fingers x 3 x 3    # end joint of each finger part
        goal = self.goal_tensor[:, :self.n_strings]

        dist2fret = dist2_p2seg(
            self.string_fret_pos,           # N x N_strings x N_frets x 1 x 1 x 3
            finger_joints0[:,None,None],    # N x       1 x 1 x N_fingers x 3 x 3
            finger_joints1[:,None,None]     # N x       1 x 1 x N_fingers x 3 x 3
        )   # N x N_strings x N_frets x N_fingers x 3
        
        # invalid press for finger part 0 and 1 (the end parts) if they are too vertial (more than 10 degrees)
        invalid_finger_pose = (finger_joints1[...,2] - finger_joints0[...,2]).div_(self.finger_len) > 0.0872 #0.1736   # N x N_fingers x 3
        invalid_finger_pose[..., 2] = False
        dist2fret += torch.where(invalid_finger_pose, torch.inf, 0).view(-1, 1, 1, 4, 3)
        dist2fret = dist2fret.view(finger_joints.size(0),
               self.string_fret_pos.size(1), 
               self.string_fret_pos.size(2), -1)        # N x N_strings x N_frets x (N_fingers x 3)

        closest_p_string, closest_p_finger = closest_seg2seg(
            self.string0_pos,           # N x N_strings x 1 x 1 x 3
            self.string1_pos,           # N x N_strings x 1 x 1 x 3
            finger_joints0[:,None],     # N x 1 x N_fingers x 3 x 3
            finger_joints1[:,None]      # N x 1 x N_fingers x 3 x 3
        )   # N x N_strings x N_fingers x 3 x 3
        dist2string = (closest_p_string - closest_p_finger).square_().sum(-1) # N x N_strings x N_fingers x 3

        # check fret pressing by y coordinate (direction along the fretboard)
        closest_p_string = closest_p_string[..., 1].view(n_envs, -1, 1) # N x (N_strings x N_fingers x 3) x 1
        fret_pos = self.fret_pos[..., 1].view(n_envs, 1, -1)            # N x 1 x (N_frets+1)
        # consider finger radius is around 0.006
        pressed_string = dist2string < 0.00004                  # N x N_strings x N_fingers x 3
        finger_over_fret = torch.logical_and(
            closest_p_string > fret_pos[..., 1:],               # N x (N_strings x N_fingers x 3) x N_frets
            closest_p_string < fret_pos[..., :-1]
        )
        pressed_fret = (
            finger_over_fret * pressed_string.view(n_envs, -1, 1) # N x (N_strings x N_fingers x 3) x N_frets
        ).view(n_envs, self.n_strings, -1, self.n_frets)          # N x N_strings x (N_fingers x 3) x N_frets

        arange_frets = self.arange_tensor[1:self.n_frets+1]
        pressed_board = torch.any(pressed_fret.view(n_envs, self.n_strings, -1, self.n_frets), 2)  # N x N_strings x N_frets
        press = (pressed_board * arange_frets).max(-1).values * torch.any(pressed_board, -1)
        self.info["press"] = press                  # N x N_strings
        self.info["press_board"] = pressed_board    # N x N_strings x N_frets
        self.pressed += arange_frets == press.unsqueeze(-1) # N x N_strings x N_frets

        m_tar = goal > 0
        not_new_note = self.goal_t[:, 0] > 1
        if statistic:
            new_note = ~not_new_note
            if self.simulation_step > self.grace_period:
                n = torch.logical_and(new_note, self.lifetime > self.grace_period)
                pressed = self.pressed >= self.goal_orig_t2.view(n_envs, 1, 1)   # N x N_strings x N_frets
                unpressed = pressed.sum(-1) == 0                  # N x N_strings
                pressed = (pressed * arange_frets).max(-1).values # N x N_strings
                pressed[unpressed] = -1
                targeted = goal != 0
                n_targets = targeted.sum(-1)
                accuracy = (((pressed * targeted) == goal).sum(-1) - self.n_strings + n_targets) / n_targets
                accuracy = accuracy[n]
                # accuracy = accuracy[~torch.isnan(accuracy)]
                accuracy.nan_to_num_(nan=1.0)

                TP = torch.logical_and(pressed == goal, m_tar).sum(-1)
                TP_FP = m_tar.sum(-1)  # number of target
                TP_FN =  torch.logical_and(pressed > 0, targeted).sum(-1)     # number of pressing, ignore those trivial strings
                recall = (TP / TP_FP)[n] # how accuracy of the target
                precision = (TP / TP_FN)[n] # how accuracy of pressing
                precision.nan_to_num_(nan=1.0)
                recall.nan_to_num_(nan=1.0)
                # f1 = (precision*recall).div_(precision+recall).mul_(2).nan_to_num_(nan=0)
                self.info["accuracy_l"] = accuracy
                self.info["precision_l"] = precision
                self.info["recall_l"] = recall
                # self.info["f1_l"] = f1
            self.goal_orig_t2 *= not_new_note
            self.goal_orig_t2 += new_note*(self.goal_t[:, 1]*0.5)    # N
        self.pressed *= not_new_note.view(n_envs, 1, 1)     # N x N_strings x N_frets

        # goal: N x N_strings
        # dist2fret:   N x N_strings x N_frets x (N_fingers x 3)
        # dist2string: N x N_strings x N_fingers x 3
        m_board_nogoal = goal.unsqueeze(-1) != self.arange_tensor[1:self.n_frets+1] # N x N_strings x N_frets

        # 0 current fret, 1 lower fret, -1 higher fret
        # mute strings (goal==0) as the lowest fret
        # open strings (goal==-1) are the highest fret
        mx = torch.sign(goal.unsqueeze(1)-goal.unsqueeze(-1) + 1000000*(goal.unsqueeze(1)==0))
        # fill the holes in the same fret as -1 if any -1 in that hole
        for _ in range(self.n_strings-1):
            mx[:,:,:-1] = torch.where(mx[:,:,:-1] > 0, mx[:,:,1:], mx[:,:,:-1])
        # fill the head and trail slots in each fret as -1
        m = (mx==0)*self.arange_tensor[1:self.n_strings+1]
        ub = torch.argmax(m, -1, keepdim=True)
        lb = torch.argmin(m+(m==0)*1000000, -1, keepdim=True)
        mx[torch.logical_or(self.arange_tensor[:self.n_strings] < lb, self.arange_tensor[:self.n_strings] > ub)]=-1
        # put an additional -1 slot as the 7th string (top most)
        mx_ = torch.cat((mx, torch.full((mx.size(0), mx.size(1), 1), -1, device=mx.device)), -1)
        # check if a new finger is need for pressing (i.e. the transfer from -1 to 0)
        y = torch.logical_and(mx_[:,:,1:]==-1, mx_[:,:,:-1]==0)
        # count from tail (top) to head (bottom)
        # so the top strings will be considered at the left of the bottom strings
        z = torch.flip(torch.cumsum(torch.flip(y, dims=(-1,)), -1), dims=(-1,))
        # add offset to each fret, so high fret or top string group has a lower id number
        # this assumption is not always true in practical cases
        xx = torch.diagonal(((goal.unsqueeze(-1)-1)*7+z), dim1 = -2, dim2 = -1)*m_tar
        fret_sorted, fret_order = torch.sort(xx)
        fret_unique = torch.cat((fret_sorted[:,:1]>0, fret_sorted[:, :-1] != fret_sorted[:, 1:]), -1)
        freedom = (4 - torch.sum(fret_unique, -1, keepdim=True)).view(n_envs, 1, 1)   # N x 1 x 1
        lb0 = torch.cumsum(fret_unique, -1)                     # N x N_strings
        lb0.add_(-1)
        lb = torch.gather(lb0, -1, fret_order.argsort(-1))
        lb.unsqueeze_(-1)                                       # N x N_strings x 1
        ub = lb + freedom                                       # N x N_strings x 1
        m_not_tar_finger = torch.logical_or(                       # N x N_strings x N_fingers
            lb > self.arange_tensor[:4],
            self.arange_tensor[:4] > ub) 
        m = torch.logical_or(                          # N x N_strings x N_frets x N_fingers
            m_board_nogoal.view(n_envs, self.n_strings, self.n_frets, 1),   # N x N_strings x N_frets x 1
            m_not_tar_finger.unsqueeze_(2))                                    # N x N_strings x 1 x N_fingers
        dist2fret = dist2fret.view(*m.shape, 3)
        dist2fret.masked_fill_(m.unsqueeze(-1), torch.inf) # N x N_strings x N_frets x N_fingers x 3
        dist2fret_min = dist2fret.view(n_envs, self.n_strings, -1).min(-1)  # N x N_strings

        dist2fret = dist2fret_min.values   # N x N_strings
        rew_press = dist2fret.mul(-1000).exp_()
        rew_press.mul_(0.8).add_(dist2fret.mul(-30).exp_(), alpha=0.2).clip_(0, 1)
        
        has_goal = torch.any(m_tar, -1, keepdim=True)
        m_notar = torch.logical_and(goal == -1, has_goal) # if all taget spring is -1 (not pressing), consider it as not any goal
        m_nogoal = goal == 0 # pressing or not does not matter
        dist2string = dist2string.view(n_envs, self.n_strings, -1) # N x N_strings x (N_fingers x 3)
        dist2string_min = dist2string.min(-1).values  # N x N_strings
        
        rew_not_press = dist2string_min.div(0.000049).square_().clip_(0, 1)

        rew_correct = torch.all(torch.logical_or(press == goal.clip(min=0), m_nogoal), -1, keepdim=True)

        rew_tar = rew_press * m_tar + rew_not_press * m_notar + m_nogoal * rew_not_press.mul(0.1).add_(0.9)
        rew_tar.mul_(0.8).add_(rew_correct, alpha=0.2)

        rew = torch.where(has_goal, rew_tar, rew_not_press)

        wrist_idx = 13 + self.wrist_l*13
        pw, pw_ = self.link_pos[:, self.wrist_l], self.state_hist[-1][:, wrist_idx:wrist_idx+3]
        qw, qw_ = self.link_orient[:, self.wrist_l], self.state_hist[-1][:, wrist_idx+3:wrist_idx+7]
        pf_ = []
        for f in self.finger_top_l:
            fidx = 13 + f*13
            pf_.append(self.state_hist[-1][:, fidx:fidx+3])
        pf_ = rotatepoint(quatconj(qw_).unsqueeze_(1), torch.stack(pf_, 1)-pw_.unsqueeze(1))
        pf = rotatepoint(quatconj(qw).unsqueeze_(1), self.link_pos[:, self.finger_top_l]-pw.unsqueeze(1))
        dw = torch.linalg.norm(pw - pw_, ord=2, dim=-1)
        df = torch.linalg.norm(pf - pf_, ord=2, dim=-1).mean(-1)
        rew += dw.add_(df, alpha=0.1).square_().mul_(-50*self.fps).exp_().unsqueeze_(-1).mul_(0.05)
        return rew
    
    def update_viewer(self):
        super().update_viewer()
        self.gym.clear_lines(self.viewer)
        self._update_viewer(self, self.goal_tensor)

    @staticmethod
    def _update_viewer(self, goal_tensor, draw_strings=True):
        goal0 = goal_tensor[:, :self.n_strings].cpu().numpy()

        tar_strings = goal0 > 0
        goal = goal0 - 1
        target = np.where(tar_strings)
        env_ids, string_ids = target[0], target[1]

        n_lines = 200
        phi = np.linspace(0, 2*np.pi, 20)
        theta = np.linspace(0, np.pi, 10)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        dx = 0.008 * (sin_phi[:, None] * cos_theta[None, :])
        dy = 0.008 * (sin_phi[:, None] * sin_theta[None, :])
        dz = 0.008 * cos_phi
        dx.shape = (-1, )
        dy.shape = (-1, )
        n_lines = len(dx)

        string_fret_pos = self.string_fret_pos_numpy
        for i, s in zip(env_ids, string_ids):
            e = self.envs[i]
            p = string_fret_pos[i, s][goal[i, s]]
            l = np.stack([
                np.stack((p[0], p[1], p[2], p[0]+x, p[1]+y, p[2]+dz[i%len(dz)]))
                for i, (x, y) in enumerate(zip(dx, dy))
            ])
            self.gym.add_lines(self.viewer, e, n_lines, np.float32(l), np.float32([[1.,0.,0.] for _ in range(n_lines)]))

        if draw_strings:
            for i, (e, s0, s1) in enumerate(zip(self.envs, self.string0_pos_numpy, self.string1_pos_numpy)):
                l = np.concatenate((s0, s1), -1)
                self.gym.add_lines(self.viewer, e, len(l), np.float32(l), np.float32([[1.,0.,0.] if goal0[i,j] == -1 else [0.,0.,0.] for j in range(len(l))]))


class ICCGANRightHand(ICCGANHandBase):
    CHARACTER_MODEL = "assets/right_hand_guitar.xml"
    def __init__(self, *args, **kwargs):
        kwargs["random_pitch_rate"] = 0
        super().__init__(*args, **kwargs)

    def get_goal_dim(self):
        policy_goal = (self.n_strings+1) * self.goal_horizon
        return policy_goal, policy_goal+6

    @staticmethod
    def _create_tensors(self):
        finger_names_r = ["RH:wrist"] + ["RH:{}{}".format(i, joint) for i in ["thumb", "index", "middle", "ring", "pinky"] for joint in ["1", "2", "3", "_top"]]
        self.pick = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[-1], "RH:pick")
        self.thumb3 = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[-1], "RH:thumb3")
        self.index3 = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[-1], "RH:index3")
        self.index2 = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[-1], "RH:index2")
        self.wrist = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[-1], "RH:wrist")
        self.pluck_range = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[-1], "G:pluck_range")
        assert self.pluck_range > -1
        self.finger_joints_r = torch.tensor([
            self.gym.find_actor_rigid_body_handle(self.envs[0], self.actors[-1], n)
            for n in finger_names_r
        ], dtype=torch.long, device=self.device)

        # for right hand reward function
        self.joint_state_hist = torch.zeros((self.state_hist.size(0), *self.joint_pos.shape), dtype=self.joint_pos.dtype, device=self.device)
        self.zero_tensor_float32 = torch.zeros((len(self.envs), self.n_strings), dtype=torch.float32, device=self.device)

        # for plucking information
        self.pluck_correct = torch.ones((len(self.envs), self.n_strings), dtype=torch.bool, device=self.device)

        # for performance statistic
        self.goal_orig = torch.zeros_like(self.goal_tensor[:, :self.n_strings])

    def create_tensors(self):
        super().create_tensors()
        self._create_tensors(self)

        self.goal_track = self.goal_track.to(torch.bool)
        self.goal_tensor = self.goal_tensor.to(torch.bool)
        if hasattr(self, "note_track"):
            self.note_track = self.note_track.to(torch.bool)
        else:
            self.note = torch.tensor([
                    # [False, False, False, False, False, False],
                    [False, False, False, False, False, True],
                    [False, False, False, False, True,  False],
                    [False, False, False, False, True,  True],
                    [False, False, False, True,  False, False],
                    [False, False, False, True,  True,  False],
                    [False, False, False, True,  True,  True],
                    [False, False, True,  False, False, False],
                    [False, False, True,  True,  False, False],
                    [False, False, True,  True,  True,  False],
                    [False, False, True,  True,  True,  True],
                    [False, True,  False, False, False, False],
                    [False, True,  True,  False, False, False],
                    [False, True,  True,  True,  False, False],
                    [False, True,  True,  True,  True,  False],
                    [False, True,  True,  True,  True,  True],
                    [True,  False, False, False, False, False],
                    [True,  True,  False, False, False, False],
                    [True,  True,  True,  False, False, False],
                    [True,  True,  True,  True,  False, False],
                    [True,  True,  True,  True,  True,  False],
                    [True,  True,  True,  True,  True,  True],
            ], device=self.device, dtype=torch.bool)
            self.reset_goal = self.reset_goal_random
    
    def reset_goal_random(self, env_ids=None):
        n_envs = len(self.envs) if env_ids is None else len(env_ids)
        horizon = self.goal_track.size(1)

        notes = torch.randint(0, len(self.note), (n_envs, horizon), device=self.device)
        goal = self.note[notes.flatten()].view(n_envs, horizon, -1)

        dt = torch.randint(5, 45, (n_envs, horizon), device=self.device, dtype=self.goal_timer.dtype)

        # grace period
        goal[:, 0] *= False
        dt[:, 0] = self.grace_period
        timer = torch.sum(dt[:, :-(self.goal_horizon-1)], -1, dtype=dt.dtype)
        restart = self.lifetime == 0
        not_restart = ~restart
        
        if n_envs == len(self.envs):
            self.goal_track[:] = goal
            self.goal_note_t[:] = dt
            self.goal_timer[:] = torch.sum(self.goal_t[:, 1:], -1, dtype=dt.dtype)*not_restart + timer
            self.goal_cursor[:] = -1
        else:
            self.goal_track[env_ids] = goal
            self.goal_note_t[env_ids] = dt
            self.goal_timer[env_ids] = torch.sum(self.goal_t[env_ids, 1:], -1, dtype=dt.dtype)*not_restart[env_ids] + timer
            self.goal_cursor[env_ids] = -1
        
        if torch.any(restart).item():
            for _ in range(self.goal_horizon - 1):
                self.update_goal_tensor(restart)

    def update_goal_tensor(self, env_ids=None):
        if env_ids is None: env_ids = slice(None)
        self.goal_t[env_ids, 0] -= 1
        self.goal_t_[env_ids] += 1

        next_note = self.goal_t[env_ids, 0] <= 0
        self.goal_cursor[env_ids] += next_note
        cursor = self.goal_cursor[env_ids]

        next_note.unsqueeze_(-1)
        not_next_note = ~next_note
        tar = self.goal_tensor[env_ids, self.n_strings:].clone()
        self.goal_tensor[env_ids, :-self.n_strings] *= not_next_note
        self.goal_tensor[env_ids, :-self.n_strings] += next_note * tar
        tar = self.goal_t[env_ids, 1:].clone()
        self.goal_t[env_ids, :-1] *= not_next_note
        self.goal_t[env_ids, :-1] += next_note * tar
        self.goal_t_[env_ids] *= not_next_note.squeeze_(-1)

        env_ids = self.arange_tensor[:len(self.envs)][env_ids]
        self.goal_tensor[env_ids, -self.n_strings:] = self.goal_track[env_ids, cursor]
        self.goal_t[env_ids, -1] = self.goal_note_t[env_ids, cursor]

    def observe_goal(self, env_ids):
        if env_ids is None:
            goal = self.goal_tensor
            t = self.goal_t
            pluck_correct = self.pluck_correct.to(torch.float32)
        else:
            goal = self.goal_tensor[env_ids]
            t = self.goal_t[env_ids]
            pluck_correct = self.pluck_correct[env_ids].to(torch.float32)
        goal = goal.to(torch.float32).view(t.size(0), t.size(1), -1)
        t = (t/20.).clip_(max=2).sub_(1).unsqueeze_(-1)
        return torch.cat((torch.cat((goal, t), -1).view(goal.size(0), -1), pluck_correct), -1)

    def reset_envs(self, env_ids):
        self.pluck_correct.index_fill_(0, env_ids, True)
        return super().reset_envs(env_ids)

    def observe(self, env_ids=None):
        if env_ids is None or len(env_ids) == len(self.envs):
            self.joint_state_hist[:-1] = self.joint_state_hist[1:].clone()
            self.joint_state_hist[-1] = self.joint_pos
        else:
            self.joint_state_hist[:-1, env_ids] = self.joint_state_hist[1:, env_ids].clone()
            self.joint_state_hist[-1, env_ids] = self.joint_pos[env_ids]
        return super().observe(env_ids)

    def termination_check(self):
        return self._termination_check(self)
    
    @staticmethod
    def _termination_check(self):
        r_finger_pos = self.link_pos[:, self.finger_joints_r]
        r_finger_pos = rotatepoint(self.guitar_orient_, r_finger_pos - self.guitar_pos)
        too_far = torch.any(torch.stack((
            r_finger_pos[:,:,0] < -0.3, r_finger_pos[:,:,0] > 0.3,
            r_finger_pos[:,:,1] < -0.6, r_finger_pos[:,:,1] > 0,
            r_finger_pos[:,:,2] < -0.1, r_finger_pos[:,:,2] > 0.3,
        ), -1).flatten(start_dim=1), -1)
        return torch.logical_and(too_far, self.lifetime>4)

    @staticmethod
    def _init_obj_tensors(self):
        # fixed guitar pos
        self.guitar_pos = self.link_pos[:, self.guitar].unsqueeze(1) # N x 1 x 3
        guitar_orient = self.link_orient[:, self.guitar]             # N x 4
        self.guitar_orient_ = quatconj(guitar_orient).unsqueeze_(1)  # N x 1 x 4

        string0_pos = self.link_pos[:, self.strings]                  # N x N_strings x 3
        string1_pos = self.link_pos[:, self.strings_end]              # N x N_strings x 3
        self.string0_pos_numpy = string0_pos.cpu().numpy()
        self.string1_pos_numpy = string1_pos.cpu().numpy()

        string0_pos = rotatepoint(self.guitar_orient_, string0_pos-self.guitar_pos)  # N x N_strings x 3
        string1_pos = rotatepoint(self.guitar_orient_, string1_pos-self.guitar_pos)  # N x N_strings x 3

        self.x1s, self.y1s, self.z1s = torch.unbind(string0_pos, -1)
        x2s, y2s, z2s = torch.unbind(string1_pos, -1)
        self.dxs = x2s - self.x1s
        self.dys = y2s - self.y1s
        self.dzs = z2s - self.z1s
        self.dxycs = self.y1s * x2s - y2s * self.x1s
        self.len2s = self.dxs.square().add_(self.dys.square())
        self.lens = self.len2s.sqrt()

    @torch.no_grad()
    def reward(self):
        if self.simulation_step < 2: self._init_obj_tensors(self)
        return self._reward(self, self.goal_tensor, self.goal_t)

    @staticmethod
    def _reward(self, goal_tensor, goal_t, ready=None, statistic=True):
        n_envs = len(self.envs)
        goal = goal_tensor[:, :self.n_strings]                         # N x N_string
        not_new_note = (goal_t[:, 0] > 1).unsqueeze_(-1)
        new_note = ~not_new_note

        if ready is not None:
            goal_ready = torch.logical_and(goal, ready)             # N x N_string
        else:
            goal_ready = goal

        pick = self.link_pos[:, self.pick].unsqueeze(1)                     # N x 1 x 3
        s = 13+self.pick*13
        pick  = rotatepoint(self.guitar_orient_, pick - self.guitar_pos)    # N x 1 x 3
        pick1 = rotatepoint(self.guitar_orient_, self.state_hist[-1][:, None, s:s+3] - self.guitar_pos)
        pick0 = rotatepoint(self.guitar_orient_, self.state_hist[-2][:, None, s:s+3] - self.guitar_pos)
        
        pick_dp = pick - pick1
        x1, y1, z1 = torch.unbind(pick1,  -1)
        x2, y2, z2 = torch.unbind(pick,  -1)
        dx, dy, dz = torch.unbind(pick_dp, -1)

        # refer: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment
        # Given horizontally
        #   string segment: self.string0_pos + s(self.string1_pos-self.string_pos0)
        #   pick tip trajectory: pick1 + t(pick-pick1)
        # the intersect point decided by s and t satisfies:
        #   x1s + s (x2s-x1s) = x1 + t (x2-x1)
        #   y1s + s (y2s-y1s) = y1 + t (y2-y1)
        den = dx*self.dys-dy*self.dxs
        parallel = den.abs() < 1e-8
        a, b = self.x1s-x1, y1-self.y1s
        c = a*self.dys + b*self.dxs
        t = torch.where(parallel, self.zero_tensor_float32, c/den)
        s = torch.where(parallel, self.zero_tensor_float32, (a*dy + b*dx)/den)
        z = z1+t*dz
        zs = self.z1s+s*self.dzs

        intersect = torch.logical_and(0 < t, t < 1)
        plucking = torch.logical_and(intersect, z < zs) # N x N_string
        if self.training:
           depth = 0.001 # string has radius of 0.0002
        else:
           depth = 0
        zs_depth = zs - depth 
        plucking_valid = torch.logical_and(intersect, z < zs_depth) # N x N_string

        is_plucking = torch.any(plucking, -1)                       # N
        pluck_cor = torch.logical_and(plucking_valid, goal_ready)
        pluck_err = torch.logical_and(plucking, ~pluck_cor)
        miss = torch.logical_and(goal, t < 0) * is_plucking.unsqueeze(-1)
        miss *= torch.any(torch.logical_and(plucking, goal), -1, keepdim=True)
        pluck_err += miss

        self.pluck_correct *= torch.logical_or((self.lifetime < 2).unsqueeze_(-1), ~pluck_err)

        s = (y2-self.y1s)/self.dys
        dist2str_v = z2 - zs
        dist2str_h = x2 - (self.x1s + s*self.dxs)
        dist2str = dist2str_h.square() + dist2str_v.square() # N x N_string

        string_min_tar = torch.argmax(goal.to(torch.int32), -1) # first target string
        string_max_tar = torch.max(goal*self.arange_tensor[:self.n_strings], -1).values # last target string
        rew2str = dist2str.mul(-10000).exp_().mul_(0.35).add_(
            dist2str.mul(-2000).exp_(), alpha=0.05)                     # N x N_string

        # expect 0.003m far away from any string
        above = 0.003
        rew_no_tar = dist2str.min(-1).values.sqrt_().div_(above).clip_(0, 1)    # N

        dist2str_z = (zs+above-z2).clip_(min=0).add_(dist2str_h.abs()).square_()
        rew2str_z = dist2str_z.mul(-10000).exp_().mul_(0.175).add_(dist2str_z.mul(-2000).exp_(), alpha=0.025)

        rew_plucking_err = torch.where(torch.logical_or(miss, ~goal_ready),
            # error for wrongly plucking
            rew2str_z,
            # error for plucking correctly but no plucking valid (not deep enough)
            0.2 + (z-zs_depth).div_(t).square_().mul_(-10000).exp_().mul_(0.6)
        ).mul_(pluck_err).add_(~pluck_err)
        rew_plucking_err = rew_plucking_err.min(-1).values

        rew2str = torch.maximum(
            rew2str[self.arange_tensor[:n_envs], string_min_tar],
            rew2str[self.arange_tensor[:n_envs], string_max_tar]
        )                                                                   # N
        
        rew_no_tar = rew_no_tar.mul_(0.4).add_(self.pluck_correct.sum(-1), alpha=0.6/self.n_strings).add_(torch.all(self.pluck_correct, -1).to(rew2str.dtype), alpha=0.5)

        cond = torch.logical_and(torch.any(goal_ready, -1), torch.all(torch.logical_or(goal_ready, ~goal), -1))
        rew = torch.where(is_plucking, rew_plucking_err, torch.where(cond, rew2str+0.2, rew_no_tar))

        a2 = ((pick_dp-(pick1 - pick0))*(self.fps**2 * 0.05)).square_().sum(-1)
        rew_a = a2.squeeze_(-1).square_()
        rew -= (self.lifetime>2) * rew_a.clip_(max=1).mul_(0.7)

        rew += pick_dp.square().sum(-1).squeeze_(-1).mul_(-20 * self.fps**2).exp_().mul_(0.05)

        i_wrist = 13+self.wrist_r*13
        wrist_v2 = ((self.link_pos[:, self.wrist_r] - self.state_hist[-1, :, i_wrist:i_wrist+3])*self.fps).square_().sum(-1)
        joint_v2 = (self.joint_state_hist[-1, :, 6:] - self.joint_pos[:, 6:]).square_().mean(-1).mul_(self.fps*self.fps)
        rew += wrist_v2.mul_(-120).exp_().mul_(0.05).add_(joint_v2.mul_(-50).exp_(), alpha=0.05)
        rew += torch.any(self.contact_force_tensor[:, self.thumb3] != 0, -1) * \
            torch.logical_or(
                torch.any(self.contact_force_tensor[:, self.index2] != 0, -1),
                torch.any(self.contact_force_tensor[:, self.index3] != 0, -1)
            ) * 0.05

        contact_string = torch.any(self.contact_force_tensor[:, self.pluck_range] !=0, -1)
        rew -= torch.logical_and(contact_string, self.lifetime > 4).to(rew.dtype)
        
        # update goal for next frame
        done = torch.logical_or(plucking, miss)
        goal_tensor[:, :self.n_strings] *= ~done

        if statistic and self.simulation_step > self.grace_period:
            # P target plucking string
            # N non-target string
            n = torch.logical_and(new_note.squeeze(-1), self.lifetime > self.grace_period)
            goal_done = torch.logical_and(~goal, self.goal_orig)
            TP = torch.logical_and(self.pluck_correct, goal_done).sum(-1) # plucked and correct
            TP_FP = self.goal_orig.sum(-1) # number of target strings
            TP_FN = torch.logical_or(~self.pluck_correct, goal_done).sum(-1) # number of plucked strings
            accuracy = torch.logical_and(self.pluck_correct, torch.logical_or(~self.goal_orig, goal_done)).sum(-1) / self.n_strings
            recall = (TP / TP_FP)[n] # how accuracy of the target strings
            precision = (TP / TP_FN)[n] # how accuracy of plucking
            recall.nan_to_num_(nan=1.0)
            precision.nan_to_num_(nan=1.0)
            # f1 = (precision*recall).div_(precision+recall).mul_(2).nan_to_num_(nan=0)
            self.info["accuracy_r"] = accuracy[n]
            self.info["precision_r"] = precision
            self.info["recall_r"] = recall
            # self.info["f1_r"] = f1

        # update indicators for new note
        self.goal_orig *= not_new_note
        self.goal_orig += new_note*goal_tensor[:, self.n_strings:self.n_strings+self.n_strings]
        # disable debug visualization
        self.pluck_correct[new_note.squeeze_(-1)] = True

        self.info["pluck"] = plucking
        self.info["new_note"] = new_note

        return rew.unsqueeze_(-1)

    def update_viewer(self):
        super().update_viewer()
        self.gym.clear_lines(self.viewer)
        self._update_viewer(self, self.goal_tensor)

    @staticmethod
    def _update_viewer(self, goal_tensor):
        goal = goal_tensor[:, :self.n_strings]
        goal = goal.cpu().numpy()
        goal_ = self.goal_orig.cpu().numpy()
        pluck_correct = self.pluck_correct.cpu().numpy()
        for i, (e, s0, s1) in enumerate(zip(self.envs, self.string0_pos_numpy, self.string1_pos_numpy)):
            l = np.concatenate((s0, s1), -1)
            c = [[1.,0.,1.] if not pluck_correct[i, j] else [1.,0.,0.] if goal[i,j] else [0.,1.,0] if goal_[i, j] else [0.,0.,0.] for j in range(len(l))]
            self.gym.add_lines(self.viewer, e, len(l), np.float32(l), np.float32(c))



class ICCGANTwoHands(ICCGANHandBase):
    CHARACTER_MODEL = ["assets/left_hand_guitar.xml", "right_hand.xml"]

    def get_goal_dim(self):
        return (self.n_strings+1)*self.goal_horizon*2 + 7

    def create_tensors(self):
        super().create_tensors()
        ICCGANLeftHand._create_tensors(self)
        ICCGANRightHand._create_tensors(self)
        # for right hand
        self.goal_tensor_right = torch.zeros_like(self.goal_tensor, dtype=torch.bool)

    def update_goal_tensor(self, env_ids=None):
        # a combination of LeftHand.update_goal_tensor and RightHand.update_goal_tensor
        if env_ids is None:
            env_ids = slice(None)
            env_idsx = self.arange_tensor[:len(self.envs)]
        else:
            env_idsx = env_ids
        self.goal_t[env_ids, 0] -= 1
        self.goal_t_[env_ids] += 1
        next_note = self.goal_t[env_ids, 0] <= 0
        self.goal_cursor[env_ids] += next_note
        cursor = self.goal_cursor[env_ids]

        next_note.unsqueeze_(-1)
        not_next_note = ~next_note
        tar = self.goal_tensor[env_ids, self.n_strings:].clone()
        self.goal_tensor[env_ids, :-self.n_strings] *= not_next_note
        self.goal_tensor[env_ids, :-self.n_strings] += next_note * tar
        tar = self.goal_t[env_ids, 1:].clone()
        self.goal_t[env_ids, :-1] *= not_next_note
        self.goal_t[env_ids, :-1] += next_note * tar
        tar = self.goal_tensor_right[env_ids, self.n_strings:].clone()
        self.goal_tensor_right[env_ids, :-self.n_strings] *= not_next_note
        self.goal_tensor_right[env_ids, :-self.n_strings] += next_note * tar
        self.goal_t_[env_ids] *= not_next_note.squeeze_(-1)

        tar_effect = self.goal_track[env_idsx, cursor]
        tar = torch.fmod(tar_effect, 100) # remove effects
        t = self.goal_note_t[env_idsx, cursor]

        tar_left = tar + self.goal_pitch_adjust[env_ids]*(tar>0)
        self.goal_tensor[env_ids, -self.n_strings:] = tar_left
        self.goal_t[env_ids, -1] = t

        # remove left-only effect notes
        tar_effect = torch.abs(tar_effect) // 100
        left_note = tar_effect % 2
        tar_right = tar * (1-left_note)
        # fill in holes
        goal_r = tar_right != 0
        string_arange = self.arange_tensor[:goal_r.size(-1)]
        tar_string = goal_r * string_arange
        goal_r = (string_arange >= torch.argmin(tar_string+100*(~goal_r), dim=-1, keepdim=True)) * \
                 (string_arange <= torch.argmax(tar_string, dim=-1, keepdim=True)) * \
                 torch.any(goal_r, -1, keepdim=True)
        self.goal_tensor_right[env_ids, -self.n_strings:] = goal_r

    def observe_goal(self, env_ids):
        if env_ids is None: 
            goal_l = self.goal_tensor
            goal_r = self.goal_tensor_right
            t = self.goal_t
            pluck_correct = self.pluck_correct.to(torch.float32)
        else:
            goal_l = self.goal_tensor[env_ids]
            goal_r = self.goal_tensor_right[env_ids]
            t = self.goal_t[env_ids]
            pluck_correct = self.pluck_correct[env_ids].to(torch.float32)
        n_envs = t.size(0)
        t = (t/20.).clip_(max=2).sub_(1).unsqueeze_(-1)
        goal_l = goal_l.to(torch.float32).view(n_envs, t.size(1), -1)
        goal_l.div_(self.n_frets+2).mul_(2)
        goal_l[goal_l > 1] -= 2
        goal_r = goal_r.to(torch.float32).view(n_envs, t.size(1), -1)
        if "ready" not in self.info:
            ready = torch.zeros_like(t[:, 0])
        else:
            if env_ids is None:
                ready = self.info["ready"].to(torch.float32)
            else:
                ready = self.info["ready"][env_ids].to(torch.float32)
        return torch.cat((torch.cat((goal_l, t),-1).view(n_envs,-1), torch.cat((goal_r, t),-1).view(n_envs,-1), ready, pluck_correct), 1)

    def reset_envs(self, env_ids):
        self.pressed.index_fill_(0, env_ids, 0)
        self.pluck_correct.index_fill_(0, env_ids, True)
        return super().reset_envs(env_ids)

    def observe(self, env_ids=None):
        if env_ids is None or len(env_ids) == len(self.envs):
            self.joint_state_hist[:-1] = self.joint_state_hist[1:].clone()
            self.joint_state_hist[-1] = self.joint_pos
        else:
            self.joint_state_hist[:-1, env_ids] = self.joint_state_hist[1:, env_ids].clone()
            self.joint_state_hist[-1, env_ids] = self.joint_pos[env_ids]
        return super().observe(env_ids)
    
    def termination_check(self):
        return torch.logical_or(ICCGANLeftHand._termination_check(self), ICCGANRightHand._termination_check(self))

    @torch.no_grad()
    def reward(self):
        if self.simulation_step < 2:
            ICCGANLeftHand._init_obj_tensors(self)
            ICCGANRightHand._init_obj_tensors(self)
        goal_left = self.goal_tensor[:, :self.n_strings]
        goal_right = self.goal_tensor_right[:, :self.n_strings]

        rew_l = ICCGANLeftHand._reward(self, statistic=True)

        pressed = torch.logical_or(
            self.info["press"] == (goal_left * (goal_left>0)),
            torch.logical_or(goal_left == 0, ~goal_right))
        timer = torch.logical_or(self.goal_t[:, 0] <= 3, self.goal_t_ >= 5)
        ready = torch.logical_or(pressed, timer.unsqueeze_(-1))
        rew_r = ICCGANRightHand._reward(self, self.goal_tensor_right, self.goal_t, ready, statistic=True)

        timer_ = torch.logical_or(self.goal_t[:, 0]-1 <= 3, self.goal_t_+1 >= 5)
        ready_ = torch.logical_or(pressed, timer_.unsqueeze_(-1))
        ready_ = torch.all(ready_, -1)
        ready2ready = torch.logical_and(torch.any(self.goal_tensor_right[:, :self.n_strings], -1), ~self.info["new_note"])
        self.info["ready"] = torch.logical_and(ready_, ready2ready).unsqueeze_(-1)

        return torch.cat((rew_l, rew_r), -1)
        
    def update_viewer(self):
        super().update_viewer()
        self.gym.clear_lines(self.viewer)
        ICCGANLeftHand._update_viewer(self, self.goal_tensor, draw_strings=False)
        ICCGANRightHand._update_viewer(self, self.goal_tensor_right)
