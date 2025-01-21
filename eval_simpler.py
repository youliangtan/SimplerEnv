"""
Test script to run the eval

python eval_simpler.py --test --env widowx_open_drawer
python eval_simpler.py --test --env widowx_close_drawer

# Openvla api call
python eval_simpler.py --env widowx_open_drawer --vla_url http://XXX.XXX.XXX.XXX:6633/act
python eval_simpler.py --env widowx_close_drawer --vla_url http://XXX.XXX.XXX.XXX:6633/act


# octo policy
python eval_simpler.py --env widowx_open_drawer --octo
python eval_simpler.py --env widowx_close_drawer --octo

"""

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import cv2
import numpy as np

# for openvla api call
import requests
import json_numpy
import argparse
import cv2
import os
import numpy as np
from collections import deque
# import gymnasium as gym
import gym

try:
    import jax
except ImportError:
    print("JAX not installed.")
    print("Please install jax using `pip install jax` if you want to use Octo model.")

from transforms3d import euler as te
from transforms3d import quaternions as tq

json_numpy.patch()

print_green = lambda x: print("\033[92m {}\033[00m".format(x))

# print numpy array with 2 decimal points
np.set_printoptions(precision=2)


########################################################################
class OpenVLAPolicy:
    def __init__(self, url):
        self.url = url

    def get_action(self, obs_dict, instruction):
        """
        Openvla api call to get the action.
            obs_dict : dict
            instuction : str
        """
        print("instruction", instruction)
        img = obs_dict["image_primary"]
        img = cv2.resize(img, (256, 256)) # ensure size is 256x256
        action = requests.post(
            self.url,
            json={"image": img, "instruction": instruction, "unnorm_key": "bridge_orig"},
        ).json()
        print("Action", action)
        action = np.array(action)
        return action

########################################################################

class OctoPolicy:
    def __init__(self, horizon=2):
        from octo.model.octo_model import OctoModel
        self.model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
        self.task = None  # created later

    def get_action(self, obs_dict, instruction):
        """
        Octo api call to get the action.
            obs_dict : dict
            instuction : str
        """
        if self.task is None:
            # assumes that each Octo model doesn't receive different tasks
            self.task = self.model.create_tasks(texts=[instruction])
            # self.task = self.agent.create_tasks(goals={"image_primary": img})   # for goal-conditioned

        actions = self.model.sample_actions(
            jax.tree_map(lambda x: x[None], obs_dict),
            self.task,
            unnormalization_statistics=self.model.dataset_statistics["bridge_dataset"][
                "action"
            ],
            rng=jax.random.PRNGKey(0),
        )
        # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
        actions = actions[0]  # note that actions here could be chucked
        # return actions from jax to numpy and take only the first action
        return np.asarray(actions)


########################################################################


class WrapSimplerEnv(gym.Wrapper):
    def __init__(self, env):
        super(WrapSimplerEnv, self).__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
                ),
                "proprio": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(7,), dtype=np.float32
        )

    def reset(self):
        obs, reset_info = self.env.reset()
        obs, additional_info = self._process_obs(obs)
        reset_info.update(additional_info)
        return obs, reset_info

    def step(self, action):
        """
        NOTE action is 7 dim
        [dx, dy, dz, droll, dpitch, dyaw, gripper]
        gripper: -1 close, 1 open
        """
        obs, reward, done, truncated, info = self.env.step(action)
        obs, additional_info = self._process_obs(obs)
        info.update(additional_info)
        return obs, reward, done, truncated, info

    def _process_obs(self, obs):
        img = get_image_from_maniskill2_obs_dict(self.env, obs)
        proprio = self._process_proprio(obs)
        return (
            {
                "image_primary": cv2.resize(img, (256, 256)),
                "proprio": proprio,
            }, 
            {
                "original_image_primary": img,
            }
        )
    
    def _process_proprio(self, obs):
        """
        Process proprioceptive information
        """
        # TODO: should we use rxyz instead of quaternion?
        # 3 dim translation, 4 dim quaternion rotation and 1 dim gripper
        eef_pose = obs['agent']["eef_pos"]
        # joint_angles = obs['agent']['qpos'] # 8-dim vector joint angles
        return eef_pose


class BridgeSimplerStateWrapper(gym.Wrapper):
    """
    NOTE(YL): this converts the prorio from the default 
    [x, y, z, qx, qy, qz, qw, gripper [0, 1]]
    is adapted from:
    https://github.com/allenzren/open-pi-zero/blob/main/src/agent/env_adapter/simpler.py
    """
    def __init__(self, env, **kwargs):
        super(BridgeSimplerStateWrapper, self).__init__(env)
        # EE pose in Bridge data was relative to a top-down pose, instead of robot base
        self.default_rot = np.array(
            [[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]
        )  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203
        
        # NOTE: now proprio is size 7
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
                ),
                "proprio": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
            }
        )

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs["proprio"] = self._preprocess_proprio(obs)
        return obs, info

    def step(self, action):
        action[-1] = self._postprocess_gripper(action[-1])
        obs, reward, done, trunc, info = super().step(action)
        obs["proprio"] = self._preprocess_proprio(obs)
        assert len(obs["proprio"]) == 7, "propio is incorrect size"
        return obs, reward, done, trunc, info

    def _preprocess_proprio(self, obs: dict) -> np.array:
        # convert ee rotation to the frame of top-down
        # proprio = obs["agent"]["eef_pos"]
        proprio = obs["proprio"]
        assert len(proprio) == 8, "original proprio should be size 8"
        rm_bridge = tq.quat2mat(proprio[3:7])
        rpy_bridge_converted = te.mat2euler(rm_bridge @ self.default_rot.T)
        gripper_openness = proprio[7]
        raw_proprio = np.concatenate(
            [
                proprio[:3],
                rpy_bridge_converted,
                [gripper_openness],
            ]
        )
        return raw_proprio

    def _postprocess_gripper(self, action: float) -> float:
        """from simpler octo inference: https://github.com/allenzren/SimplerEnv/blob/7d39d8a44e6d5ec02d4cdc9101bb17f5913bcd2a/simpler_env/policies/octo/octo_model.py#L234-L235"""
        # trained with [0, 1], 0 for close, 1 for open
        # convert to -1 close, 1 open for simpler
        action_gripper = 2.0 * (action > 0.5) - 1.0
        return action_gripper


########################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="widowx_close_drawer")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--octo", action="store_true")
    parser.add_argument("--vla_url", type=str, default="http://100.76.193.18:6633/act")
    parser.add_argument("--eval_count", type=int, default=10)
    parser.add_argument("--episode_length", type=int, default=120)
    parser.add_argument("--output_video_dir", type=str, default=None)
    args = parser.parse_args()

    base_env = simpler_env.make(args.env)
    base_env._max_episode_steps = args.episode_length # override the max episode length

    instruction = base_env.get_language_instruction()

    env = WrapSimplerEnv(base_env)

    if "widowx" in args.env:
        print("Wrap Simpler with bridge state wrapper for proprio and action convention")
        env = BridgeSimplerStateWrapper(env)

    print("Instruction", instruction)

    if not args.test:
        if args.octo:
            policy = OctoPolicy()
            from octo.utils.gym_wrappers import HistoryWrapper, TemporalEnsembleWrapper

            env = HistoryWrapper(env, horizon=2)
            env = TemporalEnsembleWrapper(env, 4)
        else:
            policy = OpenVLAPolicy(args.vla_url)

    success_count = 0

    for i in range(args.eval_count):
        print_green(f"Evaluate Episode {i}")

        done, truncated = False, False
        obs, info = env.reset()

        images = []

        step_count = 0
        while not (done or truncated):
            # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
            # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
            # image = get_image_from_maniskill2_obs_dict(env, obs)
            image = obs["image_primary"]

            if args.output_video_dir:
                images.append(image)

            # show image
            full_image = info["original_image_primary"]
            cv2.imshow("Image", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

            if args.test:
                # random action
                action = env.action_space.sample()
            else:
                action = policy.get_action(obs, instruction)

            # print(f"Step {step_count} Action: {action}")
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1

        # check if the episode is successful
        if done:
            success_count += 1
            print_green(f"Episode {i} Success")
        else:
            print_green(f"Episode {i} Failed")

        # save mp4 video of the current episode
        if args.output_video_dir:
            video_name = f"{args.output_video_dir}/{args.env}_{i}.mp4"
            print(f"Save video to {video_name}")
            height, width, _ = images[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_name, fourcc, 20.0, (width, height))
            for image in images:
                out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            out.release()

        episode_stats = info.get("episode_stats", {})
        print("Episode stats", episode_stats)
        print_green(f"Success rate: {success_count}/{i + 1}")
