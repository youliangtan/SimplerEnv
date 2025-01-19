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

# gcbc policy
# NOTE: the config is located in eval_config.py
# this also requires $PWD/goal_images/task_name.png
python eval_simpler.py --env widowx_open_drawer --gcbc
python eval_simpler.py --env widowx_close_drawer --gcbc
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


def unnormalize_actions(actions, metadata):
    """normalize the first 6 dimensions of the widowx actions"""
    gripper_action = 1.0 if actions[6] > 0 else 0.0
    actions = np.concatenate(
        [
            metadata["std"][:6] * actions[:6] + metadata["mean"][:6],
            np.array([gripper_action]),
        ]
    )
    return actions


def create_bridge_example_batch(batch_size, img_size):
    """create a dummy batch of the correct shape to create the agent"""
    example_batch = {
        "observations": {
            "proprio": np.zeros(
                (
                    batch_size,
                    7,
                ),
            ),
            "image": np.zeros(
                (batch_size, img_size, img_size, 3),
            ),
        },
        "goals": {
            "image": np.zeros(
                (batch_size, img_size, img_size, 3),
            ),
        },
        "actions": np.zeros(
            (
                batch_size,
                7,
            ),
        ),
    }
    return example_batch


class GCPolicy():
    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.device = device
        self.agent = self.create_agent()
        self.action_statistics = {
            "mean": self.config["ACT_MEAN"],
            "std": self.config["ACT_STD"],
        }


    def create_agent(self):
        # lazy imports
        from flax.training import checkpoints
        from jaxrl_m.agents import agents
        from jaxrl_m.vision import encoders

        # encoder
        encoder_def = encoders[self.config["encoder"]](**self.config["encoder_kwargs"])

        # create agent
        example_batch = create_bridge_example_batch(
            batch_size=1, img_size=self.config["obs_image_size"]
        )
        self.rng = jax.random.PRNGKey(self.config["seed"])
        self.rng, construct_rng = jax.random.split(self.rng)
        agent = agents[self.config["policy_class"]].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **self.config["agent_kwargs"],
        )
        assert os.path.exists(self.config["checkpoint_path"]), "Checkpoint not found"
        agent = checkpoints.restore_checkpoint(self.config["checkpoint_path"], agent)
        return agent

    def get_action(
        self,
        obs_dict,
        language_instruction,
        deterministic=True,
    ):
        """the run loop code should pass in a `goal` field in the obs_dict
        Otherwise, the policy will look for a pre-specified goal in the `goal_images/` dir,
        with the filename being the language instruction
        """
        obs_image = obs_dict["image_primary"]

        # get the goal image
        try:
            goal_image = obs_dict["goal"]
        except KeyError:
            print("Goal not provided in obs_dict, looking for pre-specified goal")
            goal_file = os.path.join("goal_images", language_instruction + ".png")
            assert os.path.exists(
                goal_file
            ), f"Goal file {goal_file} not found, and not provided in obs_dict"
            goal_image = cv2.imread(goal_file)

        assert obs_image.shape == (
            self.config["obs_image_size"],
            self.config["obs_image_size"],
            3,
        ), "Bad input obs image shape"
        print(goal_image.shape)
        goal_image = cv2.resize(goal_image, (256, 256))
        assert goal_image.shape == (
            self.config["obs_image_size"],
            self.config["obs_image_size"],
            3,
        ), "Bad input goal image shape"

        self.rng, action_rng = jax.random.split(self.rng)
        # actions, action_mode 
        actions = self.agent.sample_actions(
            {"image": obs_image[np.newaxis, ...]},
            {"image": goal_image[np.newaxis, ...]},
            temperature=0.0,
            argmax=deterministic,
            seed=None if deterministic else action_rng,
        )
        print("Actions", actions)
        actions = np.array(actions)[0]  # unbatch
        actions = unnormalize_actions(actions, self.action_statistics)

        return actions


########################################################################

class WrapSimplerEnv(gym.Wrapper):
    def __init__(self, env):
        super(WrapSimplerEnv, self).__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
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
        obs, reward, done, truncated, info = self.env.step(action)
        obs, additional_info = self._process_obs(obs)
        info.update(additional_info)
        return obs, reward, done, truncated, info

    def _process_obs(self, obs):
        img = get_image_from_maniskill2_obs_dict(self.env, obs)
        return (
            {
                "image_primary": cv2.resize(img, (256, 256)),
            }, 
            {
                "original_image_primary": img,
            }
        )

########################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="widowx_close_drawer")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--octo", action="store_true")
    parser.add_argument("--gcbc", action="store_true")
    parser.add_argument("--vla_url", type=str, default="http://100.76.193.18:6633/act")
    parser.add_argument("--eval_count", type=int, default=10)
    parser.add_argument("--episode_length", type=int, default=120)
    parser.add_argument("--output_video_dir", type=str, default=None)
    args = parser.parse_args()

    base_env = simpler_env.make(args.env)
    base_env._max_episode_steps = args.episode_length # override the max episode length

    instruction = base_env.get_language_instruction()

    env = WrapSimplerEnv(base_env)

    print("Instruction", instruction)

    if not args.test:
        if args.octo:
            policy = OctoPolicy()
            from octo.utils.gym_wrappers import HistoryWrapper, TemporalEnsembleWrapper

            env = HistoryWrapper(env, horizon=2)
            env = TemporalEnsembleWrapper(env, 4)
        elif args.gcbc:
            from eval_config import jaxrl_gc_policy_kwargs
            policy = GCPolicy(jaxrl_gc_policy_kwargs)
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
