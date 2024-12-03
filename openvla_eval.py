"""
Test script to run the eval

python openvla_eval.py --test --env widowx_open_drawer
python openvla_eval.py --test --env widowx_close_drawer

python openvla_eval.py --env widowx_open_drawer --vla_url http://XXX.XXX.XXX.XXX:8000/act
"""

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import cv2
import numpy as np

# for openvla api call
import requests
import json_numpy
import argparse

json_numpy.patch()


def get_action(img, url):
    """Openvla api call"""
    img = cv2.resize(img, (256, 256))
    action = requests.post(
        url,
        json={"image": img, "instruction": "close the drawer", "unnorm_key": "bridge_orig"},
    ).json()
    print("Action", action)
    action = np.array(action)
    return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="widowx_close_drawer")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--vla_url", type=str, default="http://100.76.193.18:8000/act")
    args = parser.parse_args()

    env = simpler_env.make(args.env)
    obs, reset_info = env.reset()
    image = get_image_from_maniskill2_obs_dict(env, obs)

    instruction = env.get_language_instruction()
    print("Instruction", instruction)

    done, truncated = False, False
    action = np.array([0.01, 0.0, -0.01, 0.0, 0.0, 0.0, 1.0])  # do something
    count = 1

    while not (done or truncated):
        # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
        # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
        image = get_image_from_maniskill2_obs_dict(env, obs)

        # show image
        cv2.imshow("Image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

        if count % 20 == 0:
            action = -action

        count += 1
        if args.test:
            # random action
            action = env.action_space.sample()
        else:
            action = get_action(image, args.vla_url)

        print("Action", action)

        obs, reward, done, truncated, info = env.step(
            action
        )  # for long horizon tasks, you can call env.advance_to_next_subtask() to advance to the next subtask; the environment might also autoadvance if env._e>
        # new_instruction = env.get_language_instruction()
        # if new_instruction != instruction:
        #    # for long horizon tasks, we get a new instruction when robot proceeds to the next subtask
        #    instruction = new_instruction
        #    print("New Instruction", instruction)

    episode_stats = info.get("episode_stats", {})
    print("Episode stats", episode_stats)
