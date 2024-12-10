"""
Test script to run the eval

python openvla_eval.py --test --env widowx_open_drawer
python openvla_eval.py --test --env widowx_close_drawer

python openvla_eval.py --env widowx_open_drawer --vla_url http://XXX.XXX.XXX.XXX:8000/act
python openvla_eval.py --env widowx_close_drawer --vla_url http://XXX.XXX.XXX.XXX:8000/act
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

json_numpy.patch()

print_green = lambda x: print("\033[92m {}\033[00m".format(x))

# print numpy array with 2 decimal points
np.set_printoptions(precision=2)


def get_action(img, instruction, url):
    """
    Openvla api call to get the action.
        img : np.array
        instuction : str
        url : str
    """
    print("instruction", instruction)
    img = cv2.resize(img, (256, 256))
    action = requests.post(
        url,
        json={"image": img, "instruction": instruction, "unnorm_key": "bridge_orig"},
    ).json()
    print("Action", action)
    action = np.array(action)
    return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="widowx_close_drawer")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--vla_url", type=str, default="http://100.76.193.18:8000/act")
    parser.add_argument("--eval_count", type=int, default=1)
    parser.add_argument("--episode_length", type=int, default=120)
    parser.add_argument("--output_video_dir", type=str, default=None)
    args = parser.parse_args()

    env = simpler_env.make(args.env)
    env._max_episode_steps = args.episode_length # override the max episode length

    instruction = env.get_language_instruction()
    print("Instruction", instruction)

    success_count = 0

    for i in range(args.eval_count):
        print_green(f"Evaluate Episode {i}")

        done, truncated = False, False
        obs, reset_info = env.reset()
        images = []

        step_count = 0
        while not (done or truncated):
            # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
            # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
            image = get_image_from_maniskill2_obs_dict(env, obs)

            if args.output_video_dir:
                images.append(image)

            # show image
            cv2.imshow("Image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

            if args.test:
                # random action
                action = env.action_space.sample()
            else:
                action = get_action(image, instruction, args.vla_url)

            print(f"Step {step_count} Action: {action}")
            obs, reward, done, truncated, info = env.step(action)
            # for long horizon tasks, you can call env.advance_to_next_subtask() to advance to the next subtask; the environment might also autoadvance if env._e>
            # new_instruction = env.get_language_instruction()
            # if new_instruction != instruction:
            #    # for long horizon tasks, we get a new instruction when robot proceeds to the next subtask
            #    instruction = new_instruction
            #    print("New Instruction", instruction)
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
        print_green(f"Success rate: {success_count}/{args.eval_count}")
