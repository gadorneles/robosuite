"""
A script to collect a batch of human demonstrations.

The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
import robosuite.utils.transform_utils as TU


def rotation_matrix(rot_angle, axis = 'z'):
        if axis == "x":
            return TU.quat2mat(np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0]))
        elif axis == "y":
            return TU.quat2mat(np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0]))
        elif axis == "z":
            return TU.quat2mat(np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]))


def collect_machine_trajectory(env, arm, env_configuration):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    env.reset()

    # ID = 2 always corresponds to agentview
    env.render()

    is_first = True

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    time_limit = 500

    obs, reward, done, _ = env.step(np.zeros(env.action_spec[0].shape))  # You can perform a zero action to start
    gripper_closed = False  # Keep track of whether the gripper is closed
    gripper_action = -1.0  # Open the gripper
    count = 0
    limit_count = 0
    counter = False


    # Loop until we get a reset from the input or the task completes
    while True:

        eef_pos = obs['robot0_eef_pos']  # Get the current end-effector position (3D vector)
        eef_quat = obs['robot0_eef_quat']  # Get the current end-effector orientation (4D quaternion)

        # Step 1: Convert current quaternion to rotation matrix
        current_rotation_matrix = TU.quat2mat(eef_quat)

        # Step 2: Compute the 90º (π/2 radians) rotation matrix around the Z axis
        rotation_90_z_matrix = rotation_matrix(np.pi / 2, axis='x')

        # Step 3: Compute the delta rotation matrix
        delta_rotation_matrix = np.dot(rotation_90_z_matrix, np.linalg.inv(current_rotation_matrix))

        # Step 4: Convert the delta rotation matrix back to a quaternion
        delta_quat = TU.mat2quat(delta_rotation_matrix)
        delta_axis = TU.quat2axisangle(delta_quat)

        # Get cube position
        cube_pos = obs['handle0_xpos']  # Get cube position (3D vector)
        #print(cube_pos)        
        # Compute the action to move the end-effector closer to the cube
        # Action is a delta, so we compute the difference between the cube and eef position
        pos_delta = cube_pos - eef_pos

        # Check if the gripper is close enough to the cube
        if abs(pos_delta[2]) < 0.001 and not gripper_closed:  # If end-effector is close to the cube
            print("End-effector is near the cube. Closing the gripper.")
            gripper_action = 1.0  # Set to negative value to close the gripper
            gripper_closed = True  # Mark gripper as closed

        # Create action: 3D delta for position, and keep the orientation part of action unchanged
        if not counter:
            action = np.zeros(7)
            action[:3] = pos_delta
            action[2] = pos_delta[2] - 0.05
            action[3:6] = delta_axis  
            action[6] = gripper_action
        
        # If the gripper is closed, we can stop trying to move the end-effector
        if gripper_closed:
            if count > 50 and count < 500:
                counter = True
                action = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, gripper_action])
                print("Gripper is closed. Stopping.")
            elif count > 500:
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_action])
            count += 1

        # Also break if we reach the time limit
        if limit_count > time_limit:
            print("Time limit reached.")
            break
        
        # Perform action
        obs, reward, done, _ = env.step(action)
        env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success
        
        limit_count += 1

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point
    successful_demos = 0
    unsuccessful_demos = 0

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        # Add only the successful demonstration to dataset
        if success:
            successful_demos += 1
            print("Demonstration is successful and has been saved")
            # Delete the last state. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action,
            # so we end up with an extra state at the end.
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            # store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            unsuccessful_demos += 1
            print("Demonstration is unsuccessful and has NOT been saved")
    
    # Summary of the processed demonstrations
    print(f"{successful_demos} successful demonstration(s) saved.")
    print(f"{unsuccessful_demos} unsuccessful demonstration(s) discarded.")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="AffordanceEnv")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    # collect demonstrations
    while True:
        collect_machine_trajectory(env, args.arm, args.config)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
