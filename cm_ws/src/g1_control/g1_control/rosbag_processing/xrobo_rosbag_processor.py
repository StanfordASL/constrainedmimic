"""Load a rosbag with XRobo VR data, specify a start and end index, and save the data to a pickle"""

import pickle
import argparse
from typing import List
from pathlib import Path

import numpy as np

from g1_control.rosbag_processing.rosbag_loading import load_ros2_bag


def xrobo_messages_to_numpy(messages: List[dict]) -> dict[str, np.ndarray]:
    """Converts a sequence of XRobo messages into numpy arrays

    Args:
        messages (List[dict]): List of all XRobo state messages received
            length = total number of messages. Each message is stored as a
            dictionary with the same fields as in the ROS message

    Returns:
        dict[str, np.ndarray]: Dictionary storing time and pose info as
            numpy arrays of shape (num_timesteps, ...)
    """
    # ROS stores timestamps in nanoseconds
    # Convert to seconds before returning
    times_ns = np.array([m["timestamp"] for m in messages])
    times = (times_ns - times_ns[0]) * 1e-9

    # The PICO whole-body mode tracks 24 bodies.
    # See some of the config values in wboscbf for more info
    body_poses = np.array([m["flat_body_poses"] for m in messages]).reshape(-1, 24, 7)

    # TODO: Decide if including any additional data from the XRoboState message
    # (there is a lot more data available, for instance, controller values,
    # twists, headset data, ...)

    return {"times": times, "poses": body_poses}


def slice_data(data: dict, start_idx: int, end_idx: int) -> dict:
    """Slice a segment of the recorded rosbag data

    Args:
        data (dict): Processed rosbag data as a dictionary
        start_idx (int): Start slice index
        end_idx (int): End slice index

    Returns:
        dict: Sliced data. Each entry in the dict that had values on
            a per-timestep basis now has (end_idx - start_idx) values
    """
    num_timesteps = len(data["times"])
    sliced = {}
    for k, v in data.items():
        if len(v) == num_timesteps:
            sliced[k] = v[start_idx:end_idx]
        else:
            # If there is data stored that isn't associated per-timestep,
            # keep it in the new dict but don't try to slice it
            sliced[k] = v
    # Reset the timing data to start at 0 post-slice
    sliced["times"] -= sliced["times"][0]
    return sliced


def xrobo_bag_to_pickle(bag_path: str, start_idx: int, end_idx: int) -> None:
    """Load a rosbag containing XRobo data, slice a segment, and save as pickle

    Args:
        bag_path (str): Path to the ros2 bag to load
        start_idx (int): Start slice index
        end_idx (int): End slice index
    """
    bag_path = Path(bag_path)
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    output_path = bag_path.with_name(
        f"{bag_path.name}_crop_{start_idx}_to_{end_idx}.pkl"
    )
    topic = "/xrobo/state"
    histories = load_ros2_bag(bag_path)

    if topic not in histories:
        raise ValueError(f"Topic {topic} not found in bag")

    messages = histories[topic]

    data = xrobo_messages_to_numpy(messages)
    data = slice_data(data, start_idx, end_idx)

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved {len(data['times'])} timesteps to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    xrobo_bag_to_pickle(args.bag, args.start, args.end)


if __name__ == "__main__":
    main()
