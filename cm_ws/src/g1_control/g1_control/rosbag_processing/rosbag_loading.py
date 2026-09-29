"""Reading and plotting data from ROS2 bag files"""

import argparse
import array
from typing import List, Dict, Any
from pprint import pprint

import rclpy
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def load_ros2_bag(bag_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Reads a ros2 bag file and returns a dictionary of topic histories.

    Args:
        bag_path (str): Path to the ros2 bag file directory

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary where keys are the topic names,
            and values are lists of message dictionaries (with timestamps).
    """
    # Initialize rclpy if not already done
    if not rclpy.ok():
        rclpy.init()

    bag_path = str(bag_path)  # in case it's a Path

    # Create reader
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Get topic types
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_info.name: topic_info.type for topic_info in topic_types}

    # Read messages
    topic_histories = {t.name: [] for t in topic_types}
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        msg_type = type_map[topic]
        msg_class = get_message(msg_type)
        msg = deserialize_message(data, msg_class)
        msg_dict = message_to_dict(msg)
        msg_dict["timestamp"] = timestamp
        topic_histories[topic].append(msg_dict)

    return topic_histories


def message_to_dict(msg) -> Dict[str, Any]:
    """Convert ROS message to dictionary recursively

    Args:
        msg: ROS message object

    Returns:
        Dict[str, Any]: Dictionary representation of the message
    """
    if hasattr(msg, "_fields_and_field_types"):
        converted = {}
        for field in msg._fields_and_field_types.keys():  # pylint: disable=protected-access
            value = getattr(msg, field)
            if isinstance(value, list):
                converted[field] = [message_to_dict(item) for item in value]
            elif isinstance(value, array.array):
                converted[field] = value.tolist()
            else:
                converted[field] = message_to_dict(value)
        return converted
    return msg


def main():
    parser = argparse.ArgumentParser(description="Load a ROS 2 bag into python")
    parser.add_argument(
        "--bag-path", type=str, required=True, help="Path to the ROS 2 bag directory"
    )
    args = parser.parse_args()

    rclpy.init()
    data = load_ros2_bag(args.bag_path)
    print(f"\nSuccessfully loaded data from {args.bag_path}")
    print(f"\nContains topics: {list(data.keys())}")
    for topic, messages in data.items():
        print(f"\nTopic: {topic}, Number of messages: {len(messages)}")
        if messages:
            print("First message: ")
            pprint(messages[0])

    rclpy.shutdown()


if __name__ == "__main__":
    main()
