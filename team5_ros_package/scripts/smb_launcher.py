#!/usr/bin/env python3

import subprocess
import time
import os
import signal
import threading
import rospkg
import yaml

def load_config(config_file):
    """Load configuration from a YAML file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def launch_ros_node(launch_file, args=[]):
    """Function to launch a ROS node using a given launch file."""
    command = ['roslaunch', launch_file] + args
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def monitor_ros_node(process, launch_file, args=[]):
    """Function to monitor a ROS node and respawn it if it crashes."""
    while True:
        retcode = process.poll()
        if retcode is not None:
            print(f"Process {process.pid} terminated with exit code {retcode}. Respawning...")
            process = launch_ros_node(launch_file, args)
        time.sleep(1)

def shutdown_ros_node(process):
    """Function to shut down a ROS node."""
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    print(f"Process {process.pid} terminated.")

if __name__ == "__main__":
    rospack = rospkg.RosPack()
    
    CONFIG = rospack.get_path("team5_ros_package") + "/config/exploration_robot.yaml"
    config = load_config(CONFIG)

    launch_files =config['launch_files']
    launch_delay = config['launch_delay']
    shutdown_delay = config['shutdown_delay']

    # List to hold the process objects for all launched nodes
    processes = []

    try:
        for item in launch_files:
            launch_file = rospack.get_path(item.get('package', '')) + item['launch_file']
            args = item.get('args', [])
            # Launch each ROS node with a delay between them
            process = launch_ros_node(launch_file, args)
            processes.append(process)
            print(f"ROS node launched with PID {process.pid} using {launch_file} with arguments {args}")

            # Start a separate thread to monitor each ROS node
            monitor_thread = threading.Thread(target=monitor_ros_node, args=(process, launch_file, args))
            monitor_thread.daemon = True
            monitor_thread.start()

            # Wait before launching the next ROS node
            time.sleep(launch_delay)

        # Wait for the specified delay before shutting down all ROS nodes
        # time.sleep(shutdown_delay)

    except KeyboardInterrupt:
        pass
    finally:
        # Shut down all ROS nodes
        for process in processes:
            shutdown_ros_node(process)
        print("All ROS nodes shut down successfully.")
