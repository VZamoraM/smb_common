import rospy
import sys
import time

def delay():
    rospy.init_node('delay_node')
    sleep_time = rospy.get_param('~sleep_time', 10)  # Default to 10 seconds
    time.sleep(sleep_time)

if __name__ == '__main__':
    delay()
