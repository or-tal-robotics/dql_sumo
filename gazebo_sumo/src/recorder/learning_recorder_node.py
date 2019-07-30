#!/usr/bin/env python

import rospy
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Int16

episode_count = 0
bridge = CvBridge()
ii = 1
jj = 1
out_robot = []
out_gazebo = []
pathOut_robot = '/home/lab/openai_ws/src/dql_sumo/gazebo_sumo/src/video/learning_record_robot'
pathOut_gazebo = '/home/lab/openai_ws/src/dql_sumo/gazebo_sumo/src/video/learning_record_gazebo'
record_frq = 10

def counter_cb(msg):
    global episode_count
    episode_count = msg.data

def robot_image_callback(msg):
    global episode_count, bridge, pathOut_robot, ii, out_robot, record_frq
    if (episode_count % record_frq) == 0:
        img = bridge.imgmsg_to_cv2(msg,"rgb8")
        file_name = pathOut_robot+str(episode_count)+'.avi'
        if ii == 1:
            print('Start recording robot camera!, episode:'+str(episode_count))
            out_robot = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (640,480))
            ii = 0
        img = cv2.resize(img,(640,480))
        out_robot.write(img)

    if ((episode_count-1) % record_frq) == 0 and ii == 0:
        print("Finish recording robot camera!")
        out_robot.release()
        ii = 1


def gazebo_image_callback(msg):
    global episode_count, bridge, pathOut_gazebo, jj, out_gazebo, record_frq
    if (episode_count % record_frq) == 0:
        img = bridge.imgmsg_to_cv2(msg,"rgb8")
        file_name = pathOut_gazebo+str(episode_count)+'.avi'
        if jj == 1:
            print('Start recording gazebo camera!, episode:'+str(episode_count))
            out_gazebo = cv2.VideoWriter(file_name,cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (640,480))
            jj = 0
        img = cv2.resize(img,(640,480))
        out_gazebo.write(img)

    if ((episode_count-1) % record_frq) == 0 and jj == 0:
        print("Finish recording gazebo camera!")
        out_gazebo.release()
        jj = 1


if __name__ == '__main__':
    print("Initializing recorder node....")
    rospy.init_node("learning_recorder_node")
    rospy.Subscriber('/episode_counter',Int16, counter_cb)
    rospy.Subscriber("/camera/rgb/image_raw", Image, robot_image_callback)
    rospy.Subscriber("/gazebo/image_raw", Image, gazebo_image_callback)
    rospy.wait_for_message("/gazebo/image_raw", Image)
    rospy.wait_for_message("/camera/rgb/image_raw", Image)
    rospy.wait_for_message('/episode_counter',Int16)
    print("Finish initializing, Starts recording!")
    rospy.spin()

