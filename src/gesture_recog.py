#!/usr/bin/env python3
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import rospkg
import mediapipe as mp
import numpy as np
from posture_analysis.srv import PointingDirection

class GestureAnalyzer:
    def __init__(self):
        self.service = rospy.Service('pointing_direction', PointingDirection, self.handler)
        self.bridge = CvBridge()
        self.camera_topic = rospy.get_param('~camera_topic', '/camera/color/image_rect_color')
        self.camera_subscriber = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        self.current_image = None
        self.pointing_hand = None
        self.actual_hand_value = 0
        self.last_pointing_direction = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

    def image_callback(self, image):
        self.current_image = image

    def calculate_pointing_direction(self, frame):
        if self.current_image is not None:
            try:
                frame = self.bridge.imgmsg_to_cv2(self.current_image, "bgr8")
                h,w,_ = frame.shape
                points = []
                with self.mp_hands.Hands(
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    frame.flags.writeable = False
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame)
                    # Draw the hand annotations on the image.
                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if results.multi_handedness:
                        for hand in results.multi_handedness:
                            for hand_landmarks in results.multi_hand_landmarks:
                                    
                                    for id, coord in enumerate(hand_landmarks.landmark):
                                        cx, cy = int(coord.x*w), int(coord.y*h)
                                        points.append((cx,cy))
                        
                            index_fingertip = 8
                            if points:
                                if abs(points[index_fingertip][0] - points[0][0]) > self.actual_hand_value:
                                    self.actual_hand_value = abs(points[index_fingertip][0] - points[0][0])
                            if points[index_fingertip][0] - points[0][0] > 0:
                                return("right")
                            else:
                                return("left")
                        
            except Exception as e:
                print(e)

        return None

    def handler(self, req):
        try:
            seq = []
            for i in range(10):
                direction = self.calculate_pointing_direction(self.current_image)
                if direction is not None:
                    seq.append(direction)
            for i in seq:
                if seq[0] == "left":
                    if i != "left":
                        return None
                else:
                    if i != "right":
                        return None
            return direction
        except Exception as error:
            print(error)

if __name__ == '__main__':
    rospy.init_node('pointing_direction', log_level=rospy.ERROR)
    GestureAnalyzer()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass