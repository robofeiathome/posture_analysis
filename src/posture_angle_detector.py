#!/usr/bin/env python3
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import mediapipe as mp
import numpy as np
from posture_analysis.srv import PointingDirection

class PostureAnalyzer:
    def __init__(self):
        self.service = rospy.Service('pointing_direction', PointingDirection, self.handler)
        self.mediapipe_drawing = mp.solutions.drawing_utils
        self.mediapipe_pose = mp.solutions.pose
        self.bridge = CvBridge()
        self.camera_topic = rospy.get_param('~camera_topic', '/usb_cam/image_raw')
        self.camera_subscriber = rospy.Subscriber(self.camera_topic, Image, self.image_callback)
        self.current_image = None
        self.last_pointing_direction = None

    def image_callback(self, image):
        """Image callback"""
        # Store value on a private attribute
        self.current_image = image

    @staticmethod
    def calculate_angle(point_a, point_b, point_c):
        vector_ba = np.array(point_a) - np.array(point_b)
        vector_bc = np.array(point_c) - np.array(point_b)

        cosine_angle = np.dot(vector_ba, vector_bc) / (np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def publish_pointing_side(self, right_shoulder_angle, right_elbow_angle, left_shoulder_angle, left_elbow_angle):

        if (right_shoulder_angle <= 16 and left_shoulder_angle <= 16):
            pass
        elif (left_elbow_angle <= 130 and right_elbow_angle <= 130):
            pass 
        elif (right_shoulder_angle >= 15 and right_elbow_angle > 130):
            if (left_shoulder_angle <= 15) or (left_elbow_angle < right_elbow_angle):
                return 'left'
            else:
                return 'right'
        elif (left_shoulder_angle >= 15 and left_elbow_angle > 130):
            if (right_shoulder_angle <= 15) or (left_elbow_angle > right_elbow_angle):
                return 'right'
            else:
                return 'left'
        else:
            pass

        return None

    def analyze_posture_in_frame(self, frame):
        with self.mediapipe_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as pose:
            pose_detection_results = pose.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            try:
                landmarks = pose_detection_results.pose_landmarks.landmark

                left_shoulder_position = self.get_landmark_position(landmarks, self.mediapipe_pose.PoseLandmark.LEFT_SHOULDER)
                left_elbow_position = self.get_landmark_position(landmarks, self.mediapipe_pose.PoseLandmark.LEFT_ELBOW)
                left_wrist_position = self.get_landmark_position(landmarks, self.mediapipe_pose.PoseLandmark.LEFT_WRIST)

                right_shoulder_position = self.get_landmark_position(landmarks, self.mediapipe_pose.PoseLandmark.RIGHT_SHOULDER)
                right_elbow_position = self.get_landmark_position(landmarks, self.mediapipe_pose.PoseLandmark.RIGHT_ELBOW)
                right_wrist_position = self.get_landmark_position(landmarks, self.mediapipe_pose.PoseLandmark.RIGHT_WRIST)

                left_hip_position = self.get_landmark_position(landmarks, self.mediapipe_pose.PoseLandmark.LEFT_HIP)

                right_hip_position = self.get_landmark_position(landmarks, self.mediapipe_pose.PoseLandmark.RIGHT_HIP)

                right_shoulder_angle = self.calculate_angle(right_hip_position, right_shoulder_position, right_elbow_position)
                left_shoulder_angle = self.calculate_angle(left_hip_position, left_shoulder_position, left_elbow_position)
                right_elbow_angle = self.calculate_angle(right_shoulder_position, right_elbow_position, right_wrist_position)
                left_elbow_angle = self.calculate_angle(left_shoulder_position, left_elbow_position, left_wrist_position)

                self.add_angle_text_to_frame(frame, right_shoulder_angle, right_shoulder_position)
                self.add_angle_text_to_frame(frame, left_shoulder_angle, left_shoulder_position)
                self.add_angle_text_to_frame(frame, right_elbow_angle, right_elbow_position)
                self.add_angle_text_to_frame(frame, left_elbow_angle, left_elbow_position)
            except Exception as error:
                print(error)

            self.mediapipe_drawing.draw_landmarks(frame, pose_detection_results.pose_landmarks, self.mediapipe_pose.POSE_CONNECTIONS,
                                                  self.mediapipe_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                                  self.mediapipe_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                                  )

        direction = self.publish_pointing_side(right_shoulder_angle, right_elbow_angle, left_shoulder_angle, left_elbow_angle)
        return direction

    @staticmethod
    def get_landmark_position(landmarks, landmark_type):
        return [landmarks[landmark_type.value].x, landmarks[landmark_type.value].y]

    @staticmethod
    def add_angle_text_to_frame(frame, angle, position):
        cv2.putText(frame, str(int(angle)),
                    tuple(np.multiply(position, [frame.shape[1], frame.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA
                    )

    def handler(self, req):
        try:
            for i in range(5):
                direction = self.analyze_posture_in_frame(self.current_image)
                if direction is not None:
                    break
            return direction
        except Exception as error:
            print(error)

if __name__ == '__main__':
    rospy.init_node('pointing_direction', log_level=rospy.ERROR)
    PostureAnalyzer()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass