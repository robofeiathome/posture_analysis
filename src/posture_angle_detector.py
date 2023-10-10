#!/usr/bin/env python3
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import mediapipe as mp
import numpy as np


class PostureAnalyzer:
    def __init__(self):
        self.mediapipe_drawing = mp.solutions.drawing_utils
        self.mediapipe_pose = mp.solutions.pose
        self.bridge = CvBridge()
        self.camera_topic = rospy.get_param('~camera_topic', '/usb_cam/image_raw')
        self.camera_subscriber = rospy.Subscriber(self.camera_topic, Image, self.callback_on_new_image)
        self.pointing_direction_publisher = rospy.Publisher('/pointing_direction', String, queue_size=10)
        self.processed_image_publisher = rospy.Publisher('/processed_image', Image, queue_size=10)
        self.last_pointing_direction = None

    def callback_on_new_image(self, image_message):
        try:
            image_frame = self.bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough')
            image_frame_copy = np.copy(image_frame)
            processed_image = self.analyze_posture_in_frame(image_frame_copy)
            processed_image_message = self.bridge.cv2_to_imgmsg(processed_image, encoding='bgr8')
            self.processed_image_publisher.publish(processed_image_message)

        except Exception as error:
            print(error)
            return

    @staticmethod
    def calculate_angle(point_a, point_b, point_c):
        vector_ba = np.array(point_a) - np.array(point_b)
        vector_bc = np.array(point_c) - np.array(point_b)

        cosine_angle = np.dot(vector_ba, vector_bc) / (np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def publish_pointing_side(self, right_shoulder_angle, left_shoulder_angle):
        if right_shoulder_angle > 20 > left_shoulder_angle:
            self.last_pointing_direction = 'left'
            self.pointing_direction_publisher.publish('left')
        elif right_shoulder_angle < 20 < left_shoulder_angle:
            self.last_pointing_direction = 'right'
            self.pointing_direction_publisher.publish('right')

    def analyze_posture_in_frame(self, frame):
        with self.mediapipe_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
            except Exception as error:
                print(error)

            self.mediapipe_drawing.draw_landmarks(frame, pose_detection_results.pose_landmarks, self.mediapipe_pose.POSE_CONNECTIONS,
                                                  self.mediapipe_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                                  self.mediapipe_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                                  )

            side = self.publish_pointing_side(right_shoulder_angle, left_shoulder_angle)

            return frame

    @staticmethod
    def get_landmark_position(landmarks, landmark_type):
        return [landmarks[landmark_type.value].x, landmarks[landmark_type.value].y]

    @staticmethod
    def add_angle_text_to_frame(frame, angle, position):
        cv2.putText(frame, str(int(angle)),
                    tuple(np.multiply(position, [frame.shape[1], frame.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA
                    )


if __name__ == '__main__':
    posture_analyzer = PostureAnalyzer()
    rospy.spin()
