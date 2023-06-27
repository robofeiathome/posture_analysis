#!/usr/bin/env python3
import rospy
from posture_angle_detector import PostureAnalyzer
from posture_analysis.srv import PointingDirection, PointingDirectionResponse


class PostureService:
    def __init__(self):
        self.analyzer = PostureAnalyzer()
        self.service = rospy.Service('pointing_direction', PointingDirection, self.handle_pointing_direction)

    def handle_pointing_direction(self):
        direction = self.analyzer.last_pointing_direction
        return PointingDirectionResponse(direction if direction else 'unknown')


if __name__ == '__main__':
    rospy.init_node('posture_service')
    posture_service = PostureService()
    rospy.spin()
