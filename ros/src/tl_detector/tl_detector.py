#!/usr/bin/env python
import rospy
import time

from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3
SAVE_IMAGES = True
LOGGING_THROTTLE_FACTOR = 10  # Only log at this rate (1 / Hz)
IMAGE_PROCESS_FACTOR = 8

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.has_image = False
        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=2)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=8)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=2)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.process_count = 0
        self.image_count = 0

        rospy.spin()
#         self.loop()

    def loop(self):
        rate = rospy.Rate(3)
        while not rospy.is_shutdown():
            if self.camera_image is not None:
                if(self.image_count % IMAGE_PROCESS_FACTOR == 0):
                    light_wp, state = self.process_traffic_lights()
                    rospy.logwarn("DETECT-0:light_wp={} state={}".format(light_wp,self.get_state_string(state)))

                    '''
                    Publish upcoming red lights at camera frequency.
                    Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
                    of times till we start using it. Otherwise the previous stable state is
                    used.
                    '''
                    if self.state != state:
                        self.state_count = 0
                        self.state = state
                    elif self.state_count >= STATE_COUNT_THRESHOLD:
                        self.last_state = self.state
                        rospy.logwarn("DETECT-1:light_wp={} state={}".format(light_wp,self.get_state_string(state)))
                        light_wp = light_wp if state == TrafficLight.RED else -1
                        self.last_wp = light_wp
                        rospy.logwarn("DETECT-3: light_wp={}".format(light_wp))
                        self.upcoming_red_light_pub.publish(Int32(light_wp))
                    else:
                        self.upcoming_red_light_pub.publish(Int32(self.last_wp))

                    self.state_count += 1
                self.image_count +=1

    def image_callback(self, msg):
        self.has_image = True
        self.camera_image = msg
    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)


    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        if(self.image_count % IMAGE_PROCESS_FACTOR == 0):
            self.has_image = True
            self.camera_image = msg
            light_wp, state = self.process_traffic_lights()
            rospy.logwarn("DETECT-0:light_wp={} state={}".format(light_wp,self.get_state_string(state)))

            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                rospy.logwarn("DETECT-1:light_wp={} state={}".format(light_wp,self.get_state_string(state)))
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                rospy.logwarn("DETECT-3: light_wp={}".format(light_wp))
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))

            self.state_count += 1
        self.image_count +=1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        light_state = self.light_classifier.get_classification(cv_image)
#         light_state = light.state
        #Get classification
#         if SAVE_IMAGES and (self.process_count % LOGGING_THROTTLE_FACTOR == 0):
#                 save_file = "../../../imgs/{}-{:.0f}.jpeg".format(self.get_state_string(light_state), (time.time() * 100))
#                 cv2.imwrite(save_file, cv_image)

        return light_state
    def get_state_string(self, state):
        out = "unknown"
        if state == TrafficLight.GREEN:
            out = "green"
        elif state == TrafficLight.YELLOW:
            out = "yellow"
        elif state == TrafficLight.RED:
            out = "red"
        return out

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        light_wp_idx = -1
        state = TrafficLight.UNKNOWN

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            #car_position = self.get_closest_waypoint(self.pose.pose)
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx
        if closest_light:
            self.process_count += 1
            state = self.get_light_state(closest_light)
            if(self.process_count % LOGGING_THROTTLE_FACTOR == 0):
                rospy.logwarn("DETECT: line_wp_idx={}, state={}".format(line_wp_idx, self.get_state_string(state)))

        return line_wp_idx, state

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
