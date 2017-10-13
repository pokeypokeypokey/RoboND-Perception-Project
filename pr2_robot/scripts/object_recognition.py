#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from features import compute_all_features
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import rospkg
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml
from math import atan2


class StateEnum(object):
    def __init__(self, pick_n):
        self.look_left  = 0
        self.look_right = 1
        self.look_straight = 2
        self.finish = pick_n+3

        for i in xrange(pick_n):
            setattr(self, "pick_%i" % (i+1), (i+3))


class ObjectPicker(object):
    PLACE_POSES = {"red":   (0,  0.71, 0.605),
                   "green": (0, -0.71, 0.605)}
    ARM_TO_USE = {"red":   "left",
                  "green": "right"}
    ANGLE_LEFT  =  np.pi/2. # 0.707
    ANGLE_RIGHT = -np.pi/2. # -0.707

    FRESH_SCAN = False


    def __init__(self):
        rospy.init_node("object_picker", anonymous=True)
        self.package_url = rospkg.RosPack().get_path("pr2_robot")

        # Get/Read parameters
        self.OBJECT_LIST_PARAM = rospy.get_param('/object_list')
        param_n = len(self.OBJECT_LIST_PARAM)
        self.TEST_SCENE_NUM = {3: 1, 5: 2, 8: 3}[param_n]

        # Robot states
        self.R_STATES = StateEnum(param_n)

        # Subscriber
        self.pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, self.pcl_callback, queue_size=1)
        self.tf_listen = tf.TransformListener()

        # Publishers
        self.objects_pub = rospy.Publisher("/pr2/pcl_objects", PointCloud2, queue_size=1)
        self.table_pub = rospy.Publisher("/pr2/pcl_table", PointCloud2, queue_size=1)
        self.coll_pub  = rospy.Publisher("/pr2/3D_map/points", PointCloud2, queue_size=1)
        self.cluster_pub = rospy.Publisher("/pr2/pcl_cluster", PointCloud2, queue_size=1)
        self.object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
        self.detected_objects_pub = rospy.Publisher("/detexted_objects", DetectedObjectsArray, queue_size=1)
        self.pub_base_joint = rospy.Publisher("/pr2/world_joint_controller/command", Float64, queue_size=10)

        # Collision clouds
        if self.FRESH_SCAN:
            self.r_state = self.R_STATES.look_left # Start looking left
            self.collision_cloud_table = pcl.PointCloud_PointXYZRGB() # empty point cloud
        else:
            self.r_state = self.R_STATES.look_straight
            c_url = self.package_url + "/clouds/collision_cloud.pcd"
            self.collision_cloud_table = pcl.load_XYZRGB(c_url) # load point cloud

        # Load model
        model_url = self.package_url + "/models_classification/model_hsv_c1c2c3_size.sav"
        model = pickle.load(open(model_url, 'rb'))

        self.clf = model['classifier']
        self.encoder = LabelEncoder()
        self.encoder.classes_ = model['classes']
        self.scaler = model['scaler']

        # Initialize color_list
        get_color_list.color_list = []

        while not rospy.is_shutdown():
            rospy.spin()

    def angle_is_close(self, a1, a2, eps=0.01):
        return (abs(a1-a2) < eps)

    def downsample_cloud(self, cloud, leaf_size):
        vox = cloud.make_voxel_grid_filter()
        vox.set_leaf_size(leaf_size, leaf_size, leaf_size)
        return vox.filter()

    def combine_clouds(self, clouds, ds_leaf_size=0.01):
        # Combine into new cloud
        combined_arr = np.concatenate(([c.to_array() for c in clouds]))
        combined = pcl.PointCloud_PointXYZRGB()
        combined.from_array(combined_arr)

        # Downsample (to remove duplicates)
        return self.downsample_cloud(combined, ds_leaf_size)

    # Helper function to get surface normals
    def get_normals(self, cloud):
        get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
        return get_normals_prox(cloud).cluster

    # Helper function to create a yaml friendly dictionary from ROS messages
    def make_yaml_dict(self, test_scene_num, arm_name, object_name, pick_pose, place_pose):
        yaml_dict = {}
        yaml_dict["test_scene_num"] = test_scene_num.data
        yaml_dict["arm_name"] = arm_name.data
        yaml_dict["object_name"] = object_name.data
        yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
        yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
        return yaml_dict

    # Helper function to output to yaml file
    def send_to_yaml(self, yaml_filename, dict_list):
        data_dict = {"object_list": dict_list}
        with open(yaml_filename, 'w') as outfile:
            yaml.dump(data_dict, outfile, default_flow_style=False)

    def make_pose_message(self, position):
        msg = Pose()
        msg.position.x, msg.position.y, msg.position.z = position
        return msg

    def passthrough_cloud(self, cloud, axis, start, end):
        passthrough = cloud.make_passthrough_filter()
        passthrough.set_filter_field_name(axis)
        passthrough.set_filter_limits(start, end)
        return passthrough.filter()

    def filter_cloud(self, cloud):
        # Voxel Grid filter
        cloud_filtered = self.downsample_cloud(cloud, 0.005)

        # PassThrough filter
        cloud_filtered = self.passthrough_cloud(cloud_filtered, "z", 0.6, 0.8)

        # Outlier filter
        outlier_filter = cloud_filtered.make_statistical_outlier_filter()
        outlier_filter.set_mean_k(50)
        outlier_filter.set_std_dev_mul_thresh(0.001)
        cloud_filtered = outlier_filter.filter()

        return cloud_filtered

    def update_move_state(self, target_angle, next_state):
        # get current rotation
        try:
            _, rot = self.tf_listen.lookupTransform("/world", "/base_footprint", rospy.Time(0))
            # Get rotation about z from quaternion
            current_angle = atan2(2*(rot[0]*rot[1] + rot[2]*rot[3]), 
                                  1 - 2*(rot[1]**2 + rot[2]**2))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        # Set target
        self.pub_base_joint.publish(target_angle)

        # Check if target reached
        if self.angle_is_close(current_angle, target_angle):
            self.r_state = next_state

    def accumulate_collisions(self, table_cloud, object_cloud):
        # move dat boday
        if self.r_state == self.R_STATES.look_left:
            self.update_move_state(self.ANGLE_LEFT, self.R_STATES.look_right)

        elif self.r_state == self.R_STATES.look_right:
            self.update_move_state(self.ANGLE_RIGHT, self.R_STATES.look_straight)

        else:
            self.update_move_state(0, self.R_STATES.pick_1)

        # accumulate and store collision
        self.collision_cloud_table = self.combine_clouds((self.collision_cloud_table, 
                                                            table_cloud, object_cloud))

        self.objects_pub.publish(pcl_to_ros(object_cloud))
        self.table_pub.publish(pcl_to_ros(table_cloud))
        self.coll_pub.publish(pcl_to_ros(self.collision_cloud_table))

        # if self.r_state == self.R_STATES.pick_1 and self.FRESH_SCAN:
        #     pcl.save(self.collision_cloud_table, "collision_cloud.pcd", "pcd")


    # function to load parameters and request PickPlace service
    def pr2_mover(self, object_centroids):
        # Loop through the pick list
        yaml_out = []
        for obj in self.OBJECT_LIST_PARAM:
            # Fetch the centroid
            if obj["name"] in object_centroids:
                centroid = object_centroids[obj["name"]]
            else:
                # We're only interested in objects on the list
                continue

            # Create message components
            test_scene_num = Int32(self.TEST_SCENE_NUM)
            obj_name = String(obj["name"])
            arm_name = String(self.ARM_TO_USE[obj["group"]])
            place_pose = self.make_pose_message(self.PLACE_POSES[obj["group"]])
            pick_pose  = self.make_pose_message(map(np.asscalar, centroid))

            # yaml output
            yaml_out.append(self.make_yaml_dict(test_scene_num, arm_name, 
                                                obj_name, pick_pose, place_pose))

            # Wait for 'pick_place_routine' service to come up
            rospy.wait_for_service('pick_place_routine')

            # try:
            #     pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            #     # Send as a service request
            #     resp = pick_place_routine(test_scene_num, obj_name, arm_name, pick_pose, place_pose)

            #     print ("Response: ", resp.success)

            # except rospy.ServiceException, e:
            #     print "Service call failed: %s"%e

        # Output request parameters into yaml file
        # self.send_to_yaml(package_url + "/config/output_%i.yaml" % self.TEST_SCENE_NUM, yaml_out)

    def pick_objects(self, object_cloud):
        # Clustering
        white_cloud = XYZRGB_to_XYZ(object_cloud)
        tree = white_cloud.make_kdtree()
        ec = white_cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(0.05)
        ec.set_MinClusterSize(10)
        ec.set_MaxClusterSize(10000)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()

        # Cluster-Mask Point Cloud to visualize each cluster separately
        cluster_colour = get_color_list(len(cluster_indices))
        colour_cluster_point_list = []

        for j, indices in enumerate(cluster_indices):
            for i in indices:
                colour_cluster_point_list.append([white_cloud[i][0],
                                                  white_cloud[i][1],
                                                  white_cloud[i][2],
                                                  rgb_to_float(cluster_colour[j])])
        cluster_cloud = pcl.PointCloud_PointXYZRGB()
        cluster_cloud.from_list(colour_cluster_point_list)

        # Publish
        self.objects_pub.publish(pcl_to_ros(object_cloud))
        self.cluster_pub.publish(pcl_to_ros(cluster_cloud))

        # Classify the clusters (loop through each detected cluster one at a time)
        detected_object_labels = []
        detected_objects_list = []
        object_centroids = {}

        for idx, pts_list in enumerate(cluster_indices):
            # Grab the points for the cluster
            pcl_cluster = object_cloud.extract(pts_list)
            pcl_cluster_r = pcl_to_ros(pcl_cluster)

            # Compute the associated feature vector
            feature = compute_all_features(pcl_cluster_r, self.get_normals(pcl_cluster_r))

            # Make the prediction, retrieve the label for the result
            # and add it to detected_objects_labels list
            prediction = self.clf.predict(self.scaler.transform(feature.reshape(1, -1)))
            label = self.encoder.inverse_transform(prediction)[0]
            detected_object_labels.append(label)

            # Publish a label into RViz
            label_pos = list(white_cloud[pts_list[0]])
            label_pos[2] += .4
            self.object_markers_pub.publish(make_label(label, label_pos, idx))

            # Add the detected object to the list of detected objects.
            do = DetectedObject()
            do.label = label
            do.cloud = pcl_cluster_r
            detected_objects_list.append(do)

            # Store centroid
            object_centroids[label] = np.mean(pcl_cluster.to_array(), axis=0)[:3]
            
        # Publish the list of detected objects
        # rospy.loginfo('Detected {} objects: {}'.format(len(detected_object_labels), detected_object_labels))
        self.detected_objects_pub.publish(detected_objects_list)

        try:
            self.pr2_mover(object_centroids)
        except rospy.ROSInterruptException:
            pass

    # Callback function for your Point Cloud Subscriber
    def pcl_callback(self, pcl_msg):
        cloud = ros_to_pcl(pcl_msg)
        cloud_filtered = self.filter_cloud(cloud)

        # RANSAC Plane Segmentation
        seg = cloud_filtered.make_segmenter()
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC) 
        seg.set_distance_threshold(0.01)
        inliers, _ = seg.segment()

        # Extract inliers and outliers
        object_cloud = cloud_filtered.extract(inliers, negative=True)
        table_cloud  = cloud_filtered.extract(inliers, negative=False)

        if self.r_state < self.R_STATES.pick_1:
            # build collision map
            object_cloud = self.passthrough_cloud(object_cloud, "x", -0.4, 0.4) # boxes

            self.accumulate_collisions(table_cloud, object_cloud)

        elif self.r_state < self.R_STATES.finish:
            # identify objects and pick them up
            object_cloud = self.passthrough_cloud(object_cloud, "y", -0.4, 0.4) # objects

            self.pick_objects(object_cloud)
        # else:
            # do nothing, we're done


if __name__ == '__main__':
    try:
        ObjectPicker()
    except rospy.ROSInterruptException:
        pass
