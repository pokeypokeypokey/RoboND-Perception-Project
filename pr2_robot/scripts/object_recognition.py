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


PLACE_POSES = {"red":   (0,  0.71, 0.605),
               "green": (0, -0.71, 0.605)}
ARM_TO_USE = {"red":   "left",
              "green": "right"}


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

def make_pose_message(position):
    msg = Pose()
    msg.position.x, msg.position.y, msg.position.z = position
    return msg

def passthrough_cloud(cloud, axis, start, end):
    passthrough = cloud.make_passthrough_filter()
    passthrough.set_filter_field_name(axis)
    passthrough.set_filter_limits(start, end)
    return passthrough.filter()

def filter_cloud(cloud):
    # Voxel Grid filter
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.005
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # PassThrough filter
    cloud_filtered = passthrough_cloud(cloud_filtered, "z", 0.6, 0.8)
    cloud_filtered = passthrough_cloud(cloud_filtered, "y", -0.4, 0.4)

    # Outlier filter
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(50)
    outlier_filter.set_std_dev_mul_thresh(0.001)
    cloud_filtered = outlier_filter.filter()

    return cloud_filtered

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    cloud = ros_to_pcl(pcl_msg)
    cloud_filtered = filter_cloud(cloud)

    # RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC) 
    seg.set_distance_threshold(0.01)
    inliers, _ = seg.segment()

    # Extract inliers and outliers
    object_cloud = cloud_filtered.extract(inliers, negative=True)
    table_cloud  = cloud_filtered.extract(inliers, negative=False)

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
    pcl_objects_pub.publish(pcl_to_ros(object_cloud))
    pcl_table_pub.publish(pcl_to_ros(table_cloud))
    pcl_cluster_pub.publish(pcl_to_ros(cluster_cloud))

    # Classify the clusters (loop through each detected cluster one at a time)
    detected_object_labels = []
    detected_objects_list = []
    object_centroids = {}

    for idx, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = object_cloud.extract(pts_list)
        pcl_cluster_r = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        feature = compute_all_features(pcl_cluster_r, get_normals(pcl_cluster_r))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_object_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, idx))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = pcl_cluster_r
        detected_objects_list.append(do)

        # Store centroid
        object_centroids[label] = np.mean(pcl_cluster.to_array(), axis=0)[:3]
        

    # Publish the list of detected objects
    # rospy.loginfo('Detected {} objects: {}'.format(len(detected_object_labels), detected_object_labels))
    detected_objects_pub.publish(detected_objects_list)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(object_centroids)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_centroids):
    # TODO: Rotate PR2 in place to capture side tables for the collision map
    pub_base_joint.publish(-np.pi/2.)

    # Loop through the pick list
    yaml_out = []
    for obj in OBJECT_LIST_PARAM:
        # Fetch the centroid
        if obj["name"] in object_centroids:
            centroid = object_centroids[obj["name"]]
        else:
            # We're only interested in objects on the list
            continue

        # Create message components
        test_scene_num = Int32(TEST_SCENE_NUM)
        obj_name = String(obj["name"])
        arm_name = String(ARM_TO_USE[obj["group"]])
        place_pose = make_pose_message(PLACE_POSES[obj["group"]])
        pick_pose = make_pose_message(map(np.asscalar, centroid))

        # yaml output
        yaml_out.append(make_yaml_dict(test_scene_num, arm_name, obj_name, pick_pose, place_pose))

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
    send_to_yaml(package_url + "/config/output_%i.yaml" % TEST_SCENE_NUM, yaml_out)


if __name__ == '__main__':
    rospy.init_node("clustering", anonymous=True)
    # Get/Read parameters
    OBJECT_LIST_PARAM = rospy.get_param('/object_list')
    TEST_SCENE_NUM = {3: 1, 5: 2, 8: 3}[len(OBJECT_LIST_PARAM)]

    # Subscriber
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Publishers
    pcl_objects_pub = rospy.Publisher("/pr2/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub   = rospy.Publisher("/pr2/pcl_table",   PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pr2/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detexted_objects", DetectedObjectsArray, queue_size=1)
    pub_base_joint = rospy.Publisher("/pr2/world_joint_controller/command", Float64, queue_size=10)

    # Load model
    rospack = rospkg.RosPack()
    package_url = rospack.get_path("pr2_robot")
    model_url = package_url + "/models_classification/model_hsv_c1c2c3_size.sav"

    model = pickle.load(open(model_url, 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    while not rospy.is_shutdown():
        rospy.spin()
