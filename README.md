## Project: Perception Pick & Place

---

# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Perception Pipeline
#### 1. Filtering and RANSAC plane fitting.
First the depth is downsampled (leaf size of 0.005), to reduce the processing load. Next two simple passthrough filters (0.6 < z < 0.8 and -0.4 < y < 0.4) are applied, to restrict the attention to just the table top and the objects on it. Finally a statistical outlier filter (50 neighbours, 0.001 std deviation multiplier) is applied to get rid of speckle noise.

To identify the table top, and seperate it from the objects, a plane is fit to the cloud using RANSAC. The fitted plane is assumed to be the table top. 

#### 2. Clustering for segmentation.
To segment the point cloud into individual objects, Euclidean clustering is applied with a tolerance of 0.05. 

#### 3. Feature extraction and object recognition.
The features used fall into 4 groups:
* Histogram of HSV colour values (32 bins)
* Histogram of c1c2c3 colour values (32 bins)
* Histogram of surface normals (12 bins)
* Cluster sizes (x, y and z lengths)

The c1c2c3 colour space is described [here](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.43.6496&rep=rep1&type=pdf) and reviewed [here](https://arxiv.org/abs/1702.05421).

An SVM classifier with a linear kernel was trained using 100 clouds from each object (at random orientations), achieving an average cross-validation accuracy of 0.93 across 5 folds.

![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)

### Pick and Place Setup
#### 1. Robot states.
Before the pick and place can begin the robot needs to map out its surrounds. First it looks left, then right, then finally straight ahead to examine the objects.

Movement is achieved by publishing angles to `"/pr2/world_joint_controller/command"` and the current robot direction is monitored with `tf.TransformListener().lookupTransform("/world", "/base_footprint", rospy.Time(0))`.

#### 2. Collision cloud.
To aid motion planning, a point cloud describing all obstacles is published to `"/pr2/3d_map/points"`. While the robot is looking around, the table cloud is accumulated. 

When an object is selected for pickup, all other objects are added to the collision cloud.

#### 3. Pick/Place Request
Objects are matched with pick/place requests. A service request is generated which includes the object name, the arm to use, the object's location and where to drop the object. The object's location is calculated as the centroid of the associated point cloud. The drop location is somewhere above the dropbox described by the pick/place request, with minor variations between objects to stop them falling right on top of each other.

### Challenge World