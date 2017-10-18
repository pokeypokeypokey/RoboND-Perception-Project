## Project: Perception Pick & Place

---

[//]: # (Image References)

[cloud_r]: ./misc_images/cloud_raw.png
[cloud_p]: ./misc_images/cloud_passthrough.png
[cloud_f]: ./misc_images/cloud_filtered.png
[cloud_o]: ./misc_images/cloud_objects.png
[cloud_t]: ./misc_images/cloud_table.png
[cloud_s]: ./misc_images/cloud_segmented.png

[class_c]: ./misc_images/class_confusion.png
[class_1]: ./misc_images/classified_1.png
[class_2]: ./misc_images/classified_2.png
[class_3]: ./misc_images/classified_3.png


## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points

---

### Perception Pipeline

##### Raw input cloud
![params][cloud_r]

#### 1. Filtering and RANSAC plane fitting.
First the depth is downsampled (leaf size of 0.005), to reduce the processing load. Next two simple passthrough filters (0.6 < z < 0.8 and -0.4 < y < 0.4) are applied, to restrict the attention to just the table top and the objects on it. 

![params][cloud_p]

Finally a statistical outlier filter (50 neighbours, 0.001 std deviation multiplier) is applied to get rid of speckle noise.

![params][cloud_f]

To identify the table top, and seperate it from the objects, a plane is fit to the cloud using RANSAC. The fitted plane is assumed to be the table top. 

![params][cloud_o]

#### 2. Clustering for segmentation.
To segment the point cloud into individual objects, Euclidean clustering is applied with a tolerance of 0.05. 

![params][cloud_s]

#### 3. Feature extraction and object recognition.
The features used fall into 4 groups:
* Histogram of HSV colour values (32 bins)
* Histogram of c1c2c3 colour values (32 bins)
* Histogram of surface normals (12 bins)
* Cluster sizes (x, y and z lengths)

The c1c2c3 colour space is described [here](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.43.6496&rep=rep1&type=pdf) and reviewed [here](https://arxiv.org/abs/1702.05421).

An SVM classifier with a linear kernel was trained using 100 clouds from each object (at random orientations for each cloud), achieving an average cross-validation accuracy of 0.93 across 5 folds.

![params][class_c]

##### World 1
![params][class_1]

##### World 2
![params][class_2]

##### World 3
![params][class_3]

### Pick and Place Setup
#### 1. Robot states.
Before the pick and place can begin the robot needs to map out its surrounds. First it looks left, then right, then finally straight ahead to examine the objects.

Movement is achieved by publishing angles to `"/pr2/world_joint_controller/command"` and the current robot direction is monitored with `tf.TransformListener().lookupTransform("/world", "/base_footprint", rospy.Time(0))`.

#### 2. Collision cloud.
To aid motion planning, a point cloud describing all obstacles is published to `"/pr2/3d_map/points"`. While the robot is looking around, the table cloud is accumulated. 

When an object is selected for pickup, all other objects are added to the collision cloud.

#### 3. Pick/Place Request
Objects are matched with pick/place requests. A service request is generated which includes the object name, the arm to use, the object's location and where to drop the object. The object's location is calculated as the centroid of the associated point cloud. The drop location is somewhere above the dropbox described by the pick/place request, with minor variations between objects to stop them falling right on top of each other.