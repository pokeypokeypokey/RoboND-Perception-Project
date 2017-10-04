import numpy as np
import pickle
import rospy
import pcl
import glob
import itertools
import matplotlib.pyplot as plt

from sensor_stick.pcl_helper import *
# from sensor_stick.features import compute_color_histograms
# from sensor_stick.features import compute_normal_histograms
# from sensor_stick.features import compute_size
from features import compute_all_features
from sensor_stick.srv import GetNormals
import sensor_msgs.point_cloud2 as pc2

from sklearn import svm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import cross_validation
from sklearn import metrics


MODELS_URL = "/home/nielen/RoboND-Perception-Exercises/training clouds/"
MODELS_1 = ["biscuits", "soap", "soap2"]
MODELS_2 = ["biscuits", "soap", "soap2", "book", "glue"]
MODELS_3 = ["biscuits", "soap", "soap2", "book", "glue", "sticky_notes", "snacks", "eraser"]


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{0:.2f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == "__main__":
    rospy.init_node('train_from_file_node')
    pcl_objects_pub = rospy.Publisher("/pr2/pcl_objects", PointCloud2, queue_size=1)

    # Features
    labeled_features = []

    for model_name in MODELS_3:
        print "Features: %s..." % model_name
        for c_url in glob.glob(MODELS_URL + model_name + "/*.pcd"):
            cloud = pcl.load_XYZRGB(c_url)
            cloud_r = pcl_to_ros(cloud)

            pcl_objects_pub.publish(cloud_r)

            # chists = compute_color_histograms(cloud_r, using_hsv=True)
            # nhists = compute_normal_histograms(get_normals(cloud_r))
            # bounds = compute_size(cloud_r)
            # feature = np.concatenate((chists, nhists, bounds))
            feature = compute_all_features(cloud_r, get_normals(cloud_r))
            labeled_features.append([feature, model_name])

    pickle.dump(labeled_features, open('training_set.sav', 'wb'))

    # Train SVM
    # Format the features and labels for use with scikit learn
    feature_list = []
    label_list = []

    for item in labeled_features:
        if np.isnan(item[0]).sum() < 1:
            feature_list.append(item[0])
            label_list.append(item[1])

    X = np.array(feature_list)
    X_scaler = StandardScaler().fit(X)
    X_train = X_scaler.transform(X)

    # Convert label strings to numerical encoding
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(np.array(label_list))

    # Create classifier
    print "Training SVM..."
    clf = svm.SVC(kernel='linear')
    kf = cross_validation.KFold(len(X_train),
                                n_folds=5,
                                shuffle=True,
                                random_state=1)
    scores = cross_validation.cross_val_score(cv=kf, estimator=clf,
                                              X=X_train, y=y_train,
                                              scoring='accuracy')
    print('Scores: ' + str(scores))
    print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), 2*scores.std()))

    # Gather predictions
    predictions = cross_validation.cross_val_predict(cv=kf, estimator=clf,
                                                     X=X_train, y=y_train)

    accuracy_score = metrics.accuracy_score(y_train, predictions)
    print('accuracy score: '+ str(accuracy_score))

    confusion_matrix = metrics.confusion_matrix(y_train, predictions)

    class_names = encoder.classes_.tolist()


    #Train the classifier
    clf.fit(X=X_train, y=y_train)

    model = {'classifier': clf, 'classes': encoder.classes_, 'scaler': X_scaler}

    # Save classifier to disk
    pickle.dump(model, open('model.sav', 'wb'))

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=encoder.classes_, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()