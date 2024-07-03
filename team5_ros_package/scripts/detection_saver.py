import rospy
import tf
import csv
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo
from object_detection_msgs.msg import ObjectDetectionInfoArray

class DetectionSaver:
    def __init__(self):
        rospy.init_node('detection_saver', anonymous=True)

        # self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.7)
        # self.csv_file_path = rospy.get_param('~csv_file_path', '/tmp/detections.csv')

        self.confidence_threshold =  0.7
        self.csv_file_path = '/workspaces/rss_workspace/data/object_detections.csv'

        self.listener = tf.TransformListener()

        self.csv_file = open(self.csv_file_path, mode='w')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['class_id', 'id', 'min_x', 'min_y', 'max_x', 'max_y', 'confidence', 'pose_x', 'pose_y', 'pose_z'])

        rospy.Subscriber('/object_detector/detection_info', ObjectDetectionInfoArray, self.callback)

    def callback(self, data):
        header = data.header
        for info in data.info:
            if info.confidence >= self.confidence_threshold:
                try:
                    # Transform position to global frame
                    (trans, rot) = self.listener.lookupTransform('world_graph_msf', header.frame_id, rospy.Time(0))
                    transformed_position = tf.transformations.concatenate_matrices(
                        tf.transformations.translation_matrix(trans),
                        tf.transformations.quaternion_matrix(rot),
                        tf.transformations.translation_matrix([info.position.x, info.position.y, info.position.z])
                    )

                    pose_x, pose_y, pose_z = tf.transformations.translation_from_matrix(transformed_position)
                    
                    # Write to CSV
                    self.csv_writer.writerow([info.class_id, info.id, info.bounding_box_min_x, info.bounding_box_min_y,
                                              info.bounding_box_max_x, info.bounding_box_max_y, info.confidence,
                                              pose_x, pose_y, pose_z])

                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                    rospy.logerr("TF transformation failed: %s", e) 4

    def shutdown(self):
        self.csv_file.close()

def object_detection_callback(data):
    print("Object detection callback triggered.")

if __name__ == '__main__':
    #print("Step 1 done")
    #rospy.init_node('detection_saver', anonymous=True)
    #print("Step 2 done")

    #rospy.Subscriber("/test", String, object_detection_callback)
    #rospy.Subscriber("/object_detector/detection_info", ObjectDetectionInfoArray, object_detection_callback)
    #print("Step 3 done")
    # node = DetectionSaver()
    # rospy.on_shutdown(node.shutdown)
    node = DetectionSaver()
    #rospy.on_shutdown(node.shutdown)
    rospy.spin()
    #print("Step 4 done")
