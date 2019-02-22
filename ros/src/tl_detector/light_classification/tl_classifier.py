from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import os
import rospy
import cv2
from timeit import default_timer as timer

CLASS_TRAFFIC_LIGHT = 10
MODEL_DIR = 'light_classification/models/'
IMG_DIR = 'light_classification/imgs/'
DEBUG_DIR = 'light_classification/debug/'

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.detector = MODEL_DIR+'faster_rcnn_inception_v2.pb'
        self.sess,_ = self.load_graph(self.detector)
        detection_graph = self.sess.graph

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # The classification of the object (integer id).
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # the very first decoding is slow: al inits are done
        # => do that in advance before real decoding
        for i in range(1):
            #test_image = cv2.imread('light_classification/img/left0144.jpg')
            test_image = cv2.imread(IMG_DIR + 'image11.jpg')

#            pred_image, is_red = self.detect_tl(test_image)
            image_np, box_coords, classes, scores = self.detect_tl(test_image)

            # Traditional traffic light classifier
            pred_image, is_red = self.classify_red_tl(image_np, box_coords, classes, scores)
            if is_red:
                print("Traditional classifier: RED")
            else:
                print("Traditional classifier: NOT RED")

            cv2.imwrite(IMG_DIR + 'pred_image.png', pred_image)

        self.num_image = 1

    def load_graph(self,graph_file,use_xla = False):
        config = tf.ConfigProto(log_device_placement = False)
        config.gpu_options.allow_growth = True
        session = tf.Session(config = config)

        with tf.Session(graph=tf.Graph(), config=config) as sess:
            gd = tf.GraphDef()
            with tf.gfile.Open(graph_file, 'rb') as f:
                data = f.read()
                gd.ParseFromString(data)
            tf.import_graph_def(gd, name='')
            ops = sess.graph.get_operations()
            n_ops = len(ops)
            print("number of operations = %d" % n_ops)
            return sess, ops

    def detect_tl(self,image):
        trt_image = np.copy(image)
        image_np = np.expand_dims(np.asarray(trt_image,dtype = np.uint8),0)

        # Actual detection.
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],feed_dict={self.image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.8

        # Filter traffic light boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes, keep_classes=[CLASS_TRAFFIC_LIGHT])

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        image_np = np.squeeze(image_np)
        width = image_np.shape[1]
        height = image_np.shape[0]
        box_coords = self.to_image_coords(boxes, height, width)

        return image_np, box_coords, classes, scores

    def filter_boxes(self,min_score,boxes,scores,classes,keep_classes = None):
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i]>=min_score:
                if ((keep_classes is None) or (int(classes[i]) in keep_classes)):
                    idxs.append(i)
        filter_boxes = boxes[idxs,...]
        filter_scores = scores[idxs,...]
        filter_classes = classes[idxs,...]
        return filter_boxes,filter_scores,filter_classes

    def to_image_coords(self,boxes,height,width):
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        return box_coords

    def classify_red_tl(self, image_np, boxes, classes, scores, thickness=4):
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])
            score = scores[i]
#            if class_id == CLASS_TRAFFIC_LIGHT:
            h = top - bot
            w = right - left
            if h <= 1.5 * w:
                continue # Truncated Traffic Ligth box
            cv2.rectangle(image_np,(left, top), (right, bot), (255, 0, 0), thickness) # BGR format for color
            tl_img = image_np[int(bot):int(top), int(left):int(right)]

            tl_img_simu = self.select_red_simu(tl_img) # SELECT RED
            tl_img_real = self.select_lighton_real(tl_img) # SELECT TL
            #tl_img_real_filter = self.select_white_real(tl_img) # SELECT TL with polarization filter
            tl_img = (tl_img_simu + tl_img_real) / 2

            gray_tl_img = cv2.cvtColor(tl_img, cv2.COLOR_RGB2GRAY)
            nrows, ncols = gray_tl_img.shape[0], gray_tl_img.shape[1]

            # compute center of mass of RED points
            mean_row = 0
            mean_col = 0
            npoints = 0
            for row in range(nrows):
                for col in range(ncols):
                    if (gray_tl_img[row, col] > 0):
                        mean_row += row
                        mean_col += col
                        npoints += 1
            if npoints > 0:
              mean_row = float(mean_row / npoints) / nrows
              mean_col = float(mean_col / npoints) / ncols
              #print(mean_row, mean_col, npoints)

              # if normalized center of mass of RED points
              # is in the upper part of detected Traffic Light Box
              # THEN it is a RED traffic light
              if npoints > 10 and mean_row < 0.33:
                  text = "RED LIGHT score=%.3f" % score
                  cv2.putText(image_np, text, (int(left), int(bot)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                  return image_np, True
        return image_np, False

    def select_red_simu(self,img):
        lower = np.array([ 0,   0, 200], dtype="uint8")
        upper = np.array([ 50, 50, 255], dtype="uint8")
        red_mask = cv2.inRange(img, lower, upper)
        return cv2.bitwise_and(img, img, mask = red_mask)

    def select_lighton_real(self,img):
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        lower = np.array([ 50,   150, 150], dtype="uint8")
        upper = np.array([ 100, 255, 255], dtype="uint8")
        tl_mask = cv2.inRange(hls_img, lower, upper)
        return cv2.bitwise_and(img, img, mask = tl_mask)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        # light detection
        start = timer()
        image_np, box_coords, classes, scores = self.detect_tl(image)
        end_inference = timer()

        # light color classification
        pred_image, is_red = self.classify_red_tl(image_np, box_coords, classes, scores)
        end = timer()


        time_inference = end_inference - start
        time_img_processing = (end - start) - time_inference
        print("time: inference {:.6f} post-processing {:.6f}".format(time_inference, time_img_processing))


        # fimage = DEBUG_DIR + 'image' + str(self.num_image) + '.png'
        # cv2.imwrite(fimage, pred_image)
        # self.num_image += 1

        if is_red:
            return TrafficLight.RED
        else:
            return TrafficLight.UNKNOWN
