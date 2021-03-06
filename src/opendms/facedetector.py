#! python3

"""
All functions related to face detection
"""

import logging
from typing import List
import cv2
import dlib
import numpy as np
import pkg_resources


class FaceDetector:
    """
    Base class for face detector.
    """

    def run(self, image) -> List:
        """
        Find faces from given `image`.

        :param image: Input image
        :return: List of faces detected
        """


class HaarFaceDetector(FaceDetector):
    """
    This class is used to detect face based on Haar cascade classifier.
    """

    def __init__(self, cascade_path: str = None):
        """
        Constructs detector from given cascade classifier.

        :param cascade_path: cascade classifier path
        """
        logging.info("Create face detector based on Haar cascade classifier")

        if cascade_path is None:
            cascade_path = pkg_resources.resource_filename(
                __name__, "data/haarcascade_frontalface2.xml"
            )

        #: Haar cascade classifier
        self.__classifier = cv2.CascadeClassifier(cascade_path)

    def run(self, image) -> List:
        """
        Find faces from given `image`.

        :param image: Input image
        :return: List of faces detected
        """
        # Preprocess image
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Run classifier
        faces = self.__classifier.detectMultiScale(image_bw)

        # Normalize results
        faces_normalized = []
        for x, y, width, height in faces:
            faces_normalized.append(((x, y), (x + width, y + height)))
        return faces_normalized


class DLibFaceDetector(FaceDetector):
    """
    This class is used to detect face based on DLib classifier.
    """

    def __init__(self):
        """
        Constructs DLib detector.
        """
        logging.info("Create face detector based on DLib classifier")

        #: DLib classifier
        self.__classifier = dlib.get_frontal_face_detector()

    def run(self, image) -> List:
        """
        Find faces from given `image`.

        :param image: Input image
        :return: List of faces detected
        """
        # Preprocess image
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Run classifier
        faces = self.__classifier(image_bw, 1)

        # Normalize results
        faces_normalized = []
        for face in faces:
            faces_normalized.append(
                ((face.left(), face.top()), (face.right(), face.bottom()))
            )
        return faces_normalized


class CaffeFaceDetector(FaceDetector):
    """
    This class is used to detect face based on Caffe model classifier.
    """

    def __init__(
        self,
        dnn_proto_text: str = None,
        dnn_model: str = None,
        threshold: float = 0.5,
    ):
        """
        Constructs Caffe detector from given DNN.

        :param dnn_proto_text: Model architecture (i.e., the layers themselves)
        :param dnn_model: Weights for the actual layers
        :param threshold: Confidence threshold to apply on result
        """
        logging.info("Create face detector based on Caffe dnn model")

        if dnn_proto_text is None:
            dnn_proto_text = pkg_resources.resource_filename(
                __name__, "data/deploy.prototxt.txt"
            )
        if dnn_model is None:
            dnn_model = pkg_resources.resource_filename(
                __name__, "data/res10_300x300_ssd_iter_140000.caffemodel"
            )

        #: DNN classifier
        self.__classifier = cv2.dnn.readNetFromCaffe(dnn_proto_text, dnn_model)

        #: Confidence threshold to apply on result
        self.__threshold = threshold

    def run(self, image) -> List:
        """
        Find faces from given `image`.

        :param image: Input image
        :return: List of faces detected
        """
        # Preprocess image
        image_height, image_width = image.shape[:2]
        image_resized = cv2.resize(image, (300, 300))

        # Run classifier
        self.__classifier.setInput(
            cv2.dnn.blobFromImage(
                image_resized, 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
        )
        faces = self.__classifier.forward()

        # Normalize results
        faces_normalized = []
        for i in range(faces.shape[2]):
            if faces[0, 0, i, 2] > self.__threshold:
                box = faces[0, 0, i, 3:7] * np.array(
                    [image_width, image_height, image_width, image_height]
                )
                (x, y, x1, y1) = box.astype("int")
                faces_normalized.append(((x, y), (x1, y1)))
        return faces_normalized
