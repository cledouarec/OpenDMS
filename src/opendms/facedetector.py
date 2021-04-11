#! python3

"""
All functions related to face detection
"""

import logging
from typing import List, Union
import cv2
import dlib
import numpy as np
import pkg_resources


class HaarFaceDetector:
    """
    This class is used to detect face based on Haar cascade classifier.
    """

    def __init__(self, cascade_path: str = None):
        """
        Constructs detector from given cascade classifier.
        """
        logging.info("Create face detector based on Haar cascade classifier")

        if cascade_path is None:
            cascade_path = pkg_resources.resource_stream(
                __name__, "data/haarcascade_frontalface2.xml"
            ).read()

        #: Haar cascade classifier
        self.__classifier = cv2.CascadeClassifier(cascade_path)

    def find_faces(self, image) -> List:
        """
        Find faces from given image
        :return: List of faces boxes detected
        """
        # Preprocess image
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Run classifier
        results = self.__classifier.detectMultiScale(image_bw)

        # Normalize results
        faces = []
        for x, y, w, h in results:
            faces.append(((x, y), (x + w, y + h)))
        return faces


class DLibFaceDetector:
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

    def find_faces(self, image) -> List:
        """
        Find faces from given image
        :return: List of faces boxes detected
        """
        # Preprocess image
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Run classifier
        results = self.__classifier(image_bw, 1)

        # Normalize results
        faces = []
        for result in results:
            faces.append(
                (
                    (result.left(), result.top()),
                    (result.right(), result.bottom()),
                )
            )
        return faces


class CaffeFaceDetector:
    """
    This class is used to detect face based on Caffe model classifier.
    """

    def __init__(self, dnn_proto_text: str = None, dnn_model: str = None):
        """
        Constructs detector from given cascade classifier.
        """
        logging.info("Create face detector based on Caffe dnn model")

        if dnn_proto_text is None:
            dnn_proto_text = pkg_resources.resource_stream(
                __name__, "data/deploy.prototxt.txt"
            ).read()
        if dnn_model is None:
            dnn_model = pkg_resources.resource_stream(
                __name__, "data/res10_300x300_ssd_iter_140000.caffemodel"
            ).read()

        #: DNN classifier
        self.__classifier = cv2.dnn.readNetFromCaffe(dnn_proto_text, dnn_model)

    def find_faces(self, image, threshold: float = 0.5):
        """
        Find faces from given image
        :return: List of faces boxes detected
        """
        # Preprocess image
        h, w = image.shape[:2]
        image_resized = cv2.resize(image, (300, 300))

        # Run classifier
        self.__classifier.setInput(
            cv2.dnn.blobFromImage(
                image_resized, 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
        )
        results = self.__classifier.forward()

        # Normalize results
        faces = []
        for i in range(results.shape[2]):
            if results[0, 0, i, 2] > threshold:
                box = results[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                faces.append(((x, y), (x1, y1)))
        return faces


def draw_faces_boxes(image, faces: List) -> None:
    """
    Draw rectangle for each `faces` detected on the input `image`.

    :param image: Input image
    :param faces: List of faces detected to draw
    """
    for point1, point2 in faces:
        cv2.rectangle(image, point1, point2, (0, 0, 255), 2)


def detect_from_video(
    video_stream: cv2.VideoCapture,
    detector: Union[HaarFaceDetector, DLibFaceDetector, CaffeFaceDetector],
    draw_boxes: bool = False,
):
    """
    Run given face `detector` on `video_stream`.
    The stream can be stopped by pressing q key.

    :param video_stream: Video stream input
    :param detector: Face detector
    :param draw_boxes: Draw boxes for each faces detected
    """
    while True:
        ret, frame = video_stream.read()
        if ret:
            faces_detected = detector.find_faces(frame)
            if draw_boxes:
                draw_faces_boxes(frame, faces_detected)
            cv2.imshow("img", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # When everything is done, release the capture
    video_stream.release()
    cv2.destroyAllWindows()
