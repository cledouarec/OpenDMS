#! python3

"""
All functions related to face landmarks detection
"""

import logging
from typing import List
import cv2
import dlib
import numpy as np
import pkg_resources


class FaceLandmarksDetector:
    """
    Base class for face landmarks detector.
    """

    def run(self, image, faces: List) -> List:
        """
        Find landmarks on given `faces` detected on `image`.

        :param image: Input image
        :param faces: List of faces
        :return: List of faces detected
        """


class DLibFaceLandmarksDetector(FaceLandmarksDetector):
    """
    This class is used to find landmarks on face based on DLib classifier.
    """

    def __init__(self, dnn_model: str = None):
        """
        Constructs DLib landmarks detector.
        """
        logging.info("Create face landmarks detector based on DLib classifier")

        if dnn_model is None:
            dnn_model = pkg_resources.resource_filename(
                __name__, "data/shape_predictor_68_face_landmarks.dat"
            )

        #: DLib landmarks classifier
        self.__detector = dlib.shape_predictor(dnn_model)

    def run(self, image, faces: List) -> List:
        """
        Find landmarks on given `faces` detected on `image`.

        :param image: Input image
        :param faces: List of faces
        :return: List of faces detected
        """
        # Preprocess image
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        all_faces_landmarks_list = []
        for face in faces:
            dlib_rect = dlib.rectangle(
                left=face[0][0],
                top=face[0][1],
                right=face[1][0],
                bottom=face[1][1],
            )

            # Run classifier
            landmarks = self.__detector(image_bw, dlib_rect)
            landmarks_list = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_list.append((x, y))
            all_faces_landmarks_list.append(landmarks_list)

        return all_faces_landmarks_list


class LBFFaceLandmarksDetector(FaceLandmarksDetector):
    """
    This class is used to find landmarks on face based on LBF classifier.
    """

    def __init__(self, lbf_model: str = None):
        """
        Constructs LBF landmarks detector.
        """
        logging.info("Create face landmarks detector based on LBF classifier")

        if lbf_model is None:
            lbf_model = pkg_resources.resource_filename(
                __name__, "data/lbfmodel.yaml"
            )

        #: LBF landmarks classifier
        self.__detector = cv2.face.createFacemarkLBF()
        self.__detector.loadModel(lbf_model)

    def run(self, image, faces: List) -> List:
        """
        Find landmarks on given `faces` detected on `image`.

        :param image: Input image
        :param faces: List of faces
        :return: List of faces detected
        """
        # Preprocess image
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces_converted = []
        for face in faces:
            faces_converted.append(
                [
                    face[0][0],  # - 100,
                    face[0][1],
                    face[1][0] - face[0][0],  # + 200,
                    face[1][1] - face[0][1],
                ]
            )

        # Run classifier
        _, landmarks = self.__detector.fit(
            image_bw, np.array([faces_converted])
        )

        # Normalize results
        for landmark in landmarks:
            for x, y in landmark[0]:
                # display landmarks on "frame/image,"
                # with blue colour in BGR and thickness 1
                cv2.circle(image, (int(x), int(y)), 1, (255, 0, 0), 1)

        return []


class KazemiFaceLandmarksDetector(FaceLandmarksDetector):
    """
    This class is used to find landmarks on face based on Kazemi classifier.
    """

    def __init__(self, lbf_model: str = None):
        """
        Constructs Kazemi landmarks detector.
        """
        logging.info(
            "Create face landmarks detector based on Kazemi classifier"
        )

        if lbf_model is None:
            lbf_model = pkg_resources.resource_filename(
                __name__, "data/lbfmodel.yaml"
            )

        #: Kazemi landmarks classifier
        self.__detector = cv2.face.createFacemarkKazemi()

    def run(self, image, faces: List) -> List:
        """
        Find landmarks on given `faces` detected on `image`.

        :param image: Input image
        :param faces: List of faces
        :return: List of faces detected
        """
        # Preprocess image
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces_converted = []
        for face in faces:
            faces_converted.append(
                [
                    face[0][0] - 100,
                    face[0][1],
                    face[1][0] - face[0][0] + 200,
                    face[1][1] - face[0][1],
                ]
            )

        # Run classifier
        _, landmarks = self.__detector.fit(
            image_bw, np.array([faces_converted])
        )

        # Normalize results
        for landmark in landmarks:
            for x, y in landmark[0]:
                # display landmarks on "frame/image,"
                # with blue colour in BGR and thickness 1
                cv2.circle(image, (x, y), 1, (255, 0, 0), 1)

        return []
