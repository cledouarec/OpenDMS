#! python3

"""
All functions used to filter the result of face detector
"""

import logging
import math
from typing import List
import cv2


class FaceFilter:
    """
    Base class for face filters.
    """

    def run(self, image, faces: List) -> List:
        """
        Run the filter and return its results.

        :param image: Input image
        :param faces: List of faces to filter
        :return: List of faces filtered
        """


class FaceNearestCenter(FaceFilter):
    """
    Filter faces list to keep only the face nearest the center.
    """

    def __init__(self):
        """
        Initialize filter.
        """
        logging.info("Create filter face nearest center")

    def run(self, image, faces: List) -> List:
        """
        Run the filter to keep only the face nearest the center.
        This function returns a list to be applied like all filters.

        :param image: Input image
        :param faces: List of faces to filter
        :return: List of faces filtered
        """
        face_nearest = []
        h, w = image.shape[:2]
        image_center = (w / 2, h / 2)
        min_distance = math.inf

        for point1, point2 in faces:
            face_center = (
                (point2[0] + point1[0]) / 2,
                (point2[1] + point1[1]) / 2,
            )
            face_distance = cv2.norm(face_center, image_center, cv2.NORM_L2)
            if face_distance < min_distance:
                min_distance = face_distance
                face_nearest = [(point1, point2)]
        return face_nearest


class FaceBySize(FaceFilter):
    """
    Filter faces list to keep only the face in the defined size.
    """

    def __init__(self, min_size: float = 0.1, max_size: float = 1.0):
        """
        Initialize filter with min and max size ratio.
        The ratio in percent is calculated from the input image size.
        The value must be between 0 and 1.

        :param min_size: Minimal face size ratio in percent allowed.
        :param max_size: Maximal face size ratio in percent allowed.
        """
        logging.info("Create filter faces by size")

        if max_size <= min_size:
            raise ValueError("Max size cannot be lower or equal to min size.")

        #: Minimal face size ratio in percent allowed
        self.__min_size = min_size

        #: Maximal face size ratio in percent allowed
        self.__max_size = max_size

    def run(self, image, faces: List) -> List:
        """
        Run the filter to keep only faces between the size ratio.

        :param image: Input image
        :param faces: List of faces to filter
        :return: List of faces filtered
        """
        face_filtered = []
        image_height, image_width = image.shape[:2]
        min_image_height = image_height * self.__min_size
        max_image_height = image_height * self.__max_size
        min_image_width = image_width * self.__min_size
        max_image_width = image_width * self.__max_size

        for point1, point2 in faces:
            face_width = abs(point2[0] - point1[0])
            face_height = abs(point2[1] - point1[1])
            if (
                min_image_height > face_height > max_image_height
                and min_image_width > face_width > max_image_width
            ):
                face_filtered.append((point1, point2))
        return face_filtered


class FaceByRatio(FaceFilter):
    """
    Filter faces list to keep only the face in the defined ratio between height
    and width.
    """

    def __init__(self, min_ratio: float = 0.4, max_ratio: float = 1.4):
        """
        Initialize filter with min and max ratio.
        The ratio in percent is calculated from the width divided by the
        height.

        :param min_ratio: Minimal face ratio in percent allowed.
        :param max_ratio: Maximal face ratio in percent allowed.
        """
        logging.info("Create filter faces by ratio")

        if max_ratio <= min_ratio:
            raise ValueError(
                "Max ratio cannot be lower or equal to min ratio."
            )

        #: Minimal face ratio in percent allowed
        self.__min_ratio = min_ratio

        #: Maximal face ratio in percent allowed
        self.__max_ratio = max_ratio

    def run(self, image, faces: List) -> List:
        """
        Run the filter to keep only faces with good ratio.

        :param image: Input image
        :param faces: List of faces to filter
        :return: List of faces filtered
        """
        face_filtered = []

        for point1, point2 in faces:
            face_width = float(abs(point2[0] - point1[0]))
            face_height = float(abs(point2[1] - point1[1]))
            face_ratio = face_width / face_height
            if self.__max_ratio > face_ratio > self.__min_ratio:
                face_filtered.append((point1, point2))
        return face_filtered
