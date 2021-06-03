#! python3

"""
Main entry point to start Open DMS.
"""

import argparse
import logging
import sys
import cv2
from .facedetector import CaffeFaceDetector as Detector
from .facefilter import FaceBySize, FaceByRatio, FaceNearestCenter
from .facelandmarksdetector import (
    DLibFaceLandmarksDetector as LandmarksDetector,
)
from .utils import detect_from_video


def main() -> None:
    """
    Entry point of Open DMS script.
    """
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose mode",
        dest="verbose",
    )
    args = parser.parse_args()

    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # Create logger
    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=log_level,
    )

    detect_from_video(
        video_stream=cv2.VideoCapture(0, cv2.CAP_DSHOW),
        detector=Detector(),
        face_filters=[FaceBySize(), FaceByRatio(), FaceNearestCenter()],
        draw_boxes=True,
        landmarks_detector=LandmarksDetector(),
        draw_landmarks=True,
    )
