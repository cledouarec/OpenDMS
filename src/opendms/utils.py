#! python3

"""
Utilities and helper functions
"""

from typing import List
import cv2
from .facedetector import FaceDetector
from .facefilter import FaceFilter
from .facelandmarksdetector import FaceLandmarksDetector


def draw_faces_boxes(image, faces: List) -> None:
    """
    Draw rectangle for each `faces` detected on the input `image`.

    :param image: Input image
    :param faces: List of faces detected to draw
    """
    for point1, point2 in faces:
        cv2.rectangle(image, point1, point2, (0, 0, 255), 2)


def draw_faces_landmarks(image, landmarks: List) -> None:
    """
    Draw points for each `landmarks` detected on the input `image`.

    :param image: Input image
    :param landmarks: List of faces landmarks detected to draw
    """
    for landmark in landmarks:
        for n in landmark:
            cv2.circle(image, n, 1, (255, 0, 0))


def detect_from_video(
    video_stream: cv2.VideoCapture,
    detector: FaceDetector,
    face_filters: List[FaceFilter] = None,
    draw_boxes: bool = False,
    landmarks_detector: FaceLandmarksDetector = None,
    draw_landmarks: bool = False,
) -> None:
    """
    Run given face `detector` on `video_stream`.
    The stream can be stopped by pressing q key.
    Pipeline :
    Video stream
    -> Detector
    -> Filters
    -> <Draw boxes>
    -> Landmarks_detector
    -> <draw landmarks>

    :param video_stream: Video stream input
    :param detector: Face detector
    :param face_filters: List of filter to apply on face detector results.
    :param draw_boxes: Draw boxes for each faces detected
    :param landmarks_detector: Face landmarks detector
    :param draw_landmarks: Draw landmarks for each faces detected
    """
    if face_filters is None:
        face_filters = []

    while True:
        ret, frame = video_stream.read()
        if not ret:
            break

        faces_detected = detector.run(frame)
        for face_filter in face_filters:
            # Avoid useless computation and keep at least 1 face
            if len(faces_detected) <= 1:
                break
            faces_detected = face_filter.run(frame, faces_detected)

        if draw_boxes:
            draw_faces_boxes(frame, faces_detected)

        if landmarks_detector is not None:
            faces_landmarks = landmarks_detector.run(frame, faces_detected)
            if draw_landmarks:
                draw_faces_landmarks(frame, faces_landmarks)

        cv2.imshow("img", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything is done, release the capture
    video_stream.release()
    cv2.destroyAllWindows()
