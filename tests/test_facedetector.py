#! python3

"""
Unit tests for face detector.
"""

from opendms.utils import draw_faces_boxes


def test_draw_faces_boxes_with_empty_list_do_not_raise_error() -> None:
    """
    Sample test.
    """
    try:
        draw_faces_boxes(None, [])
    except Exception as error:
        assert False, f"'draw_faces_boxes' raised an exception {error}"
