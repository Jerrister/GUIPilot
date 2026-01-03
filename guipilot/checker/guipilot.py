from __future__ import annotations

import re
import typing
from difflib import SequenceMatcher

import cv2
import numpy as np

from guipilot.entities import Inconsistency, WidgetType

from .checker import ScreenChecker

if typing.TYPE_CHECKING:
    from guipilot.entities import Widget


class GUIPilot(ScreenChecker):
    def check_widget_pair(
        self, w1: Widget, w2: Widget, wi1: np.ndarray, wi2: np.ndarray
    ) -> list[tuple]:
        
        def check_bbox_consistency(w1: Widget, w2: Widget) -> bool:
            """Check if both widgets have similar position, size, and shape on the screen"""
            return super().check_bbox_consistency(w1, w2)

        def check_text_consistency(w1: Widget, w2: Widget) -> bool:
            """Check if the text on both widgets are similar"""
            return super().check_text_consistency(w1, w2)
        
        def check_color_consistency(wi1: np.ndarray, wi2: np.ndarray) -> bool:
            """Check if the color distribution on both widgets are similar"""
            # normalized 3D color histogram, 8 bins per channel
            hist1 = cv2.calcHist(
                [wi1], [0, 1, 2], None, [8, 8, 8], [0, 250, 0, 250, 0, 250]
            )
            hist1 = cv2.normalize(hist1, hist1).flatten()

            hist2 = cv2.calcHist(
                [wi2], [0, 1, 2], None, [8, 8, 8], [0, 250, 0, 250, 0, 250]
            )
            hist2 = cv2.normalize(hist2, hist2).flatten()

            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_KL_DIV)
            return score < 8

        diff = set()
        if not check_bbox_consistency(w1, w2):
            diff.add(Inconsistency.BBOX)
        if not check_text_consistency(w1, w2):
            diff.add(Inconsistency.TEXT)
        if Inconsistency.TEXT not in diff:
            if not check_color_consistency(wi1, wi2):
                diff.add(Inconsistency.COLOR)

        return list(diff)
