import cv2
import numpy as np
from bottle_detector import BottleDetector

scale_factor_x = 2.2
scale_factor_y = 2.5
padding = 5
redundancy_range = 10
threshold = 0.6

bottle_detector = BottleDetector(redundancy_range)

class DefectDetectionPipeline:
    def __init__(self, template_path='./data/template.jpeg', defect_template_path='./data/defect_bottle_template.png'):
        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        self.defect_template = cv2.imread(defect_template_path, cv2.IMREAD_GRAYSCALE)

    def detect_defects_live(self, frame):
        try:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            for template_type, template in [('normal', self.template), ('defect', self.defect_template)]:
                best_match = None
                best_max_val = -float('inf')

                for scale in np.linspace(0.5, 1.5, 20)[::-1]:
                    resized_template = cv2.resize(template, None, fx=scale, fy=scale)
                    resized_width, resized_height = resized_template.shape[::-1]

                    if resized_width > gray_image.shape[1] or resized_height > gray_image.shape[0]:
                        continue

                    result = cv2.matchTemplate(gray_image, resized_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

                    if max_val > best_max_val:
                        best_max_val = max_val
                        best_match = (max_loc, scale)

                if best_match:
                    pt, scale = best_match
                    color = (0, 255, 0) if template_type == 'normal' and best_max_val >= threshold else (0, 0, 255)
                    top_left = (pt[0] - padding, pt[1] - padding)
                    bottom_right = (pt[0] + padding + int(resized_width * scale_factor_x),
                                    pt[1] + padding + int(resized_height * scale_factor_y))
                    cv2.rectangle(frame, top_left, bottom_right, color, 2)
                    break

            return frame

        except Exception as e:
            print(f"[ERROR] {e}")
            return frame
