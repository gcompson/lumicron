import cv2
import numpy as np

class ImageProcessor:
    @staticmethod
    def stretch_contrast(image, clip_limit=2.0, grid_size=(8, 8)):
        """Surgical CLAHE: Lower clip limit to avoid blowing out clouds."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # A tighter clipLimit (2.0) is better for daylight sky
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        l = clahe.apply(l)
        
        limg = cv2.merge((l, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    @staticmethod
    def apply_binary_isolation(image, threshold_val=200):
        """Isolates high-intensity transients (the streak) from the sky."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
        
        # Morphological opening to remove 'salt and pepper' sensor noise
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return cv2.bitwise_and(image, image, mask=mask)

    @staticmethod
    def apply_false_color(image):
        """Retained but improved: Use COLORMAP_VIRIDIS for better sky gradient."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        # VIRIDIS is often more readable than JET for UAP transients
        return cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
