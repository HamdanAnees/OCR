import cv2
import numpy as np
from skimage.transform import radon

class ImagePreprocessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.processed = None
    
    @staticmethod
    def estimate_skew_radon(gray: np.ndarray, angle_range: float = 15.0) -> float:
        """
        Improved Radon-based skew estimation using raw grayscale values without downsampling
        """
        # Compute Radon transform projections
        thetas = np.linspace(-angle_range, angle_range, 100)
        sinogram = radon(gray, theta=thetas, circle=False)
    
        # Find angle with maximum variance in projection
        variances = np.var(sinogram, axis=0)
        best_angle = thetas[np.argmax(variances)]
        return best_angle
    
    @staticmethod
    def deskew(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
        """Deskew is one of the hardest things to do in image processing this is a simple algorithum which deskews the image, but this can be further scaled"""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
    
        # Detect skew angle (limited to reasonable range)
        angle = ImagePreprocessor.estimate_skew_radon(gray)
    
        # Skip rotation if angle is negligible
        if abs(angle) < 0.1:
            return image.copy()
    
        # Limit rotation to prevent over-rotation
        angle = np.clip(angle, -max_angle, max_angle)
    
        # Rotate image with proper border handling
        h, w = image.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
        # Calculate new bounding box
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
    
        # Adjust rotation matrix
        M[0, 2] += (nW - w) / 2
        M[1, 2] += (nH - h) / 2
    
        # Apply rotation with reflection border and cubic interpolation
        return cv2.warpAffine(image, M, (nW, nH),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)


    def denoise(self):
        pass

    def threshold(self):
        pass

    def morphological(self):
        pass
    def normalize_illumination(image, kernel_size: int = 101, clip_limit: float = 2.0) -> np.ndarray:
        """
        Normalize uneven illumination using background subtraction and CLAHE
        :param kernel_size: Size of Gaussian kernel for background estimation (should be odd)
        :param clip_limit: Contrast limit for CLAHE (higher values increase contrast)
        :return: Illumination-normalized image
        """
        # Ensure kernel size is odd
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        
        # Convert to grayscale for illumination estimation
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 1. Estimate background with large Gaussian blur
        background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        # 2. Subtract background and normalize
        normalized = gray.astype(np.float32) - background.astype(np.float32)
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
        
        # 3. Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized.astype(np.uint8))
        
        # For color images, merge enhanced illumination with original color channels
        if image.ndim == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            lab = cv2.merge([enhanced, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced
    
    @staticmethod
    def resize(image, target_dpi=500, physical_width_inches=None, physical_height_inches=None):
        """
        Resize image to maintain minimum DPI of 500 for a given physical size.
        Requires one physical dimension to calculate the other while preserving aspect ratio.
        """
        h, w = image.shape[:2]
    
        # Validate inputs
        if physical_width_inches is None and physical_height_inches is None:
            raise ValueError("Must provide width or height in inches")
    
        # Calculate target dimensions
        if physical_width_inches is not None:
            target_width = int(physical_width_inches * target_dpi)
            scale = target_width / w
            target_height = int(h * scale)
        else:
            target_height = int(physical_height_inches * target_dpi)
            scale = target_height / h
            target_width = int(w * scale)
    
        # Choose interpolation method
        if scale > 1:  # Upscaling
            return cv2.resize(image, (target_width, target_height), 
                         interpolation=cv2.INTER_LANCZOS4)
        else:  # Downscaling
            return cv2.resize(image, (target_width, target_height), 
                         interpolation=cv2.INTER_AREA)
        
    def save(self, output_path):
        pass

    def get_result(self):
        pass
