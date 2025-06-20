from imagePreprocessor import ImagePreprocessor
from OCREngine import ocr
import cv2

def main():
    original = cv2.imread("09.jpg")
    resize = ImagePreprocessor.normalize_illumination(original, 8, 8)
    ocr(resize)
    return

if __name__ == "__main__":
    main()

