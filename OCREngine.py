import easyocr
import cv2
import numpy as np

# Global reader instance (initialized once)
_reader = None

def get_ocr_engine():
    """Initialize and return the shared OCR reader instance"""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(
            lang_list=['en'],
            gpu=True,
            model_storage_directory=None,
            download_enabled=True
        )
    return _reader

def extract_text(image_input, as_string=True):
    """
    Extract text from input image (file path or numpy array)
    
    Args:
        image_input: str (file path) or np.ndarray (image array)
        as_string: Return as single string (True) or list of lines (False)
        
    Returns:
        Extracted text (str or list)
    """
    reader = get_ocr_engine()
    
    # Handle file path input
    if isinstance(image_input, str):
        text_lines = reader.readtext(image_input, detail=0)
    
    # Handle numpy array input
    elif isinstance(image_input, np.ndarray):
        # Convert BGR to RGB if needed
        if image_input.ndim == 3 and image_input.shape[2] == 3:
            # Check if it's likely BGR (OpenCV format)
            if image_input[0,0,0] > image_input[0,0,2]:  # Simple BGR detection
                image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        text_lines = reader.readtext(image_input, detail=0)
    
    else:
        raise TypeError("Input must be file path (str) or image array (np.ndarray)")
    
    return '\n'.join(text_lines) if as_string else text_lines