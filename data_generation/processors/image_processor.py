import cv2
import numpy as np
import os
from PIL import Image
from typing import Tuple

class ImageNotFoundError(Exception):
    pass

class ImageProcessor:
    def __init__(self):
        pass
    
    def load_image(self, image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise ImageNotFoundError(f"Image not found: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ImageNotFoundError(f"Could not read image: {image_path}")
        
        return img
    
    def load_image_pil(self, image_path: str) -> Image.Image:
        return Image.open(image_path).convert('RGB')
    
    def get_image_shape(self, image_path: str) -> Tuple[int, int]:
        img = self.load_image(image_path)
        h, w = img.shape[:2]
        return h, w
    
    def resize_image(self, img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        return cv2.resize(img, (size[1], size[0]))
    
    def normalize_image(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.float32) / 255.0
    
    def denormalize_image(self, img: np.ndarray) -> np.ndarray:
        return (img * 255.0).astype(np.uint8)
