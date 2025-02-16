# ---------------------------  NOT USED YET ---------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class CaptchaProcessor:
    def __init__(self, target_size: Tuple[int, int] = (102, 40)):
        self.target_size = target_size
        
    def preprocess_embossed(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess embossed-style CAPTCHAs (like blue-colored ones).
        """
        # Extract blue channel for better contrast
        blue_channel = image[:, :, 0] if len(image.shape) == 3 else image
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blue_channel)
        
        # Apply Sobel edge detection
        sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize and threshold
        edges = np.uint8(255 * edges / np.max(edges))
        _, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary

class CaptchaSegmenter:
    def __init__(self, min_region_width: int = 10):
        self.min_region_width = min_region_width

    def _border_trace(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Step 1: Border tracing to detect character boundaries.
        """
        # Convert to binary
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours (character boundaries)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Bounding boxes for each detected character
        regions = [cv2.boundingRect(cnt) for cnt in contours if cv2.boundingRect(cnt)[2] > self.min_region_width]

        # Sort by x-coordinate (left to right)
        regions = sorted(regions, key=lambda x: x[0])

        return regions

    def _enlarge_regions(self, image: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Step 2: Region enlargement using interval-based segmentation.
        """
        img_width = image.shape[1]
        interval_size = img_width // len(regions) if regions else 1  # Prevent division by zero
        
        enlarged_regions = []
        for x, y, w, h in regions:
            new_x = max(0, x - 5)  # Expand left
            new_w = min(img_width - new_x, w + 10)  # Expand width but stay inside image
            enlarged_regions.append((new_x, y, new_w, h))

        return enlarged_regions
    
    def _extract_characters(self, image: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """
        Extract segmented characters from image.
        """
        characters = []
        for x, y, w, h in regions:
            char_img = image[y:y+h, x:x+w]
            characters.append(char_img)
        return characters
    
    def _draw_regions(self, image: np.ndarray, regions: List[Tuple[int, int, int, int]], color: Tuple[int, int, int]) -> np.ndarray:
        """
        Draw bounding boxes for visualization.
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        for x, y, w, h in regions:
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        return image

    def segment_with_visualization(self, image: np.ndarray) -> Tuple[List[np.ndarray], dict]:
        """
        Segment characters with visualization of each step.
        """
        vis_images = {}

        # Step 1: Border tracing
        initial_regions = self._border_trace(image)
        vis_images['border_tracing'] = self._draw_regions(image.copy(), initial_regions, (0, 0, 255))

        # Step 2: Region enlargement
        final_regions = self._enlarge_regions(image, initial_regions)
        vis_images['enlarged_regions'] = self._draw_regions(image.copy(), final_regions, (0, 255, 0))

        # Extract characters
        char_images = self._extract_characters(image, final_regions)

        return char_images, vis_images

def visualize_process(image: np.ndarray, preprocessed: np.ndarray, vis_images: dict):
    """
    Visualize segmentation steps.
    """
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(231)
    plt.title('Original')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Preprocessed image
    plt.subplot(232)
    plt.title('Preprocessed')
    plt.imshow(preprocessed, cmap='gray')
    plt.axis('off')
    
    # Border tracing
    plt.subplot(233)
    plt.title('Border Tracing')
    plt.imshow(cv2.cvtColor(vis_images['border_tracing'], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Enlarged regions
    plt.subplot(234)
    plt.title('Enlarged Regions')
    plt.imshow(cv2.cvtColor(vis_images['enlarged_regions'], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Initialize processor and segmenter
    processor = CaptchaProcessor()
    segmenter = CaptchaSegmenter()
    
    # Read image
    image = cv2.imread('C:/Users/MC/Desktop/PFE S5/Code/data/abacus_label_imgs/0b75a677690bb65d8f6d0a70507e8a1e_text_image.png')

    # Preprocess the image
    preprocessed = processor.preprocess_embossed(image)
    
    # Segment with visualization
    char_images, vis_images = segmenter.segment_with_visualization(preprocessed)
    
    # Visualize process
    visualize_process(image, preprocessed, vis_images)
    
    # Display segmented characters
    plt.figure(figsize=(10, 2))
    for i, char_img in enumerate(char_images):
        plt.subplot(1, len(char_images), i+1)
        plt.imshow(char_img, cmap='gray')
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
