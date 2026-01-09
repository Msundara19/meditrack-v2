"""
Computer Vision Service for Wound Analysis
Implements real wound segmentation and metric extraction
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

from app.services.wound_classifier import WoundClassifier, WoundFeatures

logger = logging.getLogger(__name__)


@dataclass
class WoundMetrics:
    """Data class to hold wound analysis results"""
    wound_area_pixels: int
    wound_area_cm2: float
    redness_index: float
    edge_sharpness: float
    healing_score: float
    mask: np.ndarray
    annotated_image: np.ndarray
    contours: list
    wound_type: str = "unknown"
    wound_features: Optional[WoundFeatures] = None


class WoundAnalyzer:
    """
    Advanced wound analysis using computer vision
    
    Pipeline:
    1. Preprocessing: Denoise, resize, normalize
    2. Segmentation: Color-based detection + Otsu thresholding
    3. Classification: Multi-factor wound type detection
    4. Post-processing: Morphological operations, contour selection
    5. Metric extraction: Area, redness, edge quality, healing score
    6. Visualization: Annotated images with overlays
    """
    
    def __init__(self, calibration_factor: float = 0.1):
        """
        Initialize wound analyzer
        
        Args:
            calibration_factor: Conversion from pixels to cm (default: 0.1 = 10 pixels = 1cm)
        """
        self.calibration_factor = calibration_factor
        self.classifier = WoundClassifier(calibration_factor)
        logger.info(f"WoundAnalyzer initialized with calibration factor: {calibration_factor}")
    
    def analyze_wound(self, image_path: str) -> WoundMetrics:
        """
        Complete wound analysis pipeline with classification
        
        Args:
            image_path: Path to wound image
            
        Returns:
            WoundMetrics object with all analysis results including wound type
            
        Raises:
            ValueError: If image cannot be loaded or processed
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        logger.info(f"Analyzing image: {image_path} (shape: {image.shape})")
        
        # Preprocessing
        preprocessed = self._preprocess(image)
        
        # Segmentation
        mask = self._segment_wound(preprocessed)
        
        # Classify wound type and extract enhanced features
        wound_features = self.classifier.classify_and_extract_features(mask, image)
        
        # Find contours for visualization
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate basic metrics
        area_pixels = np.sum(mask > 0)
        area_cm2 = area_pixels * (self.calibration_factor ** 2)
        redness_index = self._calculate_redness(image, mask)
        edge_sharpness = self._calculate_edge_sharpness(mask)
        healing_score = self._calculate_healing_score(area_cm2, redness_index, edge_sharpness)
        
        # Create annotated visualization
        annotated = self._create_annotated_image(image, mask, contours, wound_features)
        
        logger.info(f"Analysis complete: type={wound_features.wound_type}, "
                   f"area={area_cm2:.2f}cm², redness={redness_index:.3f}, "
                   f"edge={edge_sharpness:.3f}, score={healing_score:.1f}")
        
        return WoundMetrics(
            wound_area_pixels=int(area_pixels),
            wound_area_cm2=round(area_cm2, 2),
            redness_index=round(redness_index, 3),
            edge_sharpness=round(edge_sharpness, 3),
            healing_score=round(healing_score, 2),
            mask=mask,
            annotated_image=annotated,
            contours=contours,
            wound_type=wound_features.wound_type,
            wound_features=wound_features
        )
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image: resize, denoise, enhance
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed image
        """
        # Resize if too large (max dimension 1024px)
        max_dim = 1024
        h, w = image.shape[:2]
        
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")
        
        # Denoise using Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        return denoised
    
    def _segment_wound(self, image: np.ndarray) -> np.ndarray:
        """
        Segment wound region using multiple techniques:
        1. HSV color space for red detection
        2. LAB color space for redness in a* channel
        3. Otsu's thresholding
        4. Morphological operations for cleanup
        
        Args:
            image: Preprocessed BGR image
            
        Returns:
            Binary mask (255 = wound, 0 = background)
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Method 1: HSV red hue detection
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_hsv = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Method 2: LAB color space (a* channel detects red/green)
        a_channel = lab[:, :, 1]
        
        # Apply Otsu's thresholding on a* channel
        _, mask_lab = cv2.threshold(
            a_channel,
            0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Combine both methods
        combined_mask = cv2.bitwise_or(mask_hsv, mask_lab)
        
        # Morphological operations to clean up
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)
        
        # Find largest contour (assume it's the wound)
        contours, _ = cv2.findContours(
            mask_opened,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            logger.warning("No contours found, returning empty mask")
            return np.zeros_like(mask_opened)
        
        # Select largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create final mask with only the largest contour
        mask_final = np.zeros_like(mask_opened)
        cv2.drawContours(mask_final, [largest_contour], -1, 255, -1)
        
        # Additional smoothing of boundary
        mask_final = cv2.GaussianBlur(mask_final, (5, 5), 0)
        _, mask_final = cv2.threshold(mask_final, 127, 255, cv2.THRESH_BINARY)
        
        return mask_final
    
    def _calculate_redness(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate redness index in wound region
        
        Redness metric: (R - G) / (R + G + ε)
        Higher values indicate more inflammation
        
        Args:
            image: Original BGR image
            mask: Binary wound mask
            
        Returns:
            Redness index between 0 and 1
        """
        if np.sum(mask) == 0:
            return 0.0
        
        # Extract wound region
        mask_bool = mask > 0
        
        # Split channels (OpenCV uses BGR)
        b, g, r = cv2.split(image)
        
        # Calculate mean R and G in wound region
        r_mean = np.mean(r[mask_bool])
        g_mean = np.mean(g[mask_bool])
        
        # Redness index with epsilon to avoid division by zero
        epsilon = 1e-6
        redness = (r_mean - g_mean) / (r_mean + g_mean + epsilon)
        
        # Normalize to [0, 1] range
        redness_normalized = max(0.0, min(1.0, (redness + 1) / 2))
        
        return redness_normalized
    
    def _calculate_edge_sharpness(self, mask: np.ndarray) -> float:
        """
        Calculate wound boundary sharpness using Canny edge detection
        
        Sharp edges indicate well-defined wound boundaries
        
        Args:
            mask: Binary wound mask
            
        Returns:
            Edge sharpness score between 0 and 1
        """
        if np.sum(mask) == 0:
            return 0.0
        
        # Detect edges in mask
        edges = cv2.Canny(mask, 50, 150)
        
        # Calculate edge density relative to mask area
        edge_pixels = np.sum(edges > 0)
        mask_pixels = np.sum(mask > 0)
        
        if mask_pixels == 0:
            return 0.0
        
        # Edge density (perimeter relative to area)
        edge_density = edge_pixels / mask_pixels
        
        # Scale to [0, 1] range (cap at 0.1 edge density = perfect score)
        sharpness = min(1.0, edge_density * 10)
        
        return sharpness
    
    def _calculate_healing_score(
        self,
        area_cm2: float,
        redness: float,
        edge_sharpness: float
    ) -> float:
        """
        Calculate composite healing score (0-100)
        
        Higher score = better healing progress
        
        Factors:
        - Smaller area = better (wound closing)
        - Lower redness = better (less inflammation)
        - Higher edge sharpness = better (well-defined boundary)
        
        Args:
            area_cm2: Wound area in cm²
            redness: Redness index (0-1)
            edge_sharpness: Edge sharpness (0-1)
            
        Returns:
            Healing score (0-100)
        """
        # Normalize area (assume max wound size is 50 cm²)
        max_area = 50.0
        area_score = max(0.0, 1.0 - (area_cm2 / max_area))
        
        # Lower redness is better
        redness_score = 1.0 - redness
        
        # Higher edge sharpness is better (already 0-1)
        edge_score = edge_sharpness
        
        # Weighted combination
        composite_score = (
            area_score * 0.4 +
            redness_score * 0.4 +
            edge_score * 0.2
        )
        
        # Convert to 0-100 scale
        healing_score = composite_score * 100
        
        return max(0.0, min(100.0, healing_score))
    
    def _create_annotated_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        contours: list,
        wound_features: Optional[WoundFeatures] = None
    ) -> np.ndarray:
        """
        Create annotated visualization with wound overlay and measurements
        
        Args:
            image: Original BGR image
            mask: Binary wound mask
            contours: Detected contours
            wound_features: Optional wound classification features
            
        Returns:
            Annotated image with overlays
        """
        # Create a copy for annotation
        annotated = image.copy()
        
        # Create colored overlay (semi-transparent red)
        overlay = image.copy()
        overlay[mask > 0] = [0, 0, 255]  # Red in BGR
        
        # Blend original with overlay
        alpha = 0.3  # Transparency
        annotated = cv2.addWeighted(annotated, 1 - alpha, overlay, alpha, 0)
        
        # Draw contour boundary (green)
        if contours:
            cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)
        
        # Add text with metrics
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        
        # Show wound type if available
        if wound_features:
            # Wound type label
            wound_type_display = wound_features.wound_type.replace('_', ' ').title()
            text = f"Type: {wound_type_display}"
            cv2.putText(annotated, text, (10, y_offset), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, text, (10, y_offset), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            y_offset += 30
            
            # Show appropriate measurement based on wound type
            if wound_features.measurement_type == "linear":
                # For linear wounds (incisions, lacerations)
                text = f"Length: {wound_features.length_cm:.1f}cm x Width: {wound_features.width_cm:.1f}cm"
            else:
                # For area wounds (burns, ulcers)
                text = f"Area: {wound_features.area_cm2:.2f} cm²"
            
            cv2.putText(annotated, text, (10, y_offset), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, text, (10, y_offset), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            # Fallback to basic area display
            area_cm2 = np.sum(mask > 0) * (self.calibration_factor ** 2)
            text = f"Area: {area_cm2:.2f} cm²"
            cv2.putText(annotated, text, (10, y_offset), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, text, (10, y_offset), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        
        return annotated
    
    def save_annotated_image(self, annotated_image: np.ndarray, output_path: str) -> None:
        """
        Save annotated image to file
        
        Args:
            annotated_image: Annotated image array
            output_path: Path to save image
        """
        success = cv2.imwrite(output_path, annotated_image)
        if success:
            logger.info(f"Annotated image saved to {output_path}")
        else:
            logger.error(f"Failed to save annotated image to {output_path}")
            raise IOError(f"Could not save image to {output_path}")


# Utility functions

def validate_image(image_path: str) -> bool:
    """
    Validate if image can be loaded and processed
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False
        
        h, w = image.shape[:2]
        if h < 100 or w < 100:
            logger.warning(f"Image too small: {w}x{h}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return False