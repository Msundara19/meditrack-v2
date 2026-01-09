"""
Advanced wound classification system
Uses multiple features for robust wound type detection
"""
import cv2
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class WoundType(Enum):
    """Wound classification types"""
    SURGICAL_INCISION = "surgical_incision"
    LACERATION = "laceration"
    BURN = "burn"
    PRESSURE_ULCER = "pressure_ulcer"
    DIABETIC_ULCER = "diabetic_ulcer"
    ABRASION = "abrasion"
    PUNCTURE = "puncture"
    UNKNOWN = "unknown"


@dataclass
class WoundFeatures:
    """Extracted features for classification"""
    # Shape features
    aspect_ratio: float
    circularity: float
    solidity: float
    elongation: float
    
    # Size features
    area_cm2: float
    perimeter_cm: float
    
    # Geometric features
    has_straight_edges: bool
    edge_smoothness: float
    symmetry_score: float
    
    # Color/pattern features
    has_sutures: bool
    
    # Measurements
    length_cm: float
    width_cm: float
    
    # Wound type
    wound_type: str
    measurement_type: str  # "area" or "linear"


class WoundClassifier:
    """
    Intelligent wound classifier using multiple features
    """
    
    def __init__(self, calibration_factor: float = 0.1):
        self.calibration_factor = calibration_factor
        logger.info(f"WoundClassifier initialized with calibration={calibration_factor}")
    
    def classify_and_extract_features(
        self, 
        mask: np.ndarray,
        image: np.ndarray = None
    ) -> WoundFeatures:
        """
        Classify wound type and extract all features
        
        Args:
            mask: Binary wound mask
            image: Optional original image for color analysis
            
        Returns:
            WoundFeatures with classification and measurements
        """
        # Extract features
        features = self._extract_all_features(mask, image)
        
        # Classify
        wound_type = self._classify_from_features(features)
        
        # Determine measurement type
        if wound_type in [WoundType.SURGICAL_INCISION, WoundType.LACERATION]:
            measurement_type = "linear"
        else:
            measurement_type = "area"
        
        # Create result
        result = WoundFeatures(
            aspect_ratio=features['aspect_ratio'],
            circularity=features['circularity'],
            solidity=features['solidity'],
            elongation=features['elongation'],
            area_cm2=features['area_cm2'],
            perimeter_cm=features['perimeter_cm'],
            has_straight_edges=features['has_straight_edges'],
            edge_smoothness=features['edge_smoothness'],
            symmetry_score=features['symmetry_score'],
            has_sutures=features['has_sutures'],
            length_cm=features['length_cm'],
            width_cm=features['width_cm'],
            wound_type=wound_type.value,
            measurement_type=measurement_type
        )
        
        logger.info(f"Classified as: {wound_type.value} ({measurement_type} measurement)")
        logger.info(f"Features: aspect={features['aspect_ratio']:.2f}, "
                   f"circ={features['circularity']:.2f}, "
                   f"smooth={features['edge_smoothness']:.3f}, "
                   f"straight={features['has_straight_edges']}, "
                   f"sutures={features['has_sutures']}")
        
        return result
    
    def _extract_all_features(
        self, 
        mask: np.ndarray, 
        image: np.ndarray = None
    ) -> dict:
        """Extract comprehensive features from wound mask"""
        
        # Find contour
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return self._get_default_features()
        
        contour = max(contours, key=cv2.contourArea)
        
        # --- 1. ASPECT RATIO ---
        rect = cv2.minAreaRect(contour)
        ((cx, cy), (w, h), angle) = rect
        
        if w == 0 or h == 0:
            aspect_ratio = 1.0
            length_px = 0
            width_px = 0
        else:
            aspect_ratio = max(w, h) / min(w, h)
            length_px = max(w, h)
            width_px = min(w, h)
        
        # --- 2. CIRCULARITY ---
        area_px = cv2.contourArea(contour)
        perimeter_px = cv2.arcLength(contour, True)
        
        if perimeter_px > 0:
            circularity = 4 * np.pi * area_px / (perimeter_px ** 2)
            circularity = min(1.0, circularity)
        else:
            circularity = 0.0
        
        # --- 3. SOLIDITY ---
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area > 0:
            solidity = area_px / hull_area
        else:
            solidity = 0.0
        
        # --- 4. ELONGATION ---
        if area_px > 10:
            moments = cv2.moments(contour)
            if moments['m00'] > 0:
                mu20 = moments['mu20'] / moments['m00']
                mu02 = moments['mu02'] / moments['m00']
                mu11 = moments['mu11'] / moments['m00']
                
                term = np.sqrt((mu20 - mu02)**2 + 4*mu11**2)
                lambda1 = (mu20 + mu02 + term) / 2
                lambda2 = (mu20 + mu02 - term) / 2
                
                if lambda2 > 0:
                    elongation = np.sqrt(lambda1 / lambda2)
                else:
                    elongation = aspect_ratio
            else:
                elongation = aspect_ratio
        else:
            elongation = 1.0
        
        # --- 5. EDGE SMOOTHNESS ---
        edge_smoothness = self._calculate_edge_smoothness(contour)
        
        # --- 6. STRAIGHT EDGES ---
        has_straight_edges = self._detect_straight_edges(contour)
        
        # --- 7. SYMMETRY ---
        symmetry_score = self._calculate_symmetry(mask, rect)
        
        # --- 8. SUTURE DETECTION ---
        has_sutures = False
        if image is not None:
            has_sutures = self._detect_sutures(image, mask)
        
        # --- Convert to physical units ---
        area_cm2 = area_px * (self.calibration_factor ** 2)
        perimeter_cm = perimeter_px * self.calibration_factor
        length_cm = length_px * self.calibration_factor
        width_cm = width_px * self.calibration_factor
        
        return {
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'solidity': solidity,
            'elongation': elongation,
            'area_cm2': area_cm2,
            'perimeter_cm': perimeter_cm,
            'has_straight_edges': has_straight_edges,
            'edge_smoothness': edge_smoothness,
            'symmetry_score': symmetry_score,
            'has_sutures': has_sutures,
            'length_cm': length_cm,
            'width_cm': width_cm
        }
    
    def _calculate_edge_smoothness(self, contour: np.ndarray) -> float:
        """Calculate how smooth/regular the edges are"""
        if len(contour) < 5:
            return 0.0
        
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        smoothness = 1.0 - (len(approx) / max(len(contour), 1))
        return max(0.0, min(1.0, smoothness))
    
    def _detect_straight_edges(self, contour: np.ndarray) -> bool:
        """Detect if wound has predominantly straight edges"""
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 3:
            return False
        
        perimeter = cv2.arcLength(contour, True)
        straight_count = 0
        
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0]
            length = np.linalg.norm(p1 - p2)
            
            if length > 0.2 * perimeter:
                straight_count += 1
        
        return 1 <= straight_count <= 2
    
    def _calculate_symmetry(self, mask: np.ndarray, rect: Tuple) -> float:
        """Calculate symmetry score along major axis"""
        ((cx, cy), (w, h), angle) = rect
        
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(mask, rotation_matrix, mask.shape[::-1])
        
        center_x = int(cx)
        if center_x <= 0 or center_x >= mask.shape[1]:
            return 0.5
        
        left_half = rotated[:, :center_x]
        right_half = rotated[:, center_x:]
        
        right_flipped = cv2.flip(right_half, 1)
        
        min_width = min(left_half.shape[1], right_flipped.shape[1])
        if min_width == 0:
            return 0.5
            
        left_half = left_half[:, :min_width]
        right_flipped = right_flipped[:, :min_width]
        
        if left_half.size == 0:
            return 0.5
        
        overlap = np.sum(left_half == right_flipped) / left_half.size
        return overlap
    
    def _detect_sutures(self, image: np.ndarray, mask: np.ndarray) -> bool:
        """Detect presence of surgical sutures - EXPANDED RANGES"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect blue sutures - EXPANDED RANGE for better detection
        lower_blue = np.array([80, 30, 30])   # Expanded from [90, 50, 50]
        upper_blue = np.array([140, 255, 255])  # Expanded from [130, 255, 255]
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Also detect dark/black sutures
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 80])  # Expanded from 60
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Combine both
        suture_mask_combined = cv2.bitwise_or(blue_mask, dark_mask)
        
        suture_region = cv2.bitwise_and(suture_mask_combined, mask)
        suture_pixels = np.sum(suture_region > 0)
        wound_pixels = np.sum(mask > 0)
        
        if wound_pixels == 0:
            return False
        
        suture_ratio = suture_pixels / wound_pixels
        
        # VERY lenient threshold - 1% instead of 3%
        has_sutures = suture_ratio > 0.01
        
        logger.debug(f"Suture detection: {suture_pixels} pixels ({suture_ratio:.4f} ratio) - "
                    f"{'DETECTED' if has_sutures else 'not found'}")
        
        return has_sutures
    
    def _classify_from_features(self, features: dict) -> WoundType:
        """
        Multi-factor decision tree for wound classification
        VERY AGGRESSIVE surgical incision detection for real-world wounds
        """
        aspect = features['aspect_ratio']
        circ = features['circularity']
        solid = features['solidity']
        smooth = features['edge_smoothness']
        straight = features['has_straight_edges']
        sutures = features['has_sutures']
        area = features['area_cm2']
        
        # RULE 1: Surgical Incision - VERY AGGRESSIVE
        # If sutures detected, it's definitely surgical
        if sutures:
            logger.debug(f"→ Surgical incision (SUTURES DETECTED)")
            return WoundType.SURGICAL_INCISION
        
        # Aspect ratio 1.5+ with straight edges = likely incision
        if aspect >= 1.5 and straight:
            logger.debug(f"→ Surgical incision (aspect {aspect:.2f} + straight edges)")
            return WoundType.SURGICAL_INCISION
        
        # Low circularity (not round) + some elongation = likely incision
        if circ < 0.6 and aspect >= 1.3:
            logger.debug(f"→ Surgical incision (not circular: {circ:.2f}, aspect {aspect:.2f})")
            return WoundType.SURGICAL_INCISION
        
        # Very high smoothness = surgical (clean cut)
        if smooth > 0.9 and aspect >= 1.3:
            logger.debug(f"→ Surgical incision (very smooth: {smooth:.3f}, aspect {aspect:.2f})")
            return WoundType.SURGICAL_INCISION
        
        # High aspect ratio alone
        if aspect >= 2.5:
            logger.debug(f"→ Surgical incision (elongated: {aspect:.2f})")
            return WoundType.SURGICAL_INCISION
        
        # RULE 2: Laceration (linear but very rough)
        if aspect >= 3.0 and smooth < 0.2:
            logger.debug(f"→ Laceration (aspect {aspect:.2f}, rough)")
            return WoundType.LACERATION
        
        # RULE 3: Puncture (small and round)
        if area < 2.0 and circ > 0.7:
            logger.debug(f"→ Puncture (small {area:.1f}cm², circular {circ:.2f})")
            return WoundType.PUNCTURE
        
        # RULE 4: Pressure Ulcer (large, irregular, low solidity)
        if area > 15.0 and circ < 0.5 and solid < 0.7:
            logger.debug(f"→ Pressure ulcer (large {area:.1f}cm², irregular)")
            return WoundType.PRESSURE_ULCER
        
        # RULE 5: Diabetic Ulcer (medium, round-ish)
        if 3.0 < area < 30.0 and circ > 0.65:
            logger.debug(f"→ Diabetic ulcer (medium {area:.1f}cm², round {circ:.2f})")
            return WoundType.DIABETIC_ULCER
        
        # RULE 6: Burn (irregular but not too elongated)
        if aspect < 2.0 and solid < 0.85 and area > 2.0:
            logger.debug(f"→ Burn (irregular, aspect {aspect:.2f})")
            return WoundType.BURN
        
        # RULE 7: Abrasion (irregular, medium size)
        if solid < 0.75 and 1.0 < area < 20.0:
            logger.debug(f"→ Abrasion (irregular)")
            return WoundType.ABRASION
        
        # DEFAULT: Unknown
        logger.debug(f"→ Unknown (aspect={aspect:.2f}, circ={circ:.2f}, "
                    f"smooth={smooth:.3f}, straight={straight}, sutures={sutures})")
        return WoundType.UNKNOWN
    
    def _get_default_features(self) -> dict:
        """Return default features for empty/invalid masks"""
        return {
            'aspect_ratio': 1.0,
            'circularity': 0.0,
            'solidity': 0.0,
            'elongation': 1.0,
            'area_cm2': 0.0,
            'perimeter_cm': 0.0,
            'has_straight_edges': False,
            'edge_smoothness': 0.0,
            'symmetry_score': 0.5,
            'has_sutures': False,
            'length_cm': 0.0,
            'width_cm': 0.0
        }