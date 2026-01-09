"""
Generate synthetic wound images for testing
Creates realistic-looking wound samples with different severities
"""
import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_synthetic_wound(
    severity: str = "moderate",
    size: tuple = (512, 512),
    output_path: str = None
) -> np.ndarray:
    """
    Generate a synthetic wound image
    
    Args:
        severity: "mild", "moderate", or "severe"
        size: (width, height) of image
        output_path: Optional path to save image
        
    Returns:
        BGR image array
    """
    w, h = size
    
    # Create base skin-colored image
    # Realistic skin tone (light)
    skin_color = (220, 190, 170)  # BGR
    img = np.ones((h, w, 3), dtype=np.uint8) * np.array(skin_color, dtype=np.uint8)
    
    # Add skin texture variation
    noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Define wound parameters based on severity
    center = (w // 2, h // 2)
    
    if severity == "mild":
        radius = min(h, w) // 10
        wound_color = (80, 90, 180)  # Light red/pink
        redness_radius = int(radius * 1.2)
    elif severity == "moderate":
        radius = min(h, w) // 6
        wound_color = (60, 70, 160)  # Medium red
        redness_radius = int(radius * 1.3)
    else:  # severe
        radius = min(h, w) // 4
        wound_color = (40, 50, 140)  # Dark red
        redness_radius = int(radius * 1.5)
    
    # Draw inflamed/red area around wound
    cv2.circle(img, center, redness_radius, (100, 120, 200), -1)
    
    # Blend the redness
    mask_redness = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_redness, center, redness_radius, 255, -1)
    
    # Smooth the redness boundary
    mask_redness = cv2.GaussianBlur(mask_redness, (51, 51), 0)
    
    # Apply redness with transparency
    redness_layer = img.copy()
    cv2.circle(redness_layer, center, redness_radius, (100, 110, 200), -1)
    img = cv2.addWeighted(img, 0.6, redness_layer, 0.4, 0)
    
    # Draw main wound
    cv2.circle(img, center, radius, wound_color, -1)
    
    # Add wound texture (darker spots, uneven coloring)
    wound_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(wound_mask, center, radius, 255, -1)
    
    # Add texture variation within wound
    for _ in range(10):
        x = center[0] + np.random.randint(-radius//2, radius//2)
        y = center[1] + np.random.randint(-radius//2, radius//2)
        r = radius // 4
        darkness = np.random.randint(20, 60)
        color = tuple(max(0, c - darkness) for c in wound_color)
        cv2.circle(img, (x, y), r, color, -1)
    
    # Blend wound edges
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Add some edge definition
    edges = cv2.Canny(wound_mask, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    img[edges_dilated > 0] = [30, 40, 100]  # Dark edge
    
    # Final light noise for realism
    final_noise = np.random.normal(0, 3, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + final_noise, 0, 255).astype(np.uint8)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved {severity} wound to: {output_path}")
    
    return img


def main():
    """Generate sample wound images"""
    print("Generating synthetic wound images...")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate different severity levels
    severities = ["mild", "moderate", "severe"]
    
    for severity in severities:
        output_path = output_dir / f"wound_{severity}.jpg"
        create_synthetic_wound(
            severity=severity,
            size=(512, 512),
            output_path=str(output_path)
        )
    
    print(f"\n✓ Generated {len(severities)} sample images in: {output_dir}")
    print("\nYou can now use these images to test the wound analysis API!")
    print("\nExample usage:")
    print("  curl -X POST http://localhost:8000/api/wounds/analyze \\")
    print(f"    -F 'file=@{output_dir}/wound_moderate.jpg' \\")
    print("    -F 'patient_id=test_patient'")


if __name__ == "__main__":
    main()
