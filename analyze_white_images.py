"""
Analyze white image metrics to fix detection
"""

import numpy as np
from PIL import Image
import os

def analyze_white_image_metrics():
    """Create and analyze a white image to see its metrics"""
    print("=== ANALYZING WHITE IMAGE METRICS ===")

    # Create different types of "white" images
    test_images = {
        'pure_white': create_pure_white_image(),
        'off_white': create_off_white_image(),
        'white_with_noise': create_white_with_noise(),
        'white_document': create_white_document()
    }

    for img_type, img_path in test_images.items():
        if img_path and os.path.exists(img_path):
            print(f"\n{img_type.upper()}:")
            metrics = analyze_image_metrics(img_path)
            if metrics:
                print(f"  Metrics: {metrics}")

                # Check current rejection criteria
                rejected = False
                reasons = []

                # Current white/blank check
                if metrics["very_bright_ratio"] > 0.7 and metrics["mean_gray"] > 0.9 and metrics["edge_strength"] < 0.005:
                    rejected = True
                    reasons.append("Current white/blank check")

                # Additional checks we might need
                if metrics["std_gray"] < 0.005:  # Very low contrast
                    reasons.append("Very low contrast")
                if metrics["mean_gray"] > 0.95:  # Extremely bright
                    reasons.append("Extremely bright")
                if metrics["edge_strength"] < 0.001:  # Almost no edges
                    reasons.append("Almost no edges")

                if rejected or len(reasons) > 1:
                    print(f"  ❌ WOULD BE REJECTED: {', '.join(reasons)}")
                else:
                    print("  ⚠️  MIGHT PASS THROUGH")

            # Clean up
            try:
                os.remove(img_path)
            except:
                pass

def create_pure_white_image():
    """Create a pure white image"""
    img = np.full((224, 224, 3), 255, dtype=np.uint8)
    pil_img = Image.fromarray(img, 'RGB')
    path = 'test_pure_white.jpg'
    pil_img.save(path, 'JPEG', quality=95)
    return path

def create_off_white_image():
    """Create an off-white image"""
    img = np.full((224, 224, 3), 240, dtype=np.uint8)  # Slightly off-white
    pil_img = Image.fromarray(img, 'RGB')
    path = 'test_off_white.jpg'
    pil_img.save(path, 'JPEG', quality=95)
    return path

def create_white_with_noise():
    """Create a white image with slight noise"""
    img = np.full((224, 224, 3), 255, dtype=np.uint8)
    # Add slight noise
    noise = np.random.normal(0, 2, (224, 224, 3))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img, 'RGB')
    path = 'test_white_noise.jpg'
    pil_img.save(path, 'JPEG', quality=95)
    return path

def create_white_document():
    """Create a white document-like image"""
    img = np.full((224, 224, 3), 250, dtype=np.uint8)
    # Add some subtle texture
    for i in range(10):
        x, y = np.random.randint(0, 200, 2)
        w, h = np.random.randint(5, 15, 2)
        img[y:y+h, x:x+w] = 245  # Slightly darker patches
    pil_img = Image.fromarray(img, 'RGB')
    path = 'test_white_document.jpg'
    pil_img.save(path, 'JPEG', quality=95)
    return path

def analyze_image_metrics(img_path):
    """Analyze image metrics (copied from app.py)"""
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        gray = arr.mean(axis=2)

        # Color content
        color_std = np.std(arr, axis=2).mean()
        color_variance = np.var(arr)

        # Brightness and contrast
        mean_gray = float(gray.mean())
        std_gray = float(gray.std())

        # Edge/structure information
        grad_y, grad_x = np.gradient(gray)
        edge_strength = float((np.abs(grad_x) + np.abs(grad_y)).mean())

        # Center vs border brightness
        h, w = gray.shape
        center = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean()
        borders = np.concatenate([
            gray[:h // 8, :].flatten(),
            gray[-h // 8:, :].flatten(),
            gray[:, :w // 8].flatten(),
            gray[:, -w // 8:].flatten()
        ])
        border_mean = float(borders.mean()) if borders.size else center
        center_minus_border = center - border_mean

        # Additional metrics
        bright_pixels_ratio = float((gray > 0.9).sum()) / gray.size
        very_bright_ratio = float((gray > 0.95).sum()) / gray.size
        dark_pixels_ratio = float((gray < 0.1).sum()) / gray.size

        return {
            "color_std": round(color_std, 4),
            "color_variance": round(color_variance, 4),
            "mean_gray": round(mean_gray, 4),
            "std_gray": round(std_gray, 4),
            "edge_strength": round(edge_strength, 4),
            "bright_pixels_ratio": round(bright_pixels_ratio, 4),
            "very_bright_ratio": round(very_bright_ratio, 4),
            "center_minus_border": round(center_minus_border, 4)
        }
    except Exception as e:
        print(f"Error analyzing {img_path}: {e}")
        return None

if __name__ == "__main__":
    analyze_white_image_metrics()