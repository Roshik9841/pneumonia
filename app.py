from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Global variables for models
xray_detector = None
model = None

def load_models():
    global xray_detector, model
    if xray_detector is None:
        try:
            if os.path.exists("improved_xray_detector_model.h5"):
                xray_detector = load_model("improved_xray_detector_model.h5", safe_mode=False)
                print("Loaded improved X-ray detector model")
            else:
                print("Improved X-ray detector model not found. Using enhanced heuristics.")
        except Exception as e:
            print(f"Could not load improved X-ray detector model: {e}. Using enhanced heuristics.")
            xray_detector = None

    if model is None:
        # Load the trained pneumonia model - prefer improved v2, then v1, then simple, then legacy
        if os.path.exists("improved_pneumonia_model_v2.h5"):
            try:
                model = load_model("improved_pneumonia_model_v2.h5", safe_mode=False)
                print("Loaded improved pneumonia model v2")
            except Exception as e:
                print(f"Could not load improved pneumonia model v2: {e}")
                model = None
        elif os.path.exists("improved_pneumonia_model.h5"):
            try:
                model = load_model("improved_pneumonia_model.h5", safe_mode=False)
                print("Loaded improved pneumonia model")
            except Exception as e:
                print(f"Could not load improved pneumonia model: {e}. Falling back to simple model.")
                if os.path.exists("simple_trained_model.h5"):
                    try:
                        model = load_model("simple_trained_model.h5", safe_mode=False)
                        print("Loaded simple trained pneumonia model")
                    except Exception as e2:
                        print(f"Could not load simple model: {e2}")
                        model = None
                else:
                    model = None
        elif os.path.exists("simple_trained_model.h5"):
            try:
                model = load_model("simple_trained_model.h5", safe_mode=False)
                print("Loaded simple trained pneumonia model")
            except Exception as e:
                print(f"Could not load simple model: {e}")
                model = None
        elif os.path.exists("pneumonia_cnn.h5"):
            try:
                model = load_model("pneumonia_cnn.h5", safe_mode=False)
                print("Loaded legacy pneumonia model")
            except Exception as e:
                print(f"Could not load legacy model: {e}")
                model = None
        else:
            print("Pneumonia model file not found. Creating a simple model for testing...")
            # Create a simple CNN model for testing
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print("Created simple fallback pneumonia model for testing")

# Path for uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def cleanup_old_files():
    """Clean up files older than 1 hour to prevent disk space issues"""
    import time
    try:
        current_time = time.time()
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                # Remove files older than 1 hour (3600 seconds)
                if current_time - os.path.getmtime(file_path) > 3600:
                    try:
                        os.remove(file_path)
                        print(f"[CLEANUP] Removed old file: {file_path}")
                    except Exception as e:
                        print(f"[CLEANUP ERROR] Failed to remove {file_path}: {e}")
    except Exception as e:
        print(f"[CLEANUP ERROR] Failed to cleanup old files: {e}")

# Clean up old files on startup
cleanup_old_files()

# X-ray detection using trained model or heuristics
def is_likely_chest_xray(img_path):
    """
    Check if image is a chest X-ray using trained model or heuristics.
    Returns (bool, reason). If validation fails, the caller should report the reason to the user.
    """
    if xray_detector is not None:
        try:
            # Use trained model - X-ray detector expects 150x150
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            prediction = xray_detector.predict(img_array)[0][0]
            # STRONGER threshold: require high confidence for X-ray
            model_thinks_xray = prediction >= 0.7

            # Always run heuristics, even if model says X-ray
            heuristic_valid, heuristic_reason = is_likely_chest_xray_heuristics(img_path)

            if not heuristic_valid:
                # Heuristics think this is NOT a chest X-ray (e.g., blank/black/white/photo)
                return False, f"Image does not appear to be a valid chest X-ray. {heuristic_reason} (model score: {prediction:.2f})"

            if not model_thinks_xray:
                # Both model score and heuristics must agree it looks like a chest X-ray
                return False, f"Image does not appear to be a chest X-ray (model score: {prediction:.2f}). Please upload a clear chest X-ray image."

            # Passed both model and heuristic checks
            return True, ""

        except Exception as exc:
            print(f"X-ray detector model error: {exc}")
            # Fall back to heuristics
            return is_likely_chest_xray_heuristics(img_path)
    else:
        # Use heuristics only
        return is_likely_chest_xray_heuristics(img_path)

# Heuristic-based X-ray detection (enhanced)
def is_likely_chest_xray_heuristics(img_path):
    """
    Enhanced guard to block obvious non–chest X-ray photos before model inference.
    Uses improved heuristics to better detect non-X-ray images.
    Returns (bool, reason). If validation fails, the caller should report the reason to the user.
    """
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        gray = arr.mean(axis=2)

        # Enhanced color content: chest X-rays are almost grayscale
        color_std = np.std(arr, axis=2).mean()
        color_variance = np.var(arr)

        # Brightness and contrast
        mean_gray = float(gray.mean())
        std_gray = float(gray.std())

        # Edge/structure information (lungs & ribs should create gradients)
        grad_y, grad_x = np.gradient(gray)
        edge_strength = float((np.abs(grad_x) + np.abs(grad_y)).mean())

        # Center vs border brightness (lungs brighter than background)
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

        # Additional metrics for better detection
        bright_pixels_ratio = float((gray > 0.9).sum()) / gray.size
        very_bright_ratio = float((gray > 0.95).sum()) / gray.size
        dark_pixels_ratio = float((gray < 0.1).sum()) / gray.size

        # Text-like features (horizontal lines)
        horizontal_lines = 0
        for y in range(10, h-10, 10):
            row_dark_pixels = float((gray[y, :] < 0.3).sum()) / w
            if row_dark_pixels > 0.3:  # More than 30% dark pixels in row
                horizontal_lines += 1

        # Block-like features (diagrams, charts)
        block_score = 0
        block_size = min(h, w) // 8
        for y in range(0, h-block_size, block_size):
            for x in range(0, w-block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                block_std = float(block.std())
                if block_std < 0.05:  # Very uniform blocks
                    block_score += 1

        metrics = {
            "color_std": round(color_std, 4),
            "color_variance": round(color_variance, 4),
            "mean_gray": round(mean_gray, 4),
            "std_gray": round(std_gray, 4),
            "edge_strength": round(edge_strength, 4),
            "center_minus_border": round(center_minus_border, 4),
            "bright_pixels_ratio": round(bright_pixels_ratio, 4),
            "horizontal_lines": horizontal_lines,
            "block_score": block_score
        }
        print(f"[Enhanced X-ray guard] metrics={metrics}")

        # Enhanced checks for non-X-ray images (adjusted based on real X-ray analysis)

        # 1. Color images (photos) - RELAXED threshold
        if color_std > 0.2 or color_variance > 0.15:  # Increased from 0.15 and 0.02
            return False, "Image appears to be a color photo rather than a grayscale X-ray."

        # 2. Mostly white/blank images - ENHANCED detection
        # Check multiple conditions for white/blank images
        is_very_bright = mean_gray > 0.85  # Bright overall
        has_no_contrast = std_gray < 0.01  # Very low contrast
        has_no_edges = edge_strength < 0.001  # Almost no edges
        is_mostly_white = very_bright_ratio > 0.8  # Most pixels very bright
        is_extremely_bright = mean_gray > 0.9  # Extremely bright
        has_low_contrast = std_gray < 0.05  # Low contrast (relaxed)
        has_few_edges = edge_strength < 0.01  # Few edges (relaxed)
        is_overwhelmingly_white = very_bright_ratio > 0.9  # Over 90% white pixels

        if (is_very_bright and has_no_contrast and has_no_edges) or \
           (is_mostly_white and has_no_contrast) or \
           (mean_gray > 0.95 and std_gray < 0.005) or \
           (is_extremely_bright and has_low_contrast and has_few_edges) or \
           (is_mostly_white and has_low_contrast and mean_gray > 0.88) or \
           (is_overwhelmingly_white and mean_gray > 0.85) or \
           (mean_gray > 0.98) or \
           (very_bright_ratio > 0.95):
            return False, "Image appears to be blank/white or mostly uniform color."

        # 3. Text documents (many horizontal lines) - RELAXED threshold
        if horizontal_lines > 15 and bright_pixels_ratio > 0.5:  # Increased from 5 to 15
            return False, "Image appears to contain text/document content."

        # 4. Diagrams/charts (uniform blocks) - RELAXED threshold
        if block_score > 20 and std_gray < 0.1:  # Increased from 8 to 20
            return False, "Image appears to be a diagram or chart."

        # 5. Too dark or too bright overall - SAME
        if not (0.05 < mean_gray < 0.95):
            return False, "Image brightness is outside expected range for X-rays."

        # 6. Lack of structure/contrast - RELAXED threshold
        if std_gray < 0.01:  # Decreased from 0.03 to 0.01
            return False, "Image lacks contrast and structure expected in X-rays."

        # 7. Insufficient anatomical features - RELAXED threshold
        if edge_strength < 0.005:  # Decreased from 0.008 to 0.005
            return False, "Image lacks anatomical structure typical of chest X-rays."

        # 8. Unusual brightness distribution - RELAXED threshold
        if center_minus_border < -0.3:  # Decreased from -0.25 to -0.3 (less restrictive)
            return False, "Image brightness distribution is unusual for chest X-rays."

        return True, ""
    except Exception as exc:
        print(f"Image validation error: {exc}")
        return False, "Unable to analyze image; please upload a valid chest X-ray."


def predict_image(img_path):
    """Improved prediction with more conservative thresholds to reduce false positives"""
    # Check model's expected input shape
    input_shape = model.input_shape  # (None, height, width, channels)
    target_size = (input_shape[1], input_shape[2])  # (height, width)
    
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array, verbose=0)
    raw_score = prediction[0][0]

    # More conservative thresholds to reduce false positives
    if raw_score >= 0.7:  # Pneumonia zone
        result = "Pneumonia"
        # Confidence: distance from decision boundary (at 0.5), minimum 50%
        confidence = max(raw_score, 1 - raw_score)
    elif raw_score <= 0.4:  # Normal zone
        result = "Normal"
        # Confidence: distance from decision boundary (at 0.5), minimum 50%
        confidence = max(raw_score, 1 - raw_score)
    else:
        # Uncertainty zone: 0.4 < raw_score < 0.7
        result = "Uncertain - Consult Doctor"
        # Symmetric confidence around 0.5, still 50–100%
        confidence = max(raw_score, 1 - raw_score)

    return {
        "prediction": result,
        "confidence": float(confidence),
        "raw_score": float(raw_score)
    }

# Simple health check endpoint
@app.route("/", methods=["GET"])
def health_check():
    # Periodic cleanup
    cleanup_old_files()

    # Count files in upload folder
    try:
        upload_files = len([f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))])
    except:
        upload_files = 0

    return jsonify({
        "status": "ok",
        "models_loaded": {
            "xray_detector": xray_detector is not None,
            "pneumonia_model": model is not None
        },
        "upload_folder_files": upload_files
    })

# Manual cleanup endpoint
@app.route("/api/cleanup", methods=["POST"])
def manual_cleanup():
    """Manually clean up all files in upload folder"""
    try:
        cleanup_old_files()
        removed_count = 0
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    removed_count += 1
                    print(f"[MANUAL CLEANUP] Removed file: {file_path}")
                except Exception as e:
                    print(f"[MANUAL CLEANUP ERROR] Failed to remove {file_path}: {e}")

        return jsonify({
            "status": "success",
            "message": f"Cleaned up {removed_count} files from upload folder"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Cleanup failed: {str(e)}"
        }), 500

# API endpoint for PHP frontend
@app.route("/api/predict", methods=["POST"])
def api_predict():
    # Load models if not already loaded
    load_models()
    
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.mimetype in ["image/jpeg", "image/png"]:
        return jsonify({"error": "Only JPEG or PNG files allowed"}), 400

    # Generate a secure filename to prevent path traversal attacks
    from werkzeug.utils import secure_filename
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        # Save file temporarily for validation
        file.save(file_path)
        print(f"[UPLOAD] Saved file temporarily: {file_path}")

        # Validate if it's a chest X-ray
        is_valid_xray, reason = is_likely_chest_xray(file_path)
        if not is_valid_xray:
            # Remove the invalid file immediately
            try:
                os.remove(file_path)
                print(f"[CLEANUP] Removed invalid file: {file_path}")
            except Exception as cleanup_error:
                print(f"[ERROR] Failed to remove invalid file {file_path}: {cleanup_error}")

            message = reason or "Image rejected: please upload a chest X-ray."
            message += " Please upload a front-view chest X-ray (grayscale JPEG/PNG) showing both lungs, without annotations or people."
            return jsonify({"error": message}), 400

        # File is valid, proceed with pneumonia prediction
        result = predict_image(file_path)

        # Keep the valid X-ray file for potential review/analysis
        print(f"[SUCCESS] Valid X-ray processed: {file_path}")

        return jsonify({
            "prediction": result["prediction"],
            "confidence": f"{result['confidence']:.2%}",
            "image_file": filename
        })

    except Exception as e:
        # Clean up file on any error
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"[ERROR CLEANUP] Removed file after error: {file_path}")
        except Exception as cleanup_error:
            print(f"[ERROR] Failed to cleanup after error {file_path}: {cleanup_error}")

        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": "Failed to process image. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)