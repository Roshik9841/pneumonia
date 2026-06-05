### Run the Pneumonia API (Windows)

These steps use Command Prompt on Windows. Paths below assume the project folder is at `C:\xampp\htdocs\Pneumonia`.

### 1) Prerequisites
- **Python**: 3.9–3.11 recommended. 3.13 can work but some wheels may be slower to install.
- **Pip**: Up to date (`python -m pip install --upgrade pip`).
- Optional: A virtual environment.

### 2) Create and activate a virtual environment (recommended)
```bat
cd /d C:\xampp\htdocs\Pneumonia
python -m venv .venv
.venv\Scripts\activate
```

### 3) Install Python packages
There is no `requirements.txt` yet, so install the essentials directly:
```bat
pip install flask tensorflow==2.* pillow numpy
```

If you hit build errors on TensorFlow, try a different minor version of Python (e.g., 3.10/3.11) or install a specific TF version compatible with your Python.

### 4) Train the X-ray Detector (Optional)
The app includes an X-ray detection filter that checks if uploaded images are actually chest X-rays before running pneumonia detection. To train or retrain this model:

```bat
python train_xray_detector.py
```

This creates `xray_detector_model.h5` using the chest X-ray dataset and synthetic non-X-ray images.

### 5) Start the server
```bat
python app.py
```
You should see lines like:
- `Loaded X-ray detector model`
- `Loaded improved pneumonia model`
- `Running on http://127.0.0.1:5000`

### 6) Health check
Open a new Command Prompt window (leave the server running) and run:
```bat
curl http://127.0.0.1:5000/
```
Expected response:
```json
{"status":"ok"}
```

### 7) Make a prediction
Replace the image path with a real chest X‑ray image you have:
```bat
curl -X POST -F "file=@C:\path\to\chest_xray.jpg" http://127.0.0.1:5000/api/predict
```

The API now includes X-ray validation:
- If the image is not detected as a chest X-ray, it returns an error message
- Only confirmed chest X-rays proceed to pneumonia detection

### 8) File uploads
Uploaded files are saved to the `uploads` folder alongside `app.py`.

### Notes about models
- The server loads the first available model in this order:
  1. `improved_pneumonia_model.h5`
  2. `simple_trained_model.h5`
  3. `pneumonia_cnn.h5`
- If none exist, a tiny fallback test model may be built at runtime.

### Why startup can feel slow
- **TensorFlow initialization**: On first import, TF probes CPU features and initializes oneDNN; this prints the `oneDNN custom operations are on` and `cpu_feature_guard` messages and can take several seconds.
- **Model load**: Loading `improved_pneumonia_model.h5` from disk and preparing it for inference adds a few seconds depending on CPU/disk speed.
- **Flask debug reloader**: With `debug=True`, Flask restarts the server once after initial load, which causes the model to load twice. This is normal in development and explains duplicated logs like `Loaded improved model`.

### Speed up tips (optional)
- Start without Flask reloader:
  ```bat
  set FLASK_ENV=production
  python -c "import app; app.app.run(host='0.0.0.0', port=5000, debug=False)"
  ```
- Suppress TF info logs (less noise, not faster):
  ```bat
  set TF_CPP_MIN_LOG_LEVEL=2
  python app.py
  ```
- If port 5000 is busy, change the port:
  ```bat
  python -c "import app; app.app.run(host='0.0.0.0', port=5001, debug=True)"
  ```

### Troubleshooting
- `ModuleNotFoundError: No module named 'flask'` → Run `pip install flask` in the active venv.
- `No module named 'tensorflow'` → Run `pip install tensorflow==2.*` and ensure Python 3.9–3.11.
- Server doesn't respond → Ensure it prints `Running on http://127.0.0.1:5000` and you are curling the same port.

