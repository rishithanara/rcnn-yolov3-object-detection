# R-CNN + YOLOv3 Object Detection Project

A Windows-friendly Streamlit application for detecting and naming objects in uploaded images and videos. It runs **Faster R-CNN** (an R-CNN-family detector) and **YOLOv3** in parallel, then uses class-aware confidence-based non-maximum suppression (NMS) to suppress duplicate boxes. Each result includes an object label, confidence score, bounding box, and detector name.

## Features
- Upload JPG, JPEG, PNG, BMP, or WEBP images.
- Upload MP4, MOV, AVI, or MKV videos and process frame-by-frame.
- Use a live webcam feed with real-time object labels, bounding boxes, and confidence scores (browser camera permission required).
- Run Faster R-CNN and/or YOLOv3.
- Fuse detections and suppress duplicate boxes.
- Download annotated images and videos.
- Show object counts and detection details.

## Important note about the provided nuScenes archive
The provided `v1.0-test_meta(1).tgz` appears to contain nuScenes metadata JSON files and map images. It does **not** contain the full set of camera images referenced by the metadata. You can test the app using your own images/videos. To evaluate on nuScenes, download the appropriate camera data and annotations under the dataset's terms, then implement dataset-specific evaluation. The pretrained COCO models are not trained specifically on nuScenes; they detect the overlapping COCO classes.

## Setup on Windows

1. Install Python 3.10 or 3.11 (64-bit) and make sure `py` works in PowerShell or Command Prompt.
2. Open this project folder in PowerShell or Command Prompt.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run `.venv\Scripts\activate.bat`, or simply run the `run_windows.bat` file.

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. Install a matching PyTorch and torchvision build using the official selector: https://pytorch.org/get-started/locally/ . Confirm it works with `python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"`.
6. Download YOLOv3 model files (internet required):

   ```powershell
   python download_yolov3.py
   ```

   The YOLOv3 weights file is large (about 237 MB). Do not commit model weights to a public repository unless you have checked the relevant license/terms.
7. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```
8. Open the local URL shown in the terminal (usually `http://localhost:8501`). Choose Image, Video, or Live camera. For live mode, allow browser camera access and press START.

For live camera mode, `streamlit-webrtc` and `av` are installed from `requirements.txt`. Camera access works most reliably on `localhost` or an HTTPS deployment. Running both models on every live frame may be slow on CPU; select YOLOv3 only for better responsiveness, or use a supported GPU setup.

## Project structure

```text
rcnn_yolov3_object_detection/
├── app.py                 # Streamlit user interface, image/video/live-camera workflow
├── detectors.py            # Faster R-CNN, YOLOv3, fusion, visualization
├── download_yolov3.py      # Downloads YOLOv3 config/weights/class names
├── requirements.txt
├── run_windows.bat
├── models/                 # YOLOv3 files are downloaded here
└── outputs/                # Reserved for saved results
```

## How the detection fusion works
1. Run both detectors on the same image/frame.
2. Convert predictions to a shared structure: class name, score, and `(x1, y1, x2, y2)`.
3. Sort predictions by confidence.
4. Keep the highest-confidence prediction and suppress overlapping boxes of the same class when IoU exceeds the selected threshold.
5. Draw the retained boxes and labels.

This is output-level fusion, not a single jointly trained neural network. Because both detectors may produce different errors, evaluate fused output against each detector separately before claiming improved accuracy.

## Troubleshooting
- **`ModuleNotFoundError: cv2`**: activate the project environment and run `python -m pip install opencv-python`.
- **`ModuleNotFoundError: streamlit`**: run `python -m pip install -r requirements.txt`.
- **Torch/torchvision mismatch**: reinstall the matched pair using the official PyTorch selector.
- **YOLOv3 missing files**: run `python download_yolov3.py` and confirm the three files exist in `models/`.
- **Slow video processing**: test with a short, lower-resolution video or enable only YOLOv3 in the sidebar.
- **Unexpected object classes**: pretrained models recognize only their supported training categories. Training/fine-tuning and dataset annotations are needed for extra classes.

## Academic reporting note
Report precision, recall, mAP and FPS only after measuring them on a defined labeled test set. Do not present example confidence values as measured results.
