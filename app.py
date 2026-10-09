"""R-CNN + YOLOv3 object detection for uploaded images and videos."""
from pathlib import Path
import tempfile
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from detectors import detect_faster_rcnn, detect_yolov3, fuse_detections, draw_detections, MODELS_DIR

st.set_page_config(page_title="R-CNN + YOLOv3 Object Detection", page_icon="🚦", layout="wide")
st.title("R-CNN + YOLOv3 Object Detection")
st.write("Detect and name objects in uploaded images, uploaded videos, or a live webcam feed using Faster R-CNN + YOLOv3.")
st.info("This app runs two pretrained detectors and fuses their detections. The included nuScenes archive contains metadata and map images, not the camera-image dataset itself; use an image/video upload or obtain the full nuScenes data separately.")

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider("Minimum confidence", 0.10, 0.90, 0.35, 0.05)
    iou = st.slider("Duplicate-box IoU threshold", 0.20, 0.80, 0.50, 0.05)
    run_rcnn = st.checkbox("Run Faster R-CNN", True)
    run_yolo = st.checkbox("Run YOLOv3", True)
    st.caption("YOLOv3 requires its config, weights and COCO class names in the models/ folder. See README.md.")

kind = st.radio("Choose input type", ["Image", "Video", "Live camera"], horizontal=True)
file_types = ["jpg", "jpeg", "png", "bmp", "webp"] if kind == "Image" else ["mp4", "mov", "avi", "mkv"]
file = st.file_uploader(f"Upload a {kind.lower()}", type=file_types) if kind != "Live camera" else None

if kind == "Live camera":
    st.subheader("Live object detection")
    st.warning("Allow camera access in your browser. Both models can be computationally heavy; for smoother live detection, select YOLOv3 only in the sidebar.")
    if not (run_rcnn or run_yolo):
        st.warning("Select at least one detector in the sidebar.")
    else:
        try:
            from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
            import av

            class LiveDetector(VideoProcessorBase):
                def __init__(self):
                    self.lock = __import__("threading").Lock()
                    self.confidence = confidence
                    self.iou = iou
                    self.run_rcnn = run_rcnn
                    self.run_yolo = run_yolo
                    self.last_error = ""

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    try:
                        detections = []
                        if self.run_rcnn:
                            detections.extend(detect_faster_rcnn(img, self.confidence))
                        if self.run_yolo:
                            detections.extend(detect_yolov3(img, self.confidence))
                        final = fuse_detections(detections, self.iou)
                        img = draw_detections(img, final)
                        cv2.putText(img, f"Objects: {len(final)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                        self.last_error = ""
                    except Exception as e:
                        self.last_error = str(e)
                        cv2.putText(img, "Detection error - check model setup", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            webrtc_streamer(
                key="live-object-detection",
                video_processor_factory=LiveDetector,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            st.caption("Press START to begin camera detection and STOP to end it. Object names and confidence scores are drawn on the live frames.")
        except ImportError:
            st.error("Live camera support is not installed. Run: python -m pip install streamlit-webrtc av")

if file and not (run_rcnn or run_yolo):
    st.warning("Select at least one detector in the sidebar.")
elif file and kind == "Image" and (run_rcnn or run_yolo):
    image = Image.open(file).convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    all_dets = []
    status = st.status("Running selected detectors…", expanded=True)
    if run_rcnn:
        try:
            d = detect_faster_rcnn(frame, confidence)
            all_dets.extend(d)
            status.write(f"Faster R-CNN: {len(d)} detections")
        except Exception as e:
            status.write(f"Faster R-CNN unavailable: {e}")
    if run_yolo:
        try:
            d = detect_yolov3(frame, confidence)
            all_dets.extend(d)
            status.write(f"YOLOv3: {len(d)} detections")
        except Exception as e:
            status.write(f"YOLOv3 unavailable: {e}")
    final = fuse_detections(all_dets, iou)
    status.update(label=f"Done — {len(final)} final detections", state="complete")
    rendered = draw_detections(frame, final)
    left, right = st.columns(2)
    with left:
        st.subheader("Input image")
        st.image(image, use_container_width=True)
    with right:
        st.subheader("Detected objects")
        st.image(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB), use_container_width=True)
    if final:
        st.subheader("Detection details")
        st.dataframe([{"Object": x["label"], "Confidence": f'{x["score"]:.1%}', "Box (x1,y1,x2,y2)": str(tuple(x["box"])) , "Model": x["model"]} for x in final], use_container_width=True)
        counts = {}
        for x in final: counts[x["label"]] = counts.get(x["label"], 0) + 1
        st.write("**Object counts:**", counts)
    else:
        st.warning("No objects passed the confidence threshold. Confirm model files are installed and try a lower threshold.")
    ok, encoded = cv2.imencode(".jpg", rendered)
    if ok: st.download_button("Download annotated image", encoded.tobytes(), "detected_objects.jpg", "image/jpeg")

elif file and kind == "Video" and (run_rcnn or run_yolo):
    st.warning("Video detection processes frames sequentially and can be slow. The output video uses the selected detectors.")
    if st.button("Process video", type="primary"):
        suffix = Path(file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
            src.write(file.getvalue()); src_path = src.name
        out_path = str(Path(tempfile.gettempdir()) / f"detected_{int(time.time())}.mp4")
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            st.error("Could not open this video. Try MP4 (H.264) or AVI.")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = st.progress(0.0); status = st.empty(); frame_idx = 0; class_counts = {}
            while True:
                ok, frame = cap.read()
                if not ok: break
                dets = []
                if run_rcnn:
                    try: dets.extend(detect_faster_rcnn(frame, confidence))
                    except Exception as e: st.error(f"Faster R-CNN error: {e}"); break
                if run_yolo:
                    try: dets.extend(detect_yolov3(frame, confidence))
                    except Exception as e: st.error(f"YOLOv3 error: {e}"); break
                fused = fuse_detections(dets, iou)
                for d in fused: class_counts[d['label']] = class_counts.get(d['label'], 0) + 1
                writer.write(draw_detections(frame, fused)); frame_idx += 1
                if total > 0: progress.progress(min(frame_idx / total, 1.0))
                if frame_idx % 10 == 0: status.write(f"Processed {frame_idx}/{total or '?'} frames")
            cap.release(); writer.release()
            if Path(out_path).exists() and Path(out_path).stat().st_size > 0:
                st.success(f"Finished processing {frame_idx} frames.")
                st.video(out_path)
                with open(out_path, "rb") as f: st.download_button("Download annotated video", f.read(), "detected_video.mp4", "video/mp4")
                st.write("**Total detections across processed frames:**", class_counts)
            else: st.error("No output video was created. Check model files and video codec support.")
