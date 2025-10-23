import streamlit as st
import cv2
import os
import tempfile
import subprocess
import numpy as np
import shutil
import zipfile
from ultralytics import YOLO
from datetime import datetime
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import io
from typing import List
import base64
import inspect


@st.cache_data
def make_zip_bytes(root_dir: str = "detected_violations") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(root_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, root_dir)
                zf.write(fpath, arcname)
    buf.seek(0)
    return buf.read()


@st.cache_data
def read_file_bytes(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read()


@st.cache_data
def make_combined_zip_bytes(video_path: str | None, root_dir: str = "detected_violations") -> bytes:
    """Create a zip in-memory containing the processed video (if present) and all detected images.

    video_path: path to processed video or temp output; if None or missing, only images are zipped.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        # add video file under top-level folder 'video'
        if video_path and os.path.exists(video_path):
            arc = os.path.join("video", os.path.basename(video_path))
            zf.write(video_path, arc)

        # add detected images keeping their subfolders
        if os.path.exists(root_dir):
            for root, dirs, files in os.walk(root_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    # store path relative to the detected_violations root
                    rel_root = os.path.relpath(root, root_dir)
                    if rel_root == '.' or rel_root == './':
                        arcname = os.path.join('images', fname)
                    else:
                        arcname = os.path.join('images', rel_root, fname)
                    zf.write(fpath, arcname)

    buf.seek(0)
    return buf.read()

# -------------------------------------------------
# Streamlit setup
# -------------------------------------------------
st.set_page_config(page_title="Traffic Violation Detection", page_icon="🚦", layout="wide")
st.title("🚦 Traffic Violation Detection")
st.write("Upload a traffic video, click two points to draw a boundary line, and detect unique violations (traffic + helmet).")

# -------------------------------------------------
# Upload video
# -------------------------------------------------
uploaded_video = st.file_uploader("📹 Upload traffic video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    st.video(video_path)

    # -------------------------------------------------
    # Capture first frame
    # -------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        st.error("❌ Could not read video.")
        st.stop()

    frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame_rgb.shape

    st.subheader("🟢 Draw a line by clicking two points")
    st.write("Click exactly two points — a thin straight line will connect them automatically.")

    # Prepare background image and a data URL (in case canvas supports background_image_url)
    bg_image = Image.fromarray(frame_rgb)

    def pil_image_to_data_url(pil_img: Image.Image) -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{b64}"

    bg_url = pil_image_to_data_url(bg_image)

    # Build kwargs dynamically depending on st_canvas signature to avoid TypeError
    canvas_kwargs = dict(
        fill_color="rgba(255,255,255,0)",
        stroke_width=2,
        stroke_color="red",
        height=h,
        width=w,
        drawing_mode="point",
        key="draw_points",
    )

    try:
        canvas_sig = inspect.signature(st_canvas)
        if 'background_image_url' in canvas_sig.parameters:
            canvas_kwargs['background_image_url'] = bg_url
        else:
            canvas_kwargs['background_image'] = bg_image
    except (ValueError, TypeError):
        # If introspection fails for any reason, try a safe two-step approach:
        # prefer URL param, fall back to image.
        try:
            canvas_kwargs['background_image_url'] = bg_url
        except Exception:
            canvas_kwargs['background_image'] = bg_image

    canvas = st_canvas(**canvas_kwargs)

    if canvas.json_data and len(canvas.json_data["objects"]) >= 2:
        points = [(obj["left"], obj["top"]) for obj in canvas.json_data["objects"][:2]]
        (x1, y1), (x2, y2) = map(lambda p: (int(p[0]), int(p[1])), points)

        # Draw preview line
        preview = frame_rgb.copy()
        cv2.line(preview, (x1, y1), (x2, y2), (255, 0, 0), 2)
        try:
            st.image(preview, caption="Selected Line Preview", use_container_width=True)
        except TypeError:
            st.image(preview, caption="Selected Line Preview")

        if st.button("🚀 Start Detection"):
            st.info("Processing video... Please wait ⏳")

            # Create directories
            os.makedirs("detected_violations/traffic", exist_ok=True)
            os.makedirs("detected_violations/helmet", exist_ok=True)

            # Load YOLO models
            traffic_model = YOLO("yolo11n.pt")   # vehicle/person detection
            helmet_model = YOLO("best.pt")       # helmet detection model

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_out = "temp_output.mp4"
            out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))
            progress = st.progress(0)
            frame_no = 0

            saved_violations = []

            # ------------------------- Helpers -------------------------
            def ccw(A, B, C):
                return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

            def segments_intersect(A, B, C, D):
                return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

            def box_touches_line(x1b, y1b, x2b, y2b, tolerance=3):
                edges = [
                    ((x1b, y1b), (x2b, y1b)),
                    ((x2b, y1b), (x2b, y2b)),
                    ((x2b, y2b), (x1b, y2b)),
                    ((x1b, y2b), (x1b, y1b))
                ]
                for edge in edges:
                    if segments_intersect((x1, y1), (x2, y2), edge[0], edge[1]):
                        return True
                def point_line_distance(px, py):
                    num = abs((y2 - y1)*px - (x2 - x1)*py + x2*y1 - y2*x1)
                    den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                    return num / den if den != 0 else float('inf')
                corners = [(x1b, y1b), (x2b, y1b), (x2b, y2b), (x1b, y2b)]
                return any(point_line_distance(px, py) < tolerance for px, py in corners)

            def compute_iou(box1, box2):
                x1, y1, x2, y2 = box1
                x1p, y1p, x2p, y2p = box2
                xi1, yi1 = max(x1, x1p), max(y1, y1p)
                xi2, yi2 = min(x2, x2p), min(y2, y2p)
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                box1_area = (x2 - x1) * (y2 - y1)
                box2_area = (x2p - x1p) * (y2p - y1p)
                union_area = box1_area + box2_area - inter_area
                return inter_area / union_area if union_area > 0 else 0

            # ------------------------- Main Loop -------------------------
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                traffic_results = traffic_model(frame, verbose=False)[0]
                bike_boxes = []

                for box in traffic_results.boxes:
                    x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
                    label = traffic_results.names[int(box.cls[0])].lower()
                    if label == "person":
                        continue

                    cx, cy = (x1b + x2b)//2, (y1b + y2b)//2
                    color = (0, 255, 0)

                    if box_touches_line(x1b, y1b, x2b, y2b):
                        color = (0, 0, 255)
                        already_saved = any(abs(cx - sx) < 30 and abs(cy - sy) < 30 for (sx, sy) in saved_violations)
                        if not already_saved:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            cv2.imwrite(f"detected_violations/traffic/{label}_{ts}.jpg", frame[y1b:y2b, x1b:x2b])
                            saved_violations.append((cx, cy))

                    cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), color, 2)
                    cv2.putText(frame, label, (x1b, y1b - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if label in ["motorbike", "bicycle"]:
                        bike_boxes.append((x1b, y1b, x2b, y2b, label, cx, cy))

                # ---------------- Helmet Detection ----------------
                if bike_boxes:
                    helmet_results = helmet_model(frame, verbose=False)[0]
                    helmet_boxes = [tuple(map(int, hb.xyxy[0])) for hb in helmet_results.boxes]

                    for (x1b, y1b, x2b, y2b, label, cx, cy) in bike_boxes:
                        # Define upper 40% region (rider’s head area)
                        head_region = (x1b, y1b, x2b, y1b + int((y2b - y1b) * 0.4))
                        has_helmet = any(compute_iou(head_region, hb) > 0.1 for hb in helmet_boxes)

                        # Helmet violation logic (better detection)
                        if not has_helmet:
                            near_line = box_touches_line(x1b, y1b, x2b, y2b, tolerance=10)
                            if near_line:
                                already_saved = any(abs(cx - sx) < 50 and abs(cy - sy) < 50 for (sx, sy) in saved_violations)
                                if not already_saved:
                                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                    save_path = f"detected_violations/helmet/{label}_{ts}.jpg"
                                    cropped = frame[y1b:y2b, x1b:x2b]
                                    if cropped.size > 0:
                                        cv2.imwrite(save_path, cropped)
                                        saved_violations.append((cx, cy))
                                        print(f"[Helmet Violation] Saved: {save_path}")

                        color = (0, 255, 0) if has_helmet else (0, 0, 255)
                        msg = f"{label} - {'Helmet' if has_helmet else 'No Helmet'}"
                        cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), color, 2)
                        cv2.putText(frame, msg, (x1b, y1b - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw boundary line
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, "Boundary Line", (x1 + 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                out.write(frame)
                frame_no += 1
                if total_frames:
                    progress.progress(min(frame_no / total_frames, 1.0))

            cap.release()
            out.release()

            final_output = "processed_final.mp4"
            # use subprocess.run so we can capture ffmpeg errors and decide a fallback
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", temp_out,
                "-vcodec", "libx264", "-movflags", "faststart", final_output
            ]
            try:
                proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    st.warning("ffmpeg returned non-zero exit code. Falling back to temporary output if available.")
                    st.error(proc.stderr)
            except FileNotFoundError:
                st.warning("ffmpeg executable not found. Ensure ffmpeg is installed and on PATH. Falling back to temporary output if available.")
                proc = None


            st.success("✅ Detection completed! Unique violations captured.")
            st.video(final_output)
            st.info("📁 Traffic violations → `detected_violations/traffic/`\n📁 Helmet violations → `detected_violations/helmet/`")

            # -- Provide download for processed video --
            # Prefer final_output if it exists, otherwise try temp_out
            video_path_for_download = None
            if os.path.exists(final_output):
                video_path_for_download = final_output
            elif os.path.exists(temp_out):
                video_path_for_download = temp_out
                st.info("Using temporary output for download because final re-encode is unavailable.")
            else:
                st.error("No output video found to download.")

            if video_path_for_download:
                try:
                    video_bytes = read_file_bytes(video_path_for_download)
                    st.download_button("⬇️ Download Processed Video", data=video_bytes,
                                       file_name=os.path.basename(video_path_for_download), mime="video/mp4", key="dl_video")
                except Exception as e:
                    st.warning(f"Video download not available: {e}")

            # -- Combined download: video + detected images in one zip --
            try:
                video_for_zip = video_path_for_download if video_path_for_download and os.path.exists(video_path_for_download) else None
                combined_bytes = make_combined_zip_bytes(video_for_zip, root_dir='detected_violations')
                st.download_button("⬇️ Download Processed Video + Detected Images (ZIP)",
                                   data=combined_bytes,
                                   file_name="trafficviolations_package.zip",
                                   mime="application/zip",
                                   key="dl_combined")
            except Exception as e:
                st.warning(f"Combined download not available: {e}")

            # optional: cleanup temp files
            try:
                if os.path.exists(temp_out):
                    os.remove(temp_out)
                if os.path.exists(final_output):
                    pass  # keep final_output for download
            except:
                pass

            if st.button("🧹 Cleanup temp files"):
                try:
                    if os.path.exists(temp_out): os.remove(temp_out)
                    if os.path.exists(final_output): os.remove(final_output)
                    st.success("Temporary files removed.")
                    # clear cached bytes
                    st.cache_data.clear()
                except Exception as e:
                    st.warning(f"Cleanup failed: {e}")