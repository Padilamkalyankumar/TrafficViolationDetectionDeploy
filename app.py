import streamlit as st
import cv2
import os
import tempfile
import subprocess
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from PIL import Image
from streamlit_drawable_canvas import st_canvas

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

    bg_image = Image.fromarray(frame_rgb)
    canvas = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=2,
        stroke_color="red",
        background_image=bg_image,
        height=h,
        width=w,
        drawing_mode="point",
        key="draw_points",
    )

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
            subprocess.call([
                "ffmpeg", "-y", "-i", temp_out,
                "-vcodec", "libx264", "-movflags", "faststart", final_output
            ])

            st.success("✅ Detection completed! Unique violations captured.")
            st.video(final_output)
            st.info("📁 Traffic violations → `detected_violations/traffic/`\n📁 Helmet violations → `detected_violations/helmet/`")
