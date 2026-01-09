import os
import time
import logging
import argparse
import warnings
import signal
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms

from models import get_model, SCRFD
from utils.general import compute_euler_angles_from_rotation_matrices, draw_cube, draw_axis

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def parse_args():
    parser = argparse.ArgumentParser(description="Head pose estimation inference.")
    parser.add_argument("--network", type=str, default="mobilenetv3_small", help="Model name (resnet18, mobilenetv3_small, etc.)")
    parser.add_argument("--input", type=str, default="0", help="Path to input video file or camera id")
    parser.add_argument("--view", action="store_true", help="Display the inference results")
    parser.add_argument("--draw-type", type=str, default="axis", choices=["cube", "axis"], help="Draw cube or axis for head pose")
    parser.add_argument("--weights", type=str, required=True, help="Path to head pose estimation model weights")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output file")
    return parser.parse_args()


def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch


def expand_bbox(x_min, y_min, x_max, y_max, factor=0.2):
    width = x_max - x_min
    height = y_max - y_min
    x_min_new = x_min - int(factor * height)
    y_min_new = y_min - int(factor * width)
    x_max_new = x_max + int(factor * height)
    y_max_new = y_max + int(factor * width)
    return max(0, x_min_new), max(0, y_min_new), x_max_new, y_max_new


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load models ---
    try:
        face_detector = SCRFD(model_path="./weights/det_10g.onnx")
        logging.info("✅ Face Detection model loaded.")
    except Exception as e:
        logging.info(f"⚠️ Error loading face detector: {e}")

    try:
        head_pose = get_model(params.network, num_classes=6, pretrained=False)
        state_dict = torch.load(params.weights, map_location=device)
        head_pose.load_state_dict(state_dict)
        logging.info("✅ Head Pose model loaded.")
    except Exception as e:
        logging.info(f"⚠️ Error loading head pose model: {e}")

    head_pose.to(device)
    head_pose.eval()

    # --- Video setup ---
    video_source = params.input
    cap = cv2.VideoCapture(int(video_source) if video_source.isdigit() or video_source == '0' else video_source)
    if not cap.isOpened():
        raise IOError("❌ Cannot open webcam or IP stream")

    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(params.output, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
    else:
        out = None

    yaw_history, pitch_history = [], []
    latency_list = []

    def print_latency_summary():
        """Print average latency when exiting."""
        if len(latency_list) > 0:
            avg_latency = sum(latency_list) / len(latency_list)
            print(f"\n\n=== Program Dihentikan (Ctrl+C) ===")
            print(f"Rata-rata latency: {avg_latency:.2f} ms per frame ({1000/avg_latency:.2f} FPS)\n")
            with open("latency_log.txt", "w") as f:
                for i, l in enumerate(latency_list):
                    f.write(f"{i+1},{l:.2f}\n")
                f.write(f"\nAverage Latency: {avg_latency:.2f} ms\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, lambda sig, frame: print_latency_summary())

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                logging.info("❌ Failed to obtain frame or EOF")
                break

            bboxes, keypoints = face_detector.detect(frame)

            # --- Default values (to prevent UnboundLocalError) ---
            yaw, pitch, roll = 0.0, 0.0, 0.0
            direction = "MENCARI WAJAH"

            if len(bboxes) > 0:
                for bbox, keypoint in zip(bboxes, keypoints):
                    x_min, y_min, x_max, y_max = map(int, bbox[:4])
                    width = x_max - x_min
                    x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max)
                    image = frame[y_min:y_max, x_min:x_max]
                    image = pre_process(image).to(device)

                    start = time.time()
                    rotation_matrix = head_pose(image)
                    elapsed = (time.time() - start) * 1000
                    latency_list.append(elapsed)

                    euler = np.degrees(compute_euler_angles_from_rotation_matrices(rotation_matrix))
                    pitch, yaw, roll = euler[:, 0].item(), euler[:, 1].item(), euler[:, 2].item()

                    # --- Smoothing ---
                    yaw_history.append(yaw)
                    pitch_history.append(pitch)
                    if len(yaw_history) > 5:
                        yaw_history.pop(0)
                        pitch_history.pop(0)

                    avg_yaw = sum(yaw_history) / len(yaw_history)
                    avg_pitch = sum(pitch_history) / len(pitch_history)

                    # --- Direction logic ---
                    if abs(avg_yaw) > 25:
                        direction = "DIAM"
                    elif abs(avg_pitch) < 15 and abs(roll) < 20:
                        direction = "MAJU"
                    elif roll > 25:
                        direction = "KIRI"
                    elif roll < -25:
                        direction = "KANAN"
                    elif avg_pitch > -20:
                        direction = "STOP"
                    else:
                        direction = "DIAM"

                    if params.draw_type == "cube":
                        draw_cube(frame, yaw, pitch, roll, bbox=[x_min, y_min, x_max, y_max], size=width)
                    else:
                        draw_axis(frame, yaw, pitch, roll, bbox=[x_min, y_min, x_max, y_max], size_ratio=0.5)

            # --- Portrait mode display (9:16) ---
            if params.view:
                h, w = frame.shape[:2]
                portrait_w, portrait_h = 414, 680  # ukuran seperti layar HP

                scale = max(portrait_w / w, portrait_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(frame, (new_w, new_h))

                # Crop bagian tengah
                x_offset = (new_w - portrait_w) // 2
                y_offset = (new_h - portrait_h) // 2
                cropped = resized[y_offset:y_offset + portrait_h, x_offset:x_offset + portrait_w]

                # Tambahkan footer bawah untuk teks
                footer_height = 80
                canvas = np.zeros((portrait_h + footer_height, portrait_w, 3), dtype=np.uint8)
                canvas[:portrait_h, :] = cropped

                if len(bboxes) == 0:
                    cv2.putText(canvas, "Tidak ada wajah terdeteksi", (20, portrait_h + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(canvas, f"Arah: {direction}", (20, portrait_h + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(canvas, f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}  Roll: {roll:.1f}",
                                (20, portrait_h + 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Tampilkan hasil
                cv2.namedWindow("Head Pose Estimation", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Head Pose Estimation", portrait_w, portrait_h + footer_height)
                cv2.imshow("Head Pose Estimation", canvas)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print_latency_summary()

            if out:
                out.write(frame)

    print_latency_summary()


if __name__ == "__main__":
    args = parse_args()
    main(args)
