"""
extract_faces.py

Extracts cropped face images from FaceForensics++ videos using MTCNN.

Usage:
    python extract_faces.py --input_dir data/raw/original_sequences/youtube/c40/videos --output_dir data/faces_extracted/real --label real
    python extract_faces.py --input_dir data/raw/manipulated_sequences/Deepfakes/c40/videos --output_dir data/faces_extracted/fake --label fake
"""

import argparse
import os
from pathlib import Path

import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Extract faces from FF++ videos.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing .mp4 video files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save cropped face images.")
    parser.add_argument("--label", type=str, required=True, choices=["real", "fake"],
                        help="Label for this batch of videos (used only for logging).")
    parser.add_argument("--frames_per_video", type=int, default=10,
                        help="Number of frames to sample per video.")
    parser.add_argument("--face_size", type=int, default=224,
                        help="Output face crop size (square, in pixels).")
    parser.add_argument("--margin", type=int, default=40,
                        help="Pixel margin added around detected face box before cropping.")
    parser.add_argument("--min_face_confidence", type=float, default=0.90,
                        help="Minimum MTCNN detection confidence to keep a face.")
    return parser.parse_args()


def sample_frame_indices(total_frames, n_samples):
    if total_frames <= n_samples:
        return list(range(total_frames))
    start = int(total_frames * 0.05)
    end = int(total_frames * 0.95)
    step = max(1, (end - start) // n_samples)
    return list(range(start, end, step))[:n_samples]


def extract_from_video(video_path, mtcnn, output_dir, frames_per_video, log_file):
    video_id = video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        log_file.write(f"SKIP (unreadable/0 frames): {video_path}\n")
        cap.release()
        return 0

    indices_to_sample = set(sample_frame_indices(total_frames, frames_per_video))
    saved_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices_to_sample:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = mtcnn.detect(rgb_frame)

            if boxes is not None and len(boxes) > 0:
                best_idx = probs.argmax()
                if probs[best_idx] is not None and probs[best_idx] >= mtcnn.min_face_confidence:
                    x1, y1, x2, y2 = [int(v) for v in boxes[best_idx]]
                    h, w, _ = rgb_frame.shape
                    m = mtcnn.margin
                    x1, y1 = max(0, x1 - m), max(0, y1 - m)
                    x2, y2 = min(w, x2 + m), min(h, y2 + m)
                    face_crop = rgb_frame[y1:y2, x1:x2]

                    if face_crop.size > 0:
                        face_crop = cv2.resize(face_crop, (mtcnn.face_size, mtcnn.face_size))
                        out_path = output_dir / f"{video_id}_{frame_idx}.jpg"
                        cv2.imwrite(str(out_path), cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
                        saved_count += 1
                else:
                    log_file.write(f"LOW_CONF frame {frame_idx} in {video_id}\n")
            else:
                log_file.write(f"NO_FACE frame {frame_idx} in {video_id}\n")

        frame_idx += 1

    cap.release()
    return saved_count


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mtcnn = MTCNN(keep_all=False, device=device)
    mtcnn.margin = args.margin
    mtcnn.face_size = args.face_size
    mtcnn.min_face_confidence = args.min_face_confidence

    video_files = sorted(input_dir.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No .mp4 files found in {input_dir}")

    print(f"Found {len(video_files)} videos ({args.label}). Extracting up to "
          f"{args.frames_per_video} faces each -> {output_dir}")

    log_path = output_dir / "extraction_log.txt"
    total_saved = 0
    with open(log_path, "w") as log_file:
        for video_path in tqdm(video_files, desc=f"Extracting ({args.label})"):
            saved = extract_from_video(video_path, mtcnn, output_dir,
                                       args.frames_per_video, log_file)
            total_saved += saved

    print(f"Done. Saved {total_saved} face crops to {output_dir}")
    print(f"Skipped-frame log written to {log_path}")


if __name__ == "__main__":
    main()