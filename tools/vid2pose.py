from src.dwpose import DWposeDetector
import os
from pathlib import Path

from src.utils.util import get_fps, read_frames, save_videos_from_pil
import numpy as np


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, help="Path to a single video (optional)")
    parser.add_argument("--input_dir", type=str, help="Path to a directory containing videos to process (optional)")
    parser.add_argument("--output_dir", type=str, help="Directory where dw pose videos will be saved. Defaults to input/video folder.")
    args = parser.parse_args()

    # Validate input
    if not args.video_path and not args.input_dir:
        raise ValueError("Either --video_path or --input_dir must be provided")

    # Prepare detector once
    detector = DWposeDetector()
    detector = detector.to("cuda")

    def process_one(video_path, out_dir=None):
        if not os.path.exists(video_path):
            print(f"Skipping, path does not exist: {video_path}")
            return

        dir_path, video_name = (
            os.path.dirname(video_path),
            os.path.splitext(os.path.basename(video_path))[0],
        )
        if out_dir is None:
            out_dir = dir_path
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, video_name + "_kps.mp4")

        print(f"Processing: {video_path} -> {out_path}")
        fps = get_fps(video_path)
        frames = read_frames(video_path)
        kps_results = []
        for i, frame_pil in enumerate(frames):
            result, score = detector(frame_pil)
            score = np.mean(score, axis=-1)
            kps_results.append(result)

        save_videos_from_pil(kps_results, out_path, fps=fps)

    # Single video
    if args.video_path:
        out_dir = args.output_dir if args.output_dir else None
        process_one(args.video_path, out_dir=out_dir)

    # Batch processing (recursive) — preserve input folder structure in output_dir if provided
    if args.input_dir:
        video_exts = {".mp4", ".mov", ".avi", ".mkv"}
        input_dir = args.input_dir
        if not os.path.isdir(input_dir):
            raise ValueError(f"Input dir: {input_dir} not exists or is not a directory")
        out_root = args.output_dir if args.output_dir else None

        videos_found = False
        for root, _, files in os.walk(input_dir):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in video_exts:
                    videos_found = True
                    src = os.path.join(root, f)
                    # If user provided an output root, mirror the relative path
                    if out_root:
                        rel = os.path.relpath(root, input_dir)
                        target_out_dir = os.path.join(out_root, rel) if rel != os.curdir else out_root
                    else:
                        target_out_dir = None
                    process_one(src, out_dir=target_out_dir)
        if not videos_found:
            print(f"No videos found in {input_dir}")

