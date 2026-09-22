#!/usr/bin/env python3
"""
thermal_video_splitter_opencv.py
使用 OpenCV 将热红外长视频按固定时长（默认 30 秒）切分成多个短视频。
逐帧读取写入，可插入自定义帧处理逻辑。

python E:/real/use_code/yoloV8/pre_data/split_thermal_video.py E:/real/use_code/all_use/20240813T093911-A1021148.MP4 -o E:/real/use_code/split_all_use/20240813T093911-A1021148_output_clips
"""

import cv2
import argparse
from pathlib import Path
import math


def split_video_opencv(
    input_path: str,
    output_dir: str = "output_clips",
    segment_duration: float = 30.0,
    output_prefix: str = "clip",
    output_format: str = "mp4",
    fourcc_code: str = "mp4v",
):
    """
    使用 OpenCV 逐帧切分视频。
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_duration = total_frames / fps if fps > 0 else 0

    frames_per_segment = int(round(segment_duration * fps))
    total_segments = math.ceil(total_frames / frames_per_segment) if fps > 0 else 1

    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)

    print(f" 输入: {input_path}")
    print(f" FPS: {fps:.2f} | 分辨率: {width}x{height} | 总帧数: {total_frames}")
    print(f"  总时长: {total_duration:.2f} 秒 | 每段: {segment_duration} 秒 ({frames_per_segment} 帧)")
    print(f" 预计输出: {total_segments} 个片段")
    print("-" * 50)

    segment_index = 0
    frame_in_segment = 0
    writer = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 可选：在此处插入帧处理逻辑 ──
        # 例如：辐射定温、伪彩色映射、去噪等
        # frame = process_thermal_frame(frame)
        # ─────────────────────────────────

        # 需要新片段时创建 writer
        if frame_in_segment == 0:
            segment_index += 1
            output_name = f"{output_prefix}_{segment_index:04d}.{output_format}"
            output_path = str(output_dir / output_name)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"无法创建输出文件: {output_path}")
            print(f"[{segment_index}/{total_segments}] 开始写入 {output_name} ...")

        writer.write(frame)
        frame_in_segment += 1
        frame_idx += 1

        # 当前片段写满，释放 writer
        if frame_in_segment >= frames_per_segment:
            writer.release()
            file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
            print(f"   完成 ({file_size_mb:.1f} MB)")
            frame_in_segment = 0

        # 进度提示
        if frame_idx % (fps * 10) == 0 and fps > 0:
            pct = frame_idx / total_frames * 100
            print(f"    进度: {frame_idx}/{total_frames} ({pct:.1f}%)")

    # 收尾最后一个未满的片段
    if writer is not None and frame_in_segment > 0:
        writer.release()
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"   最后一段完成 ({file_size_mb:.1f} MB)")

    cap.release()
    print("-" * 50)
    print(f" 完成！输出目录: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="OpenCV 版视频切分")
    parser.add_argument("input", help="输入视频文件路径")
    parser.add_argument("-o", "--output-dir", default="output_clips", help="输出目录")
    parser.add_argument("-d", "--duration", type=float, default=30.0, help="每段时长/秒")
    parser.add_argument("-p", "--prefix", default="clip", help="文件名前缀")
    parser.add_argument("-f", "--format", default="mp4", help="输出格式")
    parser.add_argument("--fourcc", default="mp4v", help="FourCC 编码 (默认: mp4v, 可选: avc1, XVID)")

    args = parser.parse_args()
    split_video_opencv(
        input_path=args.input,
        output_dir=args.output_dir,
        segment_duration=args.duration,
        output_prefix=args.prefix,
        output_format=args.format,
        fourcc_code=args.fourcc,
    )


if __name__ == "__main__":
    main()