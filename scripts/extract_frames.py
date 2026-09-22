import cv2
import os
from tqdm import tqdm


def extract_frames(video_path, save_dir, frame_interval=3):
    """
    提取视频帧：
    :param video_path: 原始热成像视频路径
    :param save_dir: 帧保存根目录
    :param frame_interval: 抽帧间隔（比如设为3，就是每3帧抽1张）
    """
    # 1. 初始化视频读取
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{video_path}，请检查路径和格式")

    # 2. 获取视频基本信息
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / frame_rate if frame_rate > 0 else 0

    if total_frames == 0:
        raise ValueError("视频读取失败，总帧数为0，请检查视频文件是否损坏或编码格式")

    # 计算实际抽取的帧率 (这也是文献里8.6Hz的由来)
    extracted_fps = frame_rate / frame_interval
    extracted_total = total_frames // frame_interval

    # 3. 创建保存目录
    video_name = os.path.basename(video_path).split('.')[0]
    frame_dir = os.path.join(save_dir, video_name)
    os.makedirs(frame_dir, exist_ok=True)

    print("=" * 50)
    print(f"开始处理视频：{video_name}")
    print(f"视频总时长：{video_duration:.1f} 秒")
    print(f"原始视频帧率：{frame_rate:.2f} fps")
    print(f"抽帧间隔设置：每 {frame_interval} 帧抽1张")
    print(f"【重要】实际提取帧率：{extracted_fps:.2f} fps (请记下这个数字！后续算呼吸率要用！)")
    print(f"预计将提取出约 {extracted_total} 张图像")
    print("=" * 50)

    # 4. 逐帧提取
    for idx in tqdm(range(total_frames), desc="帧提取进度"):
        ret, frame = cap.read()
        if not ret:
            break

        # 按照设置的间隔保存图片
        if idx % frame_interval == 0:
            cv2.imwrite(os.path.join(frame_dir, f"{video_name}_frame_{idx:06d}.jpg"), frame)

    cap.release()
    print(f"\n帧提取完成！图片已保存至：{frame_dir}")
    return extracted_fps


if __name__ == "__main__":
    # 换成视频路径
    video_dir = r"E:\real\use_code\yoloV8\Dataset_new\video"
    video_name = "bs200723.MP4"
    video_path = video_dir + "/" + video_name
    save_dir = r"E:\real\use_code\yoloV8\Dataset_new\images1"

    try:
        extracted_fps = extract_frames(video_path, save_dir, frame_interval=1)
    except Exception as e:
        print(f"帧提取失败：{e}")