import os
import subprocess
import shutil
import time

FRAMES_BLUR = r'D:\Pycharmproject\ILCAS\DETRAC\images'
output_dir = r"D:\Pycharmproject\ILCAS\DETRAC\mvs"
extract_mvs_exe = r"D:\Pycharmproject\ILCAS\utils\get_mvs_data.exe"
NAME = 'DETRAC'

os.makedirs(output_dir, exist_ok=True)

FPS = 25
Length = 1  # seconds per chunk
FRAMES_PER_CHUNK = int(FPS * Length)  # = 50

# ILCAS
QP = [21, 25, 29, 33, 37, 41]

# DETRAC
RE = [[960, 540], [854, 480], [640, 360], [426, 240]]
# D²-City_1080p
# RE = [[1920, 1080], [1280, 720], [960, 540], [720, 480], [320, 240]]
# LMOT
# RE = [[1800, 1000], [1296, 720], [1080, 600], [864, 480]]
# D²-City_720p
# RE = [[1280, 720], [960, 540], [854, 480], [426, 240]]
# DSEC
# RE = [[1440, 1080], [1080, 810], [960, 720], [720, 540], [480, 360]]

# ILCAS
SKIP = [0, 1, 2, 5, 11]
# ILCAS
# SKIP = [0, 1, 2, 4, 9]

def get_sequence_lengths(video_sequences):
    """
    获取每个视频序列的文件数量，文件数量代表每个序列的长度。

    Arguments:
    - video_sequences_path: 存储视频序列文件夹的路径

    Returns:
    - sequence_lengths: 每个视频序列的长度（即文件数量列表）
    """
    sequence_lengths = []

    # 遍历每个视频序列文件夹，获取文件数量
    for sequence_folder in video_sequences:
        sequence_folder_path = os.path.join(FRAMES_BLUR, sequence_folder)

        if os.path.isdir(sequence_folder_path):
            # 统计当前文件夹下的文件数量
            num_files = len(
                [f for f in os.listdir(sequence_folder_path) if os.path.isfile(os.path.join(sequence_folder_path, f))])
            sequence_lengths.append(num_files)

    return sequence_lengths

def create_temp_dir(bs_path,sequence, chunk_no):
    chunk_start = chunk_no * FRAMES_PER_CHUNK
    chunk_end = chunk_start + FRAMES_PER_CHUNK
    print("check start ",chunk_start, chunk_end)

    # 创建临时帧目录
    blur_chunk = os.path.join(r'D:\Pycharmproject\ILCAS\input', f'{NAME}', f'{sequence:03d}_{chunk_no:03d}')
    os.makedirs(blur_chunk, exist_ok=True)
    for i in range(chunk_start, chunk_end):
        shutil.copy(os.path.join(bs_path, f"{i:04d}.jpg"),
                    os.path.join(blur_chunk, f"{i + 1 - chunk_start:02d}.jpg"))

def encoder(image_folder, video_name, w, h, fps, skip, qp):
    directory = os.path.dirname(video_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = fps / (skip + 1)
    ffmpeg_command = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-start_number', '0',
        '-i', os.path.join(image_folder, '%02d.jpg'),
        '-vf', f"select='not(mod(n,{skip + 1}))',setpts=N/({f}*TB),scale={w}:{h}",
        '-frames:v', str(FRAMES_PER_CHUNK // (skip + 1)),
        '-r', str(f),
        '-c:v', 'libx264',
        '-x264-params', f'qp={qp}',
        video_name
    ]
    start_time = time.time()
    subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ct = time.time() - start_time
    return ct

if __name__ == '__main__':
    SEQUENCES = sorted(os.listdir(FRAMES_BLUR))
    # 初始化 HDF5 表格结构
    seq_lengths = get_sequence_lengths(SEQUENCES)
    # print(seq_lengths)
    # 动态计算 CHUNK 数量
    chunks = [length // FRAMES_PER_CHUNK for length in seq_lengths]
    print("check chunks ",chunks)

    # 编码后视频输出地址
    video_path = os.path.join(r'D:\Pycharmproject\ILCAS\dataset', f'video_{NAME}')

    seq_id = 0
    for seq in SEQUENCES:
        blur_seq_path = os.path.join(FRAMES_BLUR, seq)
        for chunk_id in range(chunks[seq_id]):
            chunk_path = os.path.join(output_dir, seq, f'{chunk_id:03d}')
            os.makedirs(chunk_path, exist_ok=True)
            create_temp_dir(blur_seq_path, seq_id, chunk_id)
            config_id = 0
            for j in range(6):  # QP
                for m in range(5):  # SKIP
                    for n in range(4):  # RE
                        output_video = os.path.join(video_path, f'{seq_id:03d}_{chunk_id:03d}.mp4')
                        coding_time = encoder(os.path.join(r'D:\Pycharmproject\ILCAS\input', f'{NAME}', f'{seq_id:03d}_{chunk_id:03d}'), output_video, RE[n][0], RE[n][1], FPS, SKIP[m],QP[j])
                        output_path = os.path.join(chunk_path, f'{config_id:03d}.csv')
                        print(f"Processing {output_video} → {output_path}")
                        # 让 Windows shell 自己重定向 >
                        cmd = f'{extract_mvs_exe} {output_video} > {output_path}'
                        subprocess.run(cmd, shell=True)
                        config_id += 1
        seq_id += 1
