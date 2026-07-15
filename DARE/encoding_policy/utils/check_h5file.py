import h5py
from onnxruntime.transformers.models.llama.dist_settings import print_out


def check_h5_shapes(h5_path):
    print(f"\nChecking file: {h5_path}")
    print("=" * 60)

    with h5py.File(h5_path, 'r') as f:
        video_ids = list(f.keys())
        print(f"Total videos: {len(video_ids)}\n")

        for vid in sorted(video_ids, key=lambda x: int(x)):
            print(vid)
            l1_feats = f[vid]['P1']
            l2_feats = f[vid]['P2']
            l3_feats = f[vid]['P3']
            print(f"Video {vid:>3s} -> shape: {l1_feats.shape}")
            print(f"Video {vid:>3s} -> shape: {l2_feats.shape}")
            print(f"Video {vid:>3s} -> shape: {l3_feats.shape}")

    print("\n✓ Done checking.")


if __name__ == "__main__":
    h5_path = "../deg_feats/all_layer/D²-City_prompts.h5"  # 改成你的路径
    check_h5_shapes(h5_path)