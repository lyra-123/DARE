import pandas as pd
import matplotlib.pyplot as plt

h5_files = [
    '/home/dell/lyra/H/h5_file/sample_cpa/desc/DSEC_desc.h5',
    '/home/dell/lyra/H/h5_file/sample_cpa/desc/LMOT_desc.h5',
]

colors = ['red', 'blue', 'green', 'orange', 'purple']

plt.figure(figsize=(12,6))

for i, file in enumerate(h5_files):

    df = pd.read_hdf(file, 'encoding_data')
    df = df.reset_index()
    data = df[
        (df['SEQ'] < 2) &
        (df['QP'] == 0) &
        (df['SKIP'] == 0) &
        (df['RE'] == 0)
        ][['SEQ', 'CHUNK', 'Size']].copy()

    x = range(len(data))
    y = data['Size']

    plt.scatter(x, y,
                color=colors[i],
                alpha=0.6,
                label=f'File {i+1}')

plt.xlabel("Block Index")
plt.ylabel("Size")
plt.title("Size Scatter Comparison (First 6 SEQ)")
plt.legend()

plt.savefig('fig/multi_h5_size_scatter.png',
            dpi=300,
            bbox_inches='tight')

plt.close()

print("图像已保存")