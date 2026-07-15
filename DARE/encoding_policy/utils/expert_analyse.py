import pandas as pd

SEQ_START = 0
SEQ_END = 5
df = pd.read_hdf('/home/dell/lyra/H/h5_file/sample_cpa/desc/DSEC_desc.h5', 'encoding_data')
df = df.reset_index()
df = df[(df['SEQ'] >= SEQ_START) & (df['SEQ'] <= SEQ_END)]

df_sorted = df.sort_values(
    ['SEQ', 'CHUNK', 'Accuracy', 'Size'],
    ascending=[True, True, False, True]
)
expert_df = df_sorted.groupby(['SEQ', 'CHUNK']).head(1).copy()
expert_df['SEQ_CHUNK'] = expert_df.apply(
    lambda x: f"{int(x['SEQ']):03d}_{int(x['CHUNK']):03d}",
    axis=1
)
result = (
    expert_df.groupby(['QP', 'SKIP', 'RE'])
    .agg(
        Counts=('SEQ_CHUNK', 'count'),
        Blocks=('SEQ_CHUNK', lambda x: ' '.join(x))
    )
    .reset_index()
)
output_csv = 'ep_decision/expert_decision_stats.csv'
result.to_csv(output_csv, index=False)
print(f"统计完成，结果保存到: {output_csv}")