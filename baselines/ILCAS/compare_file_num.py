import os

ROOT_DIR = '/home/dell/lyra/ILCAS/mv_chunks/D²-City'
SEQ = sorted(os.listdir(ROOT_DIR))
filenum = 0

for seq in SEQ:
    seq_path = os.path.join(ROOT_DIR, seq)
    seq_num = len(os.listdir(seq_path))
    filenum += seq_num

print(filenum)