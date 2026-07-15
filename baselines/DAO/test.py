import os
SEQUENCES = sorted(os.listdir('/home/dell/lyra/Dataset/D²-City/train/1920x1080/images'))
for sequence in SEQUENCES:
    print(sequence)