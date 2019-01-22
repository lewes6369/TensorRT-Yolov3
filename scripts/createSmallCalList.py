import os, sys
import random
import numpy as np

cur_dir = os.path.dirname(os.path.realpath(__file__))
#calibration length
CREATE_SMALL_LENGTH = 200

def make_calibration(fileList):
    file = open(fileList)
    s = []
    for line in file:
        pic_name = line.strip('\n')
        pic_name = pic_name.split(' ')[0]
        s.append(pic_name)
    
    random.shuffle(s)        
    shuffled = s[:CREATE_SMALL_LENGTH]

    str = ""
    for pic_name in shuffled:
        str = str+ pic_name + "\n"

    with open('calibration.txt', 'w') as f:
        f.write(str)

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("Usage:  ./createSmallCalList.py list")
        exit(1)

    try:
        make_calibration(sys.argv[1])

    except Exception as e:
        print(e.message)
        exit(1)
    exit(0)