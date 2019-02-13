import os, sys
import numpy as np

def getImgFileNameList(folder):
    files= os.listdir(folder)
    lst = []
    for file in files:
        if not os.path.isdir(file) and file[-4:] == ".jpg" :
           lst.append(file)
    return lst

def makeList(image_folder,label_folder):
    lst = getImgFileNameList(image_folder)
    
    str = ""
    for filename in lst:
        image_filepath = os.path.join(image_folder,filename)

        label_filename = filename.replace(".jpg",".txt")
        label_filepath = os.path.join(label_folder,label_filename)

        #print(label_filename)
        if not os.path.exists(label_filepath):
            print("notice : can not find label file to the image %s"%image_filepath)
        else :
            with open(label_filepath) as label_file:
                label_data = []
                for line_label in label_file:
                    line_label = line_label.strip(' \n')
                    label_data.append(line_label)

                line = "{} {}\n".format(image_filepath," ".join(label_data))
                str = str + line

    with open('labels.txt', 'w') as f:
        print(str, file=f)

if __name__ == '__main__':
    if len(sys.argv) < 1 or len(sys.argv) > 3:
        print("Usage:  ./makelabels.py imageFolder labelFolder")
        exit(1)

    try:
        makeList(sys.argv[1],sys.argv[2])

    except Exception as e:
        print(e.message)
        exit(1)
    exit(0)