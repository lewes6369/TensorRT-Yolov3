import os, sys
import numpy as np
from pycocotools.coco import COCO

def createCatDict(cats):
    '''
        coco categories contains 90, but skip some nums
        so make a dict to remap cats.
    '''
    name_dict = {}
    i = 0
    with open("coco.names",'r') as f:
        lines = f.readlines()
        for name in lines:
            name = name.rstrip()
            name_dict[name]= i
            i = i + 1

    id_dicts = {}
    for cat in cats:
        if cat['name'] in name_dict:
            id_dicts[cat['id']] = name_dict[cat['name']]
        else:
            print("can not find name in name_dict %s"%cat['name'])
    
    print("id_dicts: {}".format(id_dicts))
    return id_dicts


def makeLabels(annFile,label_folder,data_type):
    # initialize COCO api for instance annotations
    coco=COCO(annFile)

    catIds = coco.getCatIds()
    cats = coco.loadCats(catIds)
    id_dicts= createCatDict(cats)

    imgIds = coco.getImgIds()
    for id in imgIds:
        annIds = coco.getAnnIds(imgIds= id)
        if len(annIds) == 0:
            print("can not find anns of image %d, continue"% id)
            continue

        anns = coco.loadAnns(annIds)
        lines = ["{} {}".format( id_dicts[ an["category_id"] ], repr(an["bbox"]).replace(' ','')) for an in anns]
        lines = '\n'.join(lines)
        txt_name = "COCO_{}_{:0>12d}.txt".format(data_type, id) 
        txt_path = os.path.join(label_folder,txt_name)
        with open(txt_path, 'w') as f:
            f.write(lines)

if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage:  ./createCOCOlabels.py annotations.json output_folder [(train|val)(2014|2017)](:val2014)")
        exit(1)

    try:
        if len(sys.argv) < 4:
            makeLabels(sys.argv[1],sys.argv[2],"val2014")
        else:
            makeLabels(sys.argv[1],sys.argv[2],sys.argv[3])

    except Exception as e:
        print(e.message)
        exit(1)
    exit(0) 