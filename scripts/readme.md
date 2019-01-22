
#desc

Create labels file for COCO

+ 1.create label txts
    ```bash
        # param 1 :the coco annotations file
        # param 2 :the output labels file dir
        # param 3 :label class
        python ./createCOCOlabels.py annotations.json {labelFolder} val2014
    ```  

+ 2.create eval list
    ```bash
        # param 1 :input image src folder
        # param 2 :label files folder created by step 1
       python ./makeList.py {imageFolder} {labelFolder}
    ```