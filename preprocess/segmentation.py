
import cv2
import random
import numpy as np
import os
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

imgs = os.listdir('../source/DatasetA_20180813/test')

for img in imgs:
    try:
        src = cv2.imread(os.path.join('../source/DatasetA_20180813/test',img))
        img_area = src.shape[0] * src.shape[1]
        segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.95, k=300, min_size=50)
        segment = segmentator.processImage(src)
        clear_border(segment)
        seg_image = np.zeros(src.shape, np.uint8)

        for region in regionprops(segment):
            color = [random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)]
            points = region.coords
            for point in points:
                seg_image[point[0],point[1]] = color
                
        result = cv2.addWeighted(src, 0.3, seg_image, 0.7, 0)
        cv2.namedWindow('image',0)
        cv2.resizeWindow('image', 500,500)

        cv2.imshow('image',result)
        cv2.imshow('origin',src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass
