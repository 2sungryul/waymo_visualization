import cv2
import numpy as np
import glob

#path=r"/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/SemanticKITTI/data_odometry_color/dataset/sequences/00/image_2"
path=r"/mnt/d/Users/2sungryul/Dropbox/Work/open3d/waymo1"

img_array = []
#for filename in glob.glob('./waymo1/*.png'):
for filename in glob.glob(path+'/'+'*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('waymo.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()