# visualize 1 frame extracted from waymo_open_dataset

import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from waymo_open_dataset.dataset_pb2 import Frame

CAMERA_NAME = {
    0: 'unknown',
    1: 'front',
    2: 'front-left',
    3: 'front-right',
    4: 'side-left',
    5: 'side-right'
}

segment_path = r"/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/Waymo/individual_files_training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"

# 데이터셋파일의 확장자가 tfrecord인지 체크 후 에러처리
if os.path.basename(segment_path).split('.')[-1] != 'tfrecord':
        raise ValueError(f'segment file has to be of ' f'{tf.data.TFRecordDataset.__name__} type')

# tfrecord파일로부터 tf.data.TFRecordDataset 객체생성(약 200개 포인트 클라우드 프레임으로 구성됨)
data_set = tf.data.TFRecordDataset(segment_path, compression_type='')

# tf.data.TFRecordDataset 객체로부터 프레임(tf.Tensor) 1개를 추출
frame_index = 0
for data in data_set.skip(frame_index).take(1):
    break

# frame 객체생성
frame = Frame() 

# tfrecord 객체데이터를 파싱
frame.ParseFromString(bytearray(data.numpy()))

# 카메라 5대의 영상을 이름으로 정렬
images = sorted(frame.images, key=lambda i: i.name)  

for i, camera_image in enumerate(images):
        image = tf.image.decode_png(camera_image.image)
        image = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
        print(i,CAMERA_NAME[camera_image.name],image.shape)
        
        # frame_index 번째 프레임의 5대 카메라 영상을 파일로 저장
        cv2.imwrite("frame{}-{}.png".format(frame_index,CAMERA_NAME[camera_image.name]), image)
        image = cv2.imread("frame{}-{}.png".format(frame_index,CAMERA_NAME[camera_image.name]))
        #plt.imshow(image)
        cv2.imshow("image{}".format(i),image)

cv2.waitKey()  
        
      
