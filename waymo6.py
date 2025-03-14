# visualize 2d bbox on front image

import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from waymo_open_dataset.dataset_pb2 import Frame
import waymo_utils

CAMERA_NAME = {
    0: 'unknown',
    1: 'front',
    2: 'front-left',
    3: 'front-right',
    4: 'side-left',
    5: 'side-right'
}

#segment_path = r"/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/Waymo/individual_files_training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"
#segment_path = r"/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/Waymo/individual_files_training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord"
#segment_path = r"/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/Waymo/individual_files_training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord"
#segment_path = r"/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/Waymo/individual_files_training_segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord"
segment_path = r"/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/Waymo/individual_files_training_segment-10072140764565668044_4060_000_4080_000_with_camera_labels.tfrecord"

# 데이터셋파일의 확장자가 tfrecord인지 체크 후 에러처리
if os.path.basename(segment_path).split('.')[-1] != 'tfrecord':
        raise ValueError(f'segment file has to be of ' f'{tf.data.TFRecordDataset.__name__} type')

# tfrecord파일로부터 tf.data.TFRecordDataset 객체생성(약 200개 포인트 클라우드 프레임으로 구성됨)
data_set = tf.data.TFRecordDataset(segment_path, compression_type='')

# tf.data.TFRecordDataset 객체로부터 프레임(tf.Tensor) 1개를 추출
frame_index = 100
#for data in data_set.skip(frame_index).take(1):
for idx, data in enumerate(data_set):
    print(f'frame index: {idx}')

    # frame 객체생성
    frame = Frame() 

    # tfrecord 객체데이터를 파싱
    frame.ParseFromString(bytearray(data.numpy()))

    # 카메라 5대의 영상을 이름으로 정렬
    images = sorted(frame.images, key=lambda i: i.name)  

    # draw 2d bbox on image of front camera
    for i, camera_image in enumerate(images):
            if CAMERA_NAME[camera_image.name] == 'front':
                image = tf.image.decode_png(camera_image.image)
                image = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
                
                print(i,CAMERA_NAME[camera_image.name],image.shape)
                
                # draw the camera labels.
                for labels in frame.camera_labels:
                    if labels.name == camera_image.name:
                        print(labels.name,camera_image.name)
                        waymo_utils.draw_labels(image, labels.labels)

                #cv2.imshow("image{}".format(i),image)
                #name = CAMERA_NAME[camera_image.name] + '-' + str(idx) + '.png'
                cv2.imwrite("img_%06d.png" % idx, image)
                cv2.imshow("image",image)
                cv2.waitKey(10)          

    

        
      
