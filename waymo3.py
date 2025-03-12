# visualize point cloud and 3d gtbbox

import os
import numpy as np
import tensorflow as tf
import open3d as o3d
from waymo_open_dataset.dataset_pb2 import Frame
from waymo_open_dataset.utils import frame_utils
import waymo_utils

segment_path = r"/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/Waymo/individual_files_training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"

# 데이터셋파일의 확장자가 tfrecord인지 체크 후 에러처리
if os.path.basename(segment_path).split('.')[-1] != 'tfrecord':
        raise ValueError(f'segment file has to be of ' f'{tf.data.TFRecordDataset.__name__} type')

# tfrecord파일로부터 tf.data.TFRecordDataset 객체생성(약 200개 포인트 클라우드 프레임으로 구성됨)
data_set = tf.data.TFRecordDataset(segment_path, compression_type='')

# tf.data.TFRecordDataset 객체로부터 프레임(tf.Tensor) 1개를 추출
frame_index = 100
for data in data_set.skip(frame_index).take(1):
    break

# point cloud frame 객체생성
frame = Frame() 
frame.ParseFromString(bytearray(data.numpy()))
range_images, camera_projections, _, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
frame.lasers.sort(key=lambda laser: laser.name)

# first return of lidar
points0, points_cp0 = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=0)

# second return of lidar
points1, points_cp1 = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=1)

print('frame index:', frame_index)
print('first return:',len(points0), len(points0[0]),len(points0[1]),len(points0[2]),len(points0[3]),len(points0[4]))
print('total point:',len(points0[0])+len(points0[1])+len(points0[2])+len(points0[3])+len(points0[4]))
print('second return:',len(points1), len(points1[0]),len(points1[1]),len(points1[2]),len(points1[3]),len(points1[4]))
print('total point:',len(points1[0])+len(points1[1])+len(points1[2])+len(points1[3])+len(points1[4]))

# merge first and second returns
points_concat = np.concatenate(points0 + points1, axis=0)
#points_concat = np.concatenate(points0, axis=0)
print(f'points_concat shape: {points_concat.shape}')

waymo_utils.show_point_cloud_binary(points_concat, frame.laser_labels)
#waymo_utils.show_point_cloud_rainbow(points_concat, frame.laser_labels)

