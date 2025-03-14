# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# waymo dataset animation example

import numpy as np
import open3d as o3d
import threading
import time
import os

import waymo_utils
import tensorflow as tf
from waymo_open_dataset.dataset_pb2 import Frame
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import label_pb2

#segment_path = r"/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/Waymo/individual_files_training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"
segment_path = r"/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/Waymo/individual_files_training_segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord"
#segment_path = r"/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/Waymo/individual_files_training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord"
#segment_path = r"/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/Waymo/individual_files_training_segment-10061305430875486848_1080_000_1100_000_with_camera_labels.tfrecord"
#segment_path = r"/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/Waymo/individual_files_training_segment-10072140764565668044_4060_000_4080_000_with_camera_labels.tfrecord"

CLOUD_NAME = "points"
FRAME_NUM = 190

Color = tuple[int, int, int]

class ColorCodes:    
    vehicle: Color = (0, 0, 1)
    pedestrian: Color = (0, 1, 0)
    cyclist: Color = (0, 1, 1)
    sign: Color = (1, 0, 0)

OBJ_TYPE_MAP = {1:label_pb2.Label.Type.TYPE_VEHICLE,
                2:label_pb2.Label.Type.TYPE_PEDESTRIAN,
                4:label_pb2.Label.Type.TYPE_CYCLIST,
                3:label_pb2.Label.Type.TYPE_SIGN}

OBJECT_COLORS = {
    label_pb2.Label.Type.TYPE_VEHICLE: ColorCodes.vehicle,
    label_pb2.Label.Type.TYPE_PEDESTRIAN: ColorCodes.pedestrian,
    label_pb2.Label.Type.TYPE_CYCLIST: ColorCodes.cyclist,
    label_pb2.Label.Type.TYPE_SIGN: ColorCodes.sign
}

class MultiWinApp:

    def __init__(self):
        self.is_done = False
        self.cloud = None
        self.main_vis = None
        self.frame_index = 0
        self.first = False
        self.bbox_num = 0
        #self.n_snapshots = 0
        #self.snapshot_pos = None

        # 데이터셋파일의 확장자가 tfrecord인지 체크 후 에러처리
        if os.path.basename(segment_path).split('.')[-1] != 'tfrecord':
                raise ValueError(f'segment file has to be of ' f'{tf.data.TFRecordDataset.__name__} type')

        # tfrecord파일로부터 tf.data.TFRecordDataset 객체생성(약 200개 포인트 클라우드 프레임으로 구성됨)
        self.data_set = tf.data.TFRecordDataset(segment_path, compression_type='')

        self.points_concat_list = []
        self.open3d_bbox_list_list = []
        # tf.data.TFRecordDataset 객체로부터 프레임(tf.Tensor) 1개를 추출
        #for data in self.data_set.skip(self.frame_index).take(1):
        #for data in self.data_set.take(FRAME_NUM):
        for i,data in enumerate(self.data_set.take(FRAME_NUM)):
            # waymo frame 객체생성
            self.frame = Frame() 
            self.frame.ParseFromString(bytearray(data.numpy()))
            range_images, camera_projections, _, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(self.frame)
            self.frame.lasers.sort(key=lambda laser: laser.name)

            # first return of lidar
            points0, points_cp0 = frame_utils.convert_range_image_to_point_cloud(self.frame, range_images, camera_projections, range_image_top_pose, ri_index=0)

            # second return of lidar
            points1, points_cp1 = frame_utils.convert_range_image_to_point_cloud(self.frame, range_images, camera_projections, range_image_top_pose, ri_index=1)

            #print('frame index:', frame_index)
            #print('first return:',len(points0), len(points0[0]),len(points0[1]),len(points0[2]),len(points0[3]),len(points0[4]))
            #print('total point:',len(points0[0])+len(points0[1])+len(points0[2])+len(points0[3])+len(points0[4]))
            #print('second return:',len(points1), len(points1[0]),len(points1[1]),len(points1[2]),len(points1[3]),len(points1[4]))
            #print('total point:',len(points1[0])+len(points1[1])+len(points1[2])+len(points1[3])+len(points1[4]))

            # merge first and second returns
            points_concat = np.concatenate(points0 + points1, axis=0)
            #points_concat = np.concatenate(points0, axis=0)
            print(i,f'points_concat shape: {points_concat.shape}')
            self.points_concat_list.append(points_concat)

            open3d_bbox_list = []
            for label in self.frame.laser_labels:
                bbox_corners = waymo_utils.transform_bbox_waymo(label)
                bbox_points = waymo_utils.build_open3d_bbox(bbox_corners, label)

                obj_color = OBJECT_COLORS[OBJ_TYPE_MAP[label.type]]
                #print(obj_color) 
                colors = [obj_color for _ in range(len(waymo_utils.LINE_SEGMENTS))]
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(bbox_points),
                    lines=o3d.utility.Vector2iVector(waymo_utils.LINE_SEGMENTS),
                )
                line_set.colors = o3d.utility.Vector3dVector(colors)
                #self.main_vis.add_geometry(line_set)
                open3d_bbox_list.append(line_set)

            self.open3d_bbox_list_list.append(open3d_bbox_list)

            #for i in range(self.bbox_num):
            #    self.main_vis.add_geometry(f"bbox{i}", open3d_bbox_list[i])

        print(len(self.points_concat_list))    
        print(len(self.open3d_bbox_list_list))
        

    def run(self):
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        self.main_vis = o3d.visualization.O3DVisualizer("WAYMO", 1280, 720)
        #self.main_vis = o3d.visualization.O3DVisualizer("WAYMO")
        self.main_vis.reset_camera_to_default()
        self.main_vis.setup_camera(60, [0, 0, 0], [-15, 0, 10], [5, 0, 10]) # center, eye, up
        
        self.main_vis.set_background(np.array([0, 0, 0, 0]), None)
        self.main_vis.show_skybox(False)
        self.main_vis.point_size = 1
        self.main_vis.show_settings = True
                
        self.main_vis.set_on_close(self.on_main_window_closing)
        app.add_window(self.main_vis)
        threading.Thread(target=self.update_thread).start()

        #self.main_vis.add_action("Take snapshot in new window", self.on_snapshot)
        #self.snapshot_pos = (self.main_vis.os_frame.x, self.main_vis.os_frame.y)

        app.run()
    
    
    def on_main_window_closing(self):
        self.is_done = True
        return True  # False would cancel the close

    def update_thread(self):
        # This is NOT the UI thread, need to call post_to_main_thread() to update
        # the scene or any part of the UI.        
           
        # Initialize point cloud geometry
        point_cloud = o3d.geometry.PointCloud()
           
        while not self.is_done:
                       
            time.sleep(2)
                   
            def update_cloud():
                print("frame_index:",self.frame_index,self.points_concat_list[self.frame_index].shape,len(self.open3d_bbox_list_list[self.frame_index]))
                if self.first:
                    self.main_vis.remove_geometry("axis")
                    self.main_vis.remove_geometry("pc")
                    for i in range(self.bbox_num):
                        self.main_vis.remove_geometry(f"bbox{i}")
                                
                axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.6, origin=[0, 0, 0])
                self.main_vis.add_geometry("axis", axis_pcd)

                # Update point cloud with new data
                point_cloud.points = o3d.utility.Vector3dVector(self.points_concat_list[self.frame_index])
                point_cloud.colors = o3d.utility.Vector3dVector(np.ones((self.points_concat_list[self.frame_index].shape[0], 3)))
                self.main_vis.add_geometry("pc", point_cloud)
                
                self.bbox_num = len(self.open3d_bbox_list_list[self.frame_index])
                for i in range(self.bbox_num):
                    self.main_vis.add_geometry(f"bbox{i}", self.open3d_bbox_list_list[self.frame_index][i])
                
                # save screen image to jpg                
                self.main_vis.export_current_image("pc_%06d.png" % self.frame_index)
                
                # Move to the next frame
                self.frame_index = (self.frame_index + 1) % FRAME_NUM
                self.first = True
                
            o3d.visualization.gui.Application.instance.post_to_main_thread(self.main_vis, update_cloud)            

            if self.is_done:  # might have changed while sleeping
                break


def main():
    MultiWinApp().run()

if __name__ == "__main__":
    main()