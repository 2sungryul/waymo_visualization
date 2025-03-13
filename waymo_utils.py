import os
import tensorflow as tf
from pathlib import Path
import numpy as np
import open3d as o3d
from typing import cast
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from waymo_open_dataset.dataset_pb2 import CameraImage, CameraLabels
from waymo_open_dataset.dataset_pb2 import Frame
from waymo_open_dataset.label_pb2 import Label
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2

#segment_path = r"/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/Waymo/individual_files_training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord"
#output_dir = r"/mnt/d/Users/2sungryul/Dropbox/Work/Dataset/Waymo"
#save = False
#visu = False

Color = tuple[int, int, int]

class ColorCodes:    
    vehicle: Color = (0, 0, 255)
    pedestrian: Color = (0, 255, 0)
    cyclist: Color = (0, 255, 255)
    sign: Color = (255, 0, 0)

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


PcdList = list[np.ndarray]
PcdReturn = tuple[PcdList, PcdList]

def save_camera_images(idx: int, frame: Frame, output_dir: Path) -> None:
    for image in frame.images:
        save_camera_image(idx, image, frame.camera_labels, output_dir)


def save_data(frame: Frame, idx: int, points: np.ndarray,
              output_dir: Path) -> None:
    save_frame(frame, idx, output_dir)
    save_points(idx, points, output_dir)


def save_frame(frame: dataset_pb2.Frame, idx: int, output_dir: Path) -> None:
    name = 'frame-' + str(idx) + '.bin'
    with open((output_dir / name), 'wb') as file:
        file.write(frame.SerializeToString())


def save_points(idx: int, points: np.ndarray, output_dir: Path) -> None:
    name = 'points-' + str(idx) + '.npy'
    np.save(str(output_dir / name), points)


def pcd_from_range_image(frame: Frame) -> tuple[PcdReturn, PcdReturn]:
    def _range_image_to_pcd(ri_index: int = 0) -> PcdReturn:
        points, points_cp = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose,
            ri_index=ri_index)
        return points, points_cp
 
    parsed_frame = frame_utils.parse_range_image_and_camera_projection(frame)
    range_images, camera_projections, _, range_image_top_pose = parsed_frame
    frame.lasers.sort(key=lambda laser: laser.name)
    return _range_image_to_pcd(), _range_image_to_pcd(1)

def visualize_pcd_return(frame: Frame, pcd_return: PcdReturn, visu: bool) -> None:
    points, points_cp = pcd_return
    points_all = np.concatenate(points, axis=0)
    print(f'points_all shape: {points_all.shape}')
 
    # camera projection corresponding to each point
    points_cp_all = np.concatenate(points_cp, axis=0)
    print(f'points_cp_all shape: {points_cp_all.shape}')
 
    if visu:
        show_point_cloud_rainbow(points_all, frame.laser_labels)

Point3D = list[float]
LineSegment = tuple[int, int]
 
# order in which bbox vertices will be connected
LINE_SEGMENTS = [[0, 1], [1, 3], [3, 2], [2, 0],
                 [4, 5], [5, 7], [7, 6], [6, 4],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
 
 
def show_point_cloud_rainbow(points: np.ndarray, laser_labels: Label) -> None:
    # pylint: disable=no-member (E1101)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
 
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
 
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.6, origin=[0, 0, 0])
 
    pcd.points = o3d.utility.Vector3dVector(points)
 
    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)
 
    for label in laser_labels:
        bbox_corners = transform_bbox_waymo(label)
        bbox_points = build_open3d_bbox(bbox_corners, label)
        print(label.id)    
        colors = [[1, 0, 0] for _ in range(len(LINE_SEGMENTS))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bbox_points),
            lines=o3d.utility.Vector2iVector(LINE_SEGMENTS),
        )
 
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)
 
    # set zoom, front, up, and lookat
    vis.get_view_control().set_zoom(0.1)
    vis.get_view_control().set_front([-2, 0, 1])
    vis.get_view_control().set_up([1, 0, 1])
    vis.get_view_control().set_lookat([0, 0, 0]) 

    vis.run()

def show_point_cloud_binary(points: np.ndarray, laser_labels: Label) -> None:
    # pylint: disable=no-member (E1101)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
 
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1.0

    # set zoom, front, up, and lookat
    vis.get_view_control().set_zoom(0.1)
    vis.get_view_control().set_front([0, 0, 1])
    vis.get_view_control().set_up([1, 0, 0])
    vis.get_view_control().set_lookat([0, 0, 0])
 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
 
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.6, origin=[0, 0, 0])
 
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
 
    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)
 
    for label in laser_labels:
        bbox_corners = transform_bbox_waymo(label)
        bbox_points = build_open3d_bbox(bbox_corners, label)
        obj_color = OBJECT_COLORS[OBJ_TYPE_MAP[label.type]]
        print(obj_color) 
        #colors = [[1, 0, 0] for _ in range(len(LINE_SEGMENTS))]
        colors = [obj_color for _ in range(len(LINE_SEGMENTS))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bbox_points),
            lines=o3d.utility.Vector2iVector(LINE_SEGMENTS),
        )
 
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)
 
    # set zoom, front, up, and lookat
    vis.get_view_control().set_zoom(0.1)
    vis.get_view_control().set_front([-2, 0, 1])
    vis.get_view_control().set_up([1, 0, 1])
    vis.get_view_control().set_lookat([0, 0, 0])

    vis.run()


def transform_bbox_waymo(label: Label) -> np.ndarray:
    """Transform object's 3D bounding box using Waymo utils"""
    heading = -label.box.heading
    bbox_corners = get_bbox(label)
 
    mat = transform_utils.get_yaw_rotation(heading)
    rot_mat = mat.numpy()[:2, :2]
 
    return bbox_corners @ rot_mat
 
def get_bbox(label: Label) -> np.ndarray:
    width, length = label.box.width, label.box.length
    return np.array([[-0.5 * length, -0.5 * width],
                     [-0.5 * length, 0.5 * width],
                     [0.5 * length, -0.5 * width],
                     [0.5 * length, 0.5 * width]])

def transform_bbox_custom(label: Label) -> np.ndarray:
    """Transform object's 3D bounding box without Waymo utils"""
    heading = -label.box.heading
    bbox_corners = get_bbox(label)
    rot_mat = np.array([[np.cos(heading), - np.sin(heading)],
                        [np.sin(heading), np.cos(heading)]])
 
    return bbox_corners @ rot_mat

def build_open3d_bbox(box: np.ndarray, label: Label) -> list[Point3D]:
    """Create bounding box's points and lines needed for drawing in open3d"""
    x, y, z = label.box.center_x, label.box.center_y, label.box.center_z
 
    z_bottom = z - label.box.height / 2
    z_top = z + label.box.height / 2
 
    points = [[0., 0., 0.]] * box.shape[0] * 2
    for idx in range(box.shape[0]):
        x_, y_ = x + box[idx][0], y + box[idx][1]
        points[idx] = [x_, y_, z_bottom]
        points[idx + 4] = [x_, y_, z_top]
 
    return points

def concatenate_pcd_returns(
        pcd_return_1: PcdReturn,
        pcd_return_2: PcdReturn) -> tuple[np.ndarray, np.ndarray]:
    points, points_cp = pcd_return_1
    points_ri2, points_cp_ri2 = pcd_return_2
    points_concat = np.concatenate(points + points_ri2, axis=0)
    points_cp_concat = np.concatenate(points_cp + points_cp_ri2, axis=0)
    print(f'points_concat shape: {points_concat.shape}')
    print(f'points_cp_concat shape: {points_cp_concat.shape}')
    return points_concat, points_cp_concat


Point = tuple[int, int]
Color = tuple[float, float, float, float]
 
CAMERA_NAME = {
    0: 'unknown',
    1: 'front',
    2: 'front-left',
    3: 'front-right',
    4: 'side-left',
    5: 'side-right'
}


def save_camera_image(idx: int, camera_image: CameraImage,
                      camera_labels: CameraLabels, output_dir: Path) -> None:
    image = tf.image.decode_png(camera_image.image)
    image = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
 
    # draw the camera labels.
    for labels in camera_labels:
        if labels.name == camera_image.name:
            draw_labels(image, labels.labels)
 
    name = CAMERA_NAME[camera_image.name] + '-' + str(idx) + '.png'
    cv2.imwrite(str(output_dir / name), image)

 
def draw_labels(image: np.ndarray, labels: CameraLabels) -> None:
    def _draw_label(label_: CameraLabels) -> None:
        def _draw_line(p1: Point, p2: Point) -> None:
            cv2.line(image, p1, p2, color, 2)
 
        color = OBJECT_COLORS[label_.type]
        x1 = int(label_.box.center_x - 0.5 * label_.box.length)
        y1 = int(label_.box.center_y - 0.5 * label_.box.width)
        x2 = x1 + int(label_.box.length)
        y2 = y1 + int(label_.box.width)
 
        # draw bounding box
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
        for idx in range(len(points) - 1):
            _draw_line(points[idx], points[idx + 1])
 
    for label in labels:
        _draw_label(label)


def visualize_camera_projection(idx: int, frame: Frame, output_dir: Path,
                                pcd_return: PcdReturn) -> None:
    points, points_cp = pcd_return
    points_all = np.concatenate(points, axis=0)
    points_cp_all = np.concatenate(points_cp, axis=0)
 
    images = sorted(frame.images, key=lambda i: i.name)  # type: ignore
 
    # distance between lidar points and vehicle frame origin
    points_tensor = tf.norm(points_all, axis=-1, keepdims=True)
    points_cp_tensor = tf.constant(points_cp_all, dtype=tf.int32)
 
    mask = tf.equal(points_cp_tensor[..., 0], images[0].name)
 
    points_cp_tensor = tf.cast(tf.gather_nd(
        points_cp_tensor, tf.where(mask)), tf.float32)
    points_tensor = tf.gather_nd(points_tensor, tf.where(mask))
 
    projected_points_from_raw_data = tf.concat(
        [points_cp_tensor[..., 1:3], points_tensor], -1).numpy()
 
    plot_points_on_image(
        idx, projected_points_from_raw_data, images[0], output_dir)


def plot_points_on_image(idx: int, projected_points: np.ndarray,
                         camera_image: CameraImage, output_dir: Path) -> None:
    image = tf.image.decode_png(camera_image.image)
    image = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
 
    for point in projected_points:
        x, y = int(point[0]), int(point[1])
        rgba = rgba_func(point[2])
        r, g, b = int(rgba[2] * 255.), int(rgba[1] * 255.), int(rgba[0] * 255.)
        cv2.circle(image, (x, y), 1, (b, g, r), 2)
 
    name = 'range-image-' + str(idx) + '-' + CAMERA_NAME[
        camera_image.name] + '.png'
    cv2.imwrite(str(output_dir / name), image)

 
def rgba_func(value: float) -> Color:
    """Generates a color based on a range value"""
    return cast(Color, plt.get_cmap('jet')(value / 50.))

 
def process_segment(segment_path: str, output_dir: Path, save: bool, visu: bool) -> None:
    data_set = tf.data.TFRecordDataset(segment_path, compression_type='')
    
    for idx, data in enumerate(data_set):
        print(f'frame index: {idx}')
        break
    process_data(idx, data, output_dir, save, visu) 
    
    

def process_data(idx: int, data: tf.Tensor, output_dir: Path, save: bool, visu: bool) -> None:
    # pylint: disable=no-member (E1101)
    frame = Frame()
    frame.ParseFromString(bytearray(data.numpy()))
 
    # visualize point clouds of 1st and 2nd return
    pcd_return_1, pcd_return_2 = pcd_from_range_image(frame)
    #visualize_pcd_return(frame, pcd_return_1, visu)
    #visualize_pcd_return(frame, pcd_return_2, visu)
 
    # concatenate 1st and 2nd return
    points, _ = concatenate_pcd_returns(pcd_return_1, pcd_return_2)
 
    if visu:
        #save_camera_images(idx, frame, output_dir)
        show_point_cloud_rainbow(points, frame.laser_labels)
        #visualize_camera_projection(idx, frame, output_dir, pcd_return_1)
 
    #if save:
        #save_data(frame, idx, points, output_dir)
