#from math import frexp
#from traceback import print_tb
#from torch import imag
from yolov5 import YOLOv5
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import yaml
from yolov5_ros2.cv_tool import px2xy
import os

# Get the ROS distribution version and set the shared directory for YoloV5 configuration files.
ros_distribution = os.environ.get("ROS_DISTRO")
#调用 ROS 2 的 ament_index_python 包中的 get_package_share_directory 函数，
# 查找并返回名为 yolov5_ros2 这个 ROS 2 包的 share 目录的绝对路径
# 这个路径通常用于读取包内的配置文件、模型文件等资源

#home/xin/ros2_ws/install/yolov5_ros2/share/yolov5_ros2，
#那么 package_share_directory 的值就是
package_share_directory = get_package_share_directory('yolov5_ros2')

# Create a ROS 2 Node class YoloV5Ros2.
class YoloV5Ros2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')
        self.get_logger().info(f"Current ROS 2 distribution: {ros_distribution}")

        # Declare ROS parameters. 声明 ROS 参数。
        self.declare_parameter("device", "cuda", ParameterDescriptor(
            name="device", description="Compute device selection, default: cpu, options: cuda:0"))

        self.declare_parameter("model", "yolov5s", ParameterDescriptor(
            name="model", description="Default model selection: yolov5s"))

        self.declare_parameter("image_topic", "/image_raw", ParameterDescriptor(
            name="image_topic", description="Image topic, default: /image_raw"))
        
        self.declare_parameter("camera_info_topic", "/camera/camera_info", ParameterDescriptor(
            name="camera_info_topic", description="Camera information topic, default: /camera/camera_info"))

        # Read parameters from the camera_info topic if available, otherwise, use the file-defined parameters.
        self.declare_parameter("camera_info_file", f"{package_share_directory}/config/camera_info.yaml", ParameterDescriptor(
            name="camera_info", description=f"Camera information file path, default: {package_share_directory}/config/camera_info.yaml"))

        # Default to displaying detection results.
        self.declare_parameter("show_result", False, ParameterDescriptor(
            name="show_result", description="Whether to display detection results, default: False"))

        # Default to publishing detection result images.
        self.declare_parameter("pub_result_img", False, ParameterDescriptor(
            name="pub_result_img", description="Whether to publish detection result images, default: False"))

        # 1. Load the model. 加载yolov5s.pt模型。
        model_path = package_share_directory + "/config/" + self.get_parameter('model').value + ".pt"
        device = self.get_parameter('device').value
        self.yolov5 = YOLOv5(model_path=model_path, device=device)

        # 2. Create publishers.
        #发布检测结果的消息。
        #创建一个发布器，发布检测结果的消息。
        self.yolo_result_pub = self.create_publisher(
            Detection2DArray, "yolo_result", 10)
        #创建一个 Detection2DArray 消息对象，用于存储检测结果
        self.result_msg = Detection2DArray()
        
        #发布检测后的图像
        self.result_img_pub = self.create_publisher(Image, "result_img", 10)

        # 3. Create an image subscriber (subscribe to depth information for 3D cameras, load camera info for 2D cameras).
        #订阅相机的图像话题。
        image_topic = self.get_parameter('image_topic').value
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10)
        #相机信息话题
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 1)

        # Get camera information.
        with open(self.get_parameter('camera_info_file').value) as f:
            self.camera_info = yaml.full_load(f.read())
            self.get_logger().info(f"default_camera_info: {self.camera_info['k']} \n {self.camera_info['d']}")

        # 4. Image format conversion (using cv_bridge).
        self.bridge = CvBridge()

        self.show_result = self.get_parameter('show_result').value
        self.pub_result_img = self.get_parameter('pub_result_img').value

    def camera_info_callback(self, msg: CameraInfo):
        """
        Get camera parameters through a callback function.
        """
        self.camera_info['k'] = msg.k
        self.camera_info['p'] = msg.p
        self.camera_info['d'] = msg.d
        self.camera_info['r'] = msg.r
        self.camera_info['roi'] = msg.roi
        #获取相机信息后，销毁订阅器。
        self.camera_info_sub.destroy()

    def image_callback(self, msg: Image):
        # 5. Detect and publish results.
        image = self.bridge.imgmsg_to_cv2(msg)
        detect_result = self.yolov5.predict(image)
        self.get_logger().info(str(detect_result))

        #先清空result_msg中的检测结果。
        self.result_msg.detections.clear()
        self.result_msg.header.frame_id = "camera"
        self.result_msg.header.stamp = self.get_clock().now().to_msg()

        # Parse the results.
        predictions = detect_result.pred[0]#获取可信度最佳的预测结果
        #在 YOLOv5 的输出中，通常每一行的格式为 [x1, y1, x2, y2, conf, cls]，
        #所以 predictions[:, :4] 得到的就是所有检测框的 [x1, y1, x2, y2] 坐标（像素值）
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4] #conf
        categories = predictions[:, 5] # cls 可以识别多个种类
        self.get_logger().info(f"categories: {categories} \n categories lenth:{len(categories)}")
        for index in range(len(categories)):
            name = detect_result.names[int(categories[index])]
            detection2d = Detection2D()
            detection2d.id = name
            x1, y1, x2, y2= boxes[index]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            center_x = (x1+x2)/2.0
            center_y = (y1+y2)/2.0

            if ros_distribution=='galactic':
                detection2d.bbox.center.x = center_x
                detection2d.bbox.center.y = center_y
            else:
                detection2d.bbox.center.position.x = center_x
                detection2d.bbox.center.position.y = center_y

            detection2d.bbox.size_x = float(x2-x1)
            detection2d.bbox.size_y = float(y2-y1)

            obj_pose = ObjectHypothesisWithPose()#姿态物体识别假设With位置
            obj_pose.hypothesis.class_id = name
            obj_pose.hypothesis.score = float(scores[index])

            # px2xy
            world_x, world_y = px2xy(
                [center_x, center_y], self.camera_info["k"], self.camera_info["d"], 1)
            obj_pose.pose.pose.position.x = world_x
            obj_pose.pose.pose.position.y = world_y
            detection2d.results.append(obj_pose)#因为detection2d.results结构是一个数组,所以需要用append方法添加元素。
            self.get_logger().info(f"detection2d.results lenth: {len(detection2d.results)} \n")
            self.result_msg.detections.append(detection2d)
            self.get_logger().info(f"self.result_msg.detections lenth: {len(self.result_msg.detections)} \n")
            

            # Draw results.
            if self.show_result or self.pub_result_img:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{name}({world_x:.2f},{world_y:.2f})", (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.waitKey(1)

        # Display results if needed.
        if self.show_result:
            cv2.imshow('result', image)
            cv2.waitKey(1)

        # Publish result images if needed.
        if self.pub_result_img:
            result_img_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            result_img_msg.header = msg.header
            self.result_img_pub.publish(result_img_msg)

        if len(categories) > 0:
            self.yolo_result_pub.publish(self.result_msg)

def main():
    rclpy.init()
    rclpy.spin(YoloV5Ros2())
    rclpy.shutdown()

if __name__ == "__main__":
    main()
