#pragma once
#include <sys/stat.h>

#include <chrono>
#include <execution>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
// ROS
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/point_cloud_conversion.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/transform.hpp>
// extention node
#include "extension_node/extension_node.hpp"
// common_utils
#define USE_PCL
#define USE_ROS2
#include "common_utils/common_utils.hpp"
// tf2
#include <tf2/utils.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
// OpenMP
#include <omp.h>
// OpenCV
#include <opencv2/opencv.hpp>
#include "cv_bridge/cv_bridge.h"

using namespace std::chrono_literals;

class ScaleAdjuster : public ext_rclcpp::ExtensionNode
{
public:
  ScaleAdjuster(const rclcpp::NodeOptions &options) : ScaleAdjuster("", options) {}
  ScaleAdjuster(
      const std::string &name_space = "",
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
      : ext_rclcpp::ExtensionNode("scale_adjuster_node", name_space, options), tf_buffer_(this->get_clock()), listener_(tf_buffer_)
  {
    RCLCPP_INFO(this->get_logger(), "start scale_adjuster_node");
    BASE_FRAME = param<std::string>("scale_adjuster.base_frame", "base_link");
    VOXEL_SIZE = param<double>("scale_adjuster.voxel_size", 700.0);
    CUT_RANGE = param<double>("scale_adjuster.cut_range", 0.1); // 0~1 (床面から何％の点を抜き取るか(1がすべての点を使用))
    RANSAC_THRESHOLD = param<double>("scale_adjuster.ransac_threshold", 500.0);
    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "out_points", rclcpp::QoS(10));
    scale_pub_ = this->create_publisher<std_msgs::msg::Float32>("scale_adjuster/scale", rclcpp::QoS(10));
    plane_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "scale_adjuster/plane_points", rclcpp::QoS(10));
    filtered_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "scale_adjuster/filtered_points", rclcpp::QoS(10));
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "in_points", rclcpp::QoS(10),
        [&](const sensor_msgs::msg::PointCloud2::SharedPtr msg)
        {
          auto camera_link_tf = ros2_utils::lookup_transform(tf_buffer_, msg->header.frame_id, BASE_FRAME);
          double camera_h = 0.0;
          if (camera_link_tf)
          {
            auto transform = camera_link_tf.value().transform;
            // std::cout << "camera_link_tf:" << transform.translation.x << "," << transform.translation.y << "," << transform.translation.z << std::endl;
            camera_h = transform.translation.z;
          }
          cloud_header_ = msg->header;
          pcl::PointCloud<pcl::PointXYZ> cloud;
          pcl::fromROSMsg(*msg, cloud);
          cloud = pcl_utils::voxelgrid_filter(cloud, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);
          // スケール計算
          double scale = calc_scale(cloud, 1.0);
// スケール補正
#pragma omp parallel for schedule(dynamic)
          for (auto &p : cloud.points)
          {
            p.x *= scale;
            p.y *= scale;
            p.z *= scale;
          }
          cloud_pub_->publish(ros2_utils::make_ros_pointcloud2(msg->header, cloud));
          scale_pub_->publish(ros2_utils::make_float32(scale));
        });
  }

private:
  std::string BASE_FRAME;
  double CUT_RANGE;
  double VOXEL_SIZE;
  double RANSAC_THRESHOLD;
  // subscriber
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  // publisher
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr scale_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr plane_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cloud_pub_;
  // tf
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener listener_;

  std_msgs::msg::Header cloud_header_;

  template <typename POINT_TYPE = pcl::PointXYZ>
  double calc_scale(const pcl::PointCloud<POINT_TYPE> &in_cloud, double camera_h)
  {
    pcl::PointCloud<POINT_TYPE> out_cloud = in_cloud;
    // filtering points
		pcl::PointXYZ min_p, max_p;
		pcl::getMinMax3D (in_cloud, min_p, max_p);
    double diff_y = max_p.y - min_p.y;
    double cut_y = max_p.y - diff_y * CUT_RANGE;
    // filtering points
    pcl::PointCloud<POINT_TYPE> cut_cloud = pcl_utils::passthrough_filter<POINT_TYPE>("y", in_cloud, cut_y , max_p.y);
    // plane detection
    auto [inliers, coefficients] = pcl_utils::ransac<POINT_TYPE>(cut_cloud, RANSAC_THRESHOLD);
    pcl::PointCloud<POINT_TYPE> plane_cloud = pcl_utils::extract_cloud<POINT_TYPE>(cut_cloud, inliers);
    Eigen::Vector3d normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    normal.normalize();
    std::cout << "normal:" << normal.transpose() << std::endl;
    // calc inner product
    std::vector<double> h;
    for (auto &cp : plane_cloud.points)
    {
      Eigen::Vector3d p(cp.x, cp.y, cp.z);
      h.push_back(normal.transpose() * p);
    }
    // calc median
    size_t h_size = h.size();
    std::sort(h.begin(), h.end());
    size_t median_index = h_size / 2;
    double h_median = (h_size % 2 == 0
                           ? (h[median_index] + h[median_index - 1]) / 2
                           : h[median_index]);
    std::cout << "h_median:" << h_median << std::endl;
    // calc scale
    double scale = camera_h / std::abs(h_median);
    plane_cloud_pub_->publish(ros2_utils::make_ros_pointcloud2(cloud_header_, plane_cloud));
    filtered_cloud_pub_->publish(ros2_utils::make_ros_pointcloud2(cloud_header_, cut_cloud));
    return scale;
  }
};