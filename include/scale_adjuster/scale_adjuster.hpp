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
// PCL
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace std::chrono_literals;

class ScaleAdjuster : public rclcpp::Node
{
public:
  ScaleAdjuster(const rclcpp::NodeOptions &options) : ScaleAdjuster("", options) {}
  ScaleAdjuster(
      const std::string &name_space = "",
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
      : rclcpp::Node("scale_adjuster_node", name_space, options), tf_buffer_(this->get_clock()), listener_(tf_buffer_)
  {
    RCLCPP_INFO(this->get_logger(), "start scale_adjuster_node");
    BASE_FRAME = param<std::string>("scale_adjuster.base_frame", "base_link");
    VOXEL_SIZE = param<double>("scale_adjuster.voxel_size", 700.0);
    H_RANGE = param<double>("scale_adjuster.h_range", 1000.0);
    RANSAC_THRESHOLD = param<double>("scale_adjuster.ransac_threshold", 300.0);
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
          auto camera_link_tf = lookup_transform(tf_buffer_, msg->header.frame_id, BASE_FRAME);
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
          cloud = voxelgrid_filter(cloud, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);
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
          cloud_pub_->publish(make_ros_pointcloud2(msg->header, cloud));
          scale_pub_->publish(make_float32(scale));
        });
  }

private:
  std::string BASE_FRAME;
  double H_RANGE;
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
    pcl::PointCloud<POINT_TYPE> cut_cloud = passthrough_filter<POINT_TYPE>("y", in_cloud, -(camera_h + H_RANGE), 1000000.0);
    // plane detection
    auto [inliers, coefficients] = ransac<POINT_TYPE>(cut_cloud, RANSAC_THRESHOLD);
    pcl::PointCloud<POINT_TYPE> plane_cloud = extract_cloud<POINT_TYPE>(cut_cloud, inliers);
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
    plane_cloud_pub_->publish(make_ros_pointcloud2(cloud_header_, plane_cloud));
    filtered_cloud_pub_->publish(make_ros_pointcloud2(cloud_header_, cut_cloud));
    return scale;
  }

  // utility functions
  template <class T>
  T param(const std::string &name, const T &def)
  {
    T value;
    declare_parameter(name, def);
    get_parameter(name, value);
    return value;
  }

  template <typename FLOATING_TYPE = double>
  inline std_msgs::msg::Float32 make_float32(const FLOATING_TYPE &val)
  {
    std_msgs::msg::Float32 msg;
    msg.data = val;
    return msg;
  }

  inline std::optional<geometry_msgs::msg::TransformStamped> lookup_transform(const tf2_ros::Buffer &buffer, const std::string &source_frame,
                                                                              const std::string &target_frame,
                                                                              const rclcpp::Time &time = rclcpp::Time(0),
                                                                              const tf2::Duration timeout = tf2::durationFromSec(0.0))
  {
    std::optional<geometry_msgs::msg::TransformStamped> ret = std::nullopt;
    try
    {
      ret = buffer.lookupTransform(target_frame, source_frame, time, timeout);
    }
    catch (tf2::LookupException &ex)
    {
      std::cerr << "[ERROR]" << ex.what() << std::endl;
    }
    catch (tf2::ExtrapolationException &ex)
    {
      std::cerr << "[ERROR]" << ex.what() << std::endl;
    }
    return ret;
  }

  template <typename POINT_TYPE = pcl::PointXYZ>
  inline pcl::PointCloud<POINT_TYPE> voxelgrid_filter(const pcl::PointCloud<POINT_TYPE> &input_cloud, double lx, double ly, double lz)
  {
    pcl::PointCloud<POINT_TYPE> output_cloud;
    pcl::ApproximateVoxelGrid<POINT_TYPE> sor;
    sor.setInputCloud(input_cloud.makeShared());
    sor.setLeafSize(lx, ly, lz);
    sor.filter(output_cloud);
    return output_cloud;
  }

  template <typename POINT_TYPE = pcl::PointXYZ>
  inline pcl::PointCloud<POINT_TYPE> passthrough_filter(std::string field, const pcl::PointCloud<POINT_TYPE> &input_cloud, double min,
                                                        double max)
  {
    pcl::PointCloud<POINT_TYPE> output_cloud;
    pcl::PassThrough<POINT_TYPE> pass;
    pass.setFilterFieldName(field);
    pass.setFilterLimits(min, max);
    pass.setInputCloud(input_cloud.makeShared()); // Set cloud
    pass.filter(output_cloud);                    // Apply the filter
    return output_cloud;
  }

  template <typename POINT_TYPE = pcl::PointXYZ>
  inline std::pair<pcl::PointIndices::Ptr, pcl::ModelCoefficients::Ptr> ransac(const pcl::PointCloud<POINT_TYPE> &cloud, double threshold = 0.5)
  {
    // 平面検出
    // 平面方程式と平面と検出された点のインデックス
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // RANSACによる検出．
    pcl::SACSegmentation<POINT_TYPE> seg;
    seg.setOptimizeCoefficients(true);     // 外れ値の存在を前提とし最適化を行う
    seg.setModelType(pcl::SACMODEL_PLANE); // モードを平面検出に設定
    seg.setMethodType(pcl::SAC_RANSAC);    // 検出方法をRANSACに設定
    seg.setDistanceThreshold(threshold);   // しきい値を設定
    seg.setInputCloud(cloud.makeShared()); // 入力点群をセット
    seg.segment(*inliers, *coefficients);  // 検出を行う
    return {inliers, coefficients};
  }

  template <typename POINT_TYPE = pcl::PointXYZ>
  inline pcl::PointCloud<POINT_TYPE> extract_cloud(const pcl::PointCloud<POINT_TYPE> &cloud, pcl::PointIndices::Ptr inliers,
                                                   bool negative = false)
  {
    pcl::PointCloud<POINT_TYPE> extrac_cloud;
    pcl::ExtractIndices<POINT_TYPE> extract;
    if (inliers->indices.size() == 0)
      return extrac_cloud;
    extract.setInputCloud(cloud.makeShared());
    extract.setIndices(inliers);
    extract.setNegative(negative);
    extract.filter(extrac_cloud);
    return extrac_cloud;
  }

  template <typename POINT_TYPE = pcl::PointXYZ>
  inline sensor_msgs::msg::PointCloud2 make_ros_pointcloud2(std_msgs::msg::Header header, const pcl::PointCloud<POINT_TYPE> &cloud)
  {
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(cloud, cloud_msg);
    cloud_msg.header = header;
    return cloud_msg;
  }
};