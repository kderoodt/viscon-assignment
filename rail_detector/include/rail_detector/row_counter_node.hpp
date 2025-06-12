#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/u_int32.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.hpp>

class RowCounterNode : public rclcpp::Node
{
public:
  RowCounterNode(); 
  explicit RowCounterNode(const rclcpp::NodeOptions & options);  

private:

  double  row_spacing_;
  int     min_area_px_;
  int     min_overlap_px_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_mask_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;

  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr pub_count_;

  nav_msgs::msg::Odometry    latest_odom_;
  bool                       have_last_row_pose_{false};
  geometry_msgs::msg::Pose   last_row_pose_;
  bool                       active_rail_{false};
  cv::Rect                   active_bb_;
  uint32_t                   row_count_{0};
  
  void maskCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
  void odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& msg);

  bool findNearestRailBlob(const cv::Mat& mask, cv::Rect& out_bb) const;
  void handleRowEvent();
};