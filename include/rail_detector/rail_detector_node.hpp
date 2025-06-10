#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem> 
#include <onnxruntime_cxx_api.h>

class RailDetectorNode : public rclcpp::Node
{
public:
    RailDetectorNode();

private:
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

    int min_area_px_{1000};
    
    image_transport::Subscriber sub_;
    image_transport::Publisher pub_preprocessed_;
    image_transport::Publisher pub_overlay_;
    image_transport::Publisher pub_mask_;

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
};
