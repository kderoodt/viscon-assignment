#include "rail_detector/rail_detector_node.hpp"
#include <onnxruntime_cxx_api.h>

using std::placeholders::_1;

RailDetectorNode::RailDetectorNode()
: Node("rail_detector"),
  env_(ORT_LOGGING_LEVEL_WARNING, "rail_detector"),
  session_options_(),
  session_(nullptr)
{
  min_area_px_ = this->declare_parameter<int>("min_area", 500);
  bool use_gpu  = this->declare_parameter<bool>("use_gpu", true);

  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  // OrtCUDAProviderOptions cuda_opts{};
  // cuda_opts.device_id                 = 0;
  // cuda_opts.gpu_mem_limit             = SIZE_MAX;
  // cuda_opts.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchExhaustive;
  // cuda_opts.do_copy_in_default_stream = 1;
  // session_options_.AppendExecutionProvider_CUDA(cuda_opts);

    if (use_gpu) {                     
    OrtCUDAProviderOptions cuda_opts{};
    cuda_opts.device_id                 = 0;
    cuda_opts.gpu_mem_limit             = SIZE_MAX;
    cuda_opts.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchExhaustive;
    cuda_opts.do_copy_in_default_stream = 1;
    session_options_.AppendExecutionProvider_CUDA(cuda_opts);
  }

  std::string model_path =
      declare_parameter<std::string>("model_path",
                                     "models/rail_detector.onnx");

  if (model_path.empty() || model_path.front() != '/') {
    const std::filesystem::path pkg_share =
        ament_index_cpp::get_package_share_directory("rail_detector");
    model_path = (pkg_share / model_path).string();
  }

  session_ = Ort::Session(env_, model_path.c_str(), session_options_);

  auto node_ptr = std::shared_ptr<rclcpp::Node>(this, [](rclcpp::Node*){});
  image_transport::ImageTransport it(node_ptr);

  sub_ = it.subscribe("/d456_pole/infra1/image_rect_raw", 1,
                      std::bind(&RailDetectorNode::imageCallback, this, _1));
  pub_preprocessed_ = it.advertise("~/preprocessed", 1);
  pub_overlay_      = it.advertise("~/overlay",      1);
  pub_mask_         = it.advertise("/rail_mask",     1);
}

void RailDetectorNode::imageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
  try {

    cv_bridge::CvImageConstPtr cv_ptr =
        cv_bridge::toCvShare(msg, "mono8");

    cv::Mat rot;
    cv::rotate(cv_ptr->image, rot, cv::ROTATE_90_CLOCKWISE);

    const int crop_y = static_cast<int>(0.4 * rot.rows);
    cv::Mat crop = rot(cv::Range(crop_y, rot.rows), cv::Range::all());

    cv::Mat eq;
    cv::createCLAHE(4.0, {8,8})->apply(crop, eq);

    cv::Mat blur;
    cv::GaussianBlur(eq, blur, cv::Size(5,5), 0);

    pub_preprocessed_.publish(
        cv_bridge::CvImage(msg->header, "mono8", blur).toImageMsg());

    cv::Mat in_f;
    blur.convertTo(in_f, CV_32F, 1.0/255.0);           

    const std::array<int64_t,4> shape{1,1,768,720};
    std::vector<float> in_buf(shape[2]*shape[3]);
    std::memcpy(in_buf.data(), in_f.data,
                in_buf.size()*sizeof(float));

    Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        mem, in_buf.data(), in_buf.size(),
        shape.data(), shape.size());

    Ort::AllocatedStringPtr in_name  =
        session_.GetInputNameAllocated (0, Ort::AllocatorWithDefaultOptions());
    Ort::AllocatedStringPtr out_name =
        session_.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    const char* input_names[]  = { in_name.get()  };
    const char* output_names[] = { out_name.get() };

    auto outs = session_.Run(Ort::RunOptions{nullptr},
                             input_names,  &in_tensor, 1,
                             output_names, 1);

    float* out_data = outs.front().GetTensorMutableData<float>();
    cv::Mat logits(768, 720, CV_32F, out_data);

    cv::Mat rail_mask;
    cv::threshold(logits, rail_mask, 0.5, 255, cv::THRESH_BINARY_INV);
    rail_mask.convertTo(rail_mask, CV_8U);

    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(rail_mask, labels,
                                             stats, centroids, 8, CV_32S);

    rail_mask.setTo(0);                             
    for (int i = 1; i < n; ++i) {                    
      if (stats.at<int>(i, cv::CC_STAT_AREA) >= min_area_px_) {
        rail_mask.setTo(255, labels == i);           
      }
    }

    pub_mask_.publish(
        cv_bridge::CvImage(msg->header, "mono8", rail_mask).toImageMsg());

    cv::Mat base;
    cv::cvtColor(blur, base, cv::COLOR_GRAY2BGR);

    const double alpha = 0.3;
    cv::Mat red(base.size(), CV_8UC3, cv::Scalar(0,0,255));
    cv::Mat blend;
    cv::addWeighted(base, 1.0 - alpha, red, alpha, 0.0, blend);

    blend.copyTo(base, rail_mask);      

    pub_overlay_.publish(
        cv_bridge::CvImage(msg->header, "bgr8", base).toImageMsg());

  } catch (const std::exception& e) {
    RCLCPP_WARN(this->get_logger(),
                "Processing failed: %s", e.what());
  }
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RailDetectorNode>());
  rclcpp::shutdown();
  return 0;
}