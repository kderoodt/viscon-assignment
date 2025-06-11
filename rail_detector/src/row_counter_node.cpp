#include "rail_detector/row_counter_node.hpp"
#include <rclcpp/rclcpp.hpp>

using std::placeholders::_1;

RowCounterNode::RowCounterNode()
: RowCounterNode(rclcpp::NodeOptions()) {}  

RowCounterNode::RowCounterNode(const rclcpp::NodeOptions & options)
: Node("row_counter", options)
{
  row_spacing_     = declare_parameter("row_spacing",     0.50);  
  min_area_px_     = declare_parameter("min_area_px",     2000);
  roi_y_ratio_     = declare_parameter("roi_y_ratio",     0.6);   
  min_overlap_px_  = declare_parameter("min_overlap_px",  500);

  sub_mask_ = create_subscription<sensor_msgs::msg::Image>(
      "/rail_mask", rclcpp::SensorDataQoS(),
      std::bind(&RowCounterNode::maskCallback, this, _1));

  sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odometry/filtered", rclcpp::SensorDataQoS(),
      std::bind(&RowCounterNode::odomCallback, this, _1));

  pub_count_ = create_publisher<std_msgs::msg::UInt32>("rows_count", 1);

  RCLCPP_INFO(get_logger(), "RowCounterNode initialised");
}

void RowCounterNode::odomCallback(
    const nav_msgs::msg::Odometry::ConstSharedPtr& msg)
{
  latest_odom_ = *msg;
}

void RowCounterNode::maskCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
  cv::Mat mask = cv_bridge::toCvCopy(msg, msg->encoding)->image;

  cv::Rect bb;
  bool rail_found = findNearestRailBlob(mask, bb);

  if (!active_rail_ && rail_found)          
  {
    active_bb_  = bb;
    active_rail_ = true;
    handleRowEvent();
  }
  else if (active_rail_)
  {
    int overlap = rail_found ? (bb & active_bb_).area() : 0;
    if (overlap < min_overlap_px_)
      active_rail_ = false;              
  }
}

bool RowCounterNode::findNearestRailBlob(const cv::Mat& mask,
                                         cv::Rect& out_bb) const
{
  cv::Mat labels, stats, centroids;
  int n = cv::connectedComponentsWithStats(mask, labels,
                                           stats, centroids, 8, CV_32S);

  double best_score = -1;
  int img_h = mask.rows;
  int roi_y = static_cast<int>(roi_y_ratio_ * img_h);

  for (int i = 1; i < n; ++i) 
  {
    int area   = stats.at<int>(i, cv::CC_STAT_AREA);
    int top    = stats.at<int>(i, cv::CC_STAT_TOP);
    int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
    int bottom = top + height;

    if (area < min_area_px_ || bottom < roi_y) continue;

    double score = bottom + 0.01 * area;   
    if (score > best_score)
    {
      best_score = score;
      out_bb = cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT),
                        top,
                        stats.at<int>(i, cv::CC_STAT_WIDTH),
                        height);
    }
  }
  return best_score > 0;
}

void RowCounterNode::handleRowEvent()
{
  const auto& p = latest_odom_.pose.pose.position;

  if (!have_last_row_pose_ ||
      hypot(p.x - last_row_pose_.position.x,
            p.y - last_row_pose_.position.y) >= row_spacing_)
  {
    row_count_++;
    last_row_pose_      = latest_odom_.pose.pose;
    have_last_row_pose_ = true;

    std_msgs::msg::UInt32 out;
    out.data = row_count_;
    pub_count_->publish(out);

    RCLCPP_INFO(get_logger(), "Rows passed: %u", row_count_);
  }
}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(RowCounterNode)

#ifdef ROW_COUNTER_BUILD_EXECUTABLE
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RowCounterNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
#endif