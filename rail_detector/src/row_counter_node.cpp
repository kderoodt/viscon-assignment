#include "rail_detector/row_counter_node.hpp"
#include <rclcpp/rclcpp.hpp>

// ── RowCounterNode overview ──────────────────────────────────────────────
// Counts greenhouse heating rails (rows) based on a binary rail mask and
// odometry spacing.
//   • Subs: /rail_mask  (sensor_msgs/Image)
//           /odometry/filtered (nav_msgs/Odometry)
//   • Pub : rows_count (std_msgs/UInt32)
//   • Params: row_spacing (m), min_area_px, min_overlap_px
// ──────────────────────────────────────────────────────────────────────────

using std::placeholders::_1;

RowCounterNode::RowCounterNode()
: RowCounterNode(rclcpp::NodeOptions()) {}

RowCounterNode::RowCounterNode(const rclcpp::NodeOptions & options)
: Node("row_counter", options)
{
  // ── parameters ────────────────────────────────────────────────────────
  row_spacing_     = declare_parameter("row_spacing",     0.60);
  min_area_px_     = declare_parameter("min_area_px",     1000);
  min_overlap_px_  = declare_parameter("min_overlap_px",  500);

  // ── subscriptions & publisher ────────────────────────────────────────
  sub_mask_ = create_subscription<sensor_msgs::msg::Image>(
      "/rail_mask", rclcpp::SensorDataQoS(),
      std::bind(&RowCounterNode::maskCallback, this, _1));

  sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odometry/filtered", rclcpp::SensorDataQoS(),
      std::bind(&RowCounterNode::odomCallback, this, _1));

  pub_count_ = create_publisher<std_msgs::msg::UInt32>("rows_count", 1);

  RCLCPP_INFO(get_logger(), "RowCounterNode initialised");
}

// ── odometry callback ───────────────────────────────────────────────────
void RowCounterNode::odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& msg)
{
  latest_odom_ = *msg;  // keep most recent pose for spacing calculation
}

// ── mask callback ───────────────────────────────────────────────────────
void RowCounterNode::maskCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
  cv::Mat mask = cv_bridge::toCvCopy(msg, msg->encoding)->image;

  cv::Rect bb;
  bool rail_found = findNearestRailBlob(mask, bb);  // locate most likely rail blob

  if (!active_rail_ && rail_found)
  {
    // ── entering a new rail ────────────────────────────────────────────
    active_bb_  = bb;
    active_rail_ = true;
    handleRowEvent();
  }
  else if (active_rail_)
  {
    // ── check if we left the active rail ──────────────────────────────
    int overlap = rail_found ? (bb & active_bb_).area() : 0;

    int active_cx  = active_bb_.x + active_bb_.width / 2;
    int current_cx = bb.x + bb.width / 2;
    int horizontal_gap = std::abs(current_cx - active_cx);

    if (overlap < min_overlap_px_ && horizontal_gap > 100)
      active_rail_ = false;  // ready to detect the next rail
  }
}

// ── findNearestRailBlob ─────────────────────────────────────────────────
bool RowCounterNode::findNearestRailBlob(const cv::Mat& mask, cv::Rect& out_bb) const
{
  cv::Mat labels, stats, centroids;
  int n = cv::connectedComponentsWithStats(mask, labels,
                                           stats, centroids, 8, CV_32S);

  double best_score = -1e9;
  int center_x = mask.cols / 2;

  for (int i = 1; i < n; ++i)
  {
    int area   = stats.at<int>(i, cv::CC_STAT_AREA);
    int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
    int width  = stats.at<int>(i, cv::CC_STAT_WIDTH);
    int left   = stats.at<int>(i, cv::CC_STAT_LEFT);
    int top    = stats.at<int>(i, cv::CC_STAT_TOP);

    if (area < min_area_px_ || height < width) continue; // filter non‑rail blobs

    int blob_cx = left + width / 2;
    double score = -std::abs(blob_cx - center_x);        // favour centre blobs

    if (score > best_score)
    {
      best_score = score;
      out_bb = cv::Rect(left, top, width, height);
    }
  }
  return best_score > -1e9;
}

// ── handleRowEvent ─────────────────────────────────────────────────────
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

    std_msgs::msg::UInt32 out; out.data = row_count_;
    pub_count_->publish(out);

    RCLCPP_INFO(get_logger(), "Rows passed: %u", row_count_);
  }
}

// ── component registration & optional executable ──────────────────────
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
