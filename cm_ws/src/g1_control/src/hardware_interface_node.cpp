#include <rclcpp/rclcpp.hpp>
#include <unitree_hg/msg/low_state.hpp>
#include <unitree_hg/msg/low_cmd.hpp>
#include "common/motor_crc_hg.h"
#include <g1_control_msgs/msg/robot_state.hpp>
#include <g1_control_msgs/msg/robot_command.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <std_msgs/msg/empty.hpp>
#include <array>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>

class HardwareInterfaceNode : public rclcpp::Node
{
public:
  HardwareInterfaceNode() : Node("hardware_interface_node")
  {
    // EMA parameters
    this->declare_parameter("alpha_pos", 0.9);
    this->declare_parameter("alpha_vel", 0.8);
    this->declare_parameter("alpha_tau", 0.7);
    this->declare_parameter("alpha_quat", 0.9);
    this->declare_parameter("alpha_omega", 0.8);
    this->declare_parameter("alpha_accel", 0.7);
    this->declare_parameter("alpha_cmd", 0.9);
    this->declare_parameter("enable_fall_detection", true);
    this->declare_parameter("fall_tilt_threshold", 0.3);
    alpha_pos_ = this->get_parameter("alpha_pos").as_double();
    alpha_vel_ = this->get_parameter("alpha_vel").as_double();
    alpha_tau_ = this->get_parameter("alpha_tau").as_double();
    alpha_quat_ = this->get_parameter("alpha_quat").as_double();
    alpha_omega_ = this->get_parameter("alpha_omega").as_double();
    alpha_accel_ = this->get_parameter("alpha_accel").as_double();
    alpha_cmd_ = this->get_parameter("alpha_cmd").as_double();
    enable_fall_detection_ = this->get_parameter("enable_fall_detection").as_bool();
    fall_tilt_threshold_ = this->get_parameter("fall_tilt_threshold").as_double();

    // QoS
    auto qos_best_effort = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
    auto qos_reliable = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
    auto qos_reliable_transient = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

    // Subscribers
    state_sub_ = this->create_subscription<unitree_hg::msg::LowState>(
        "/lowstate", qos_best_effort, std::bind(&HardwareInterfaceNode::lowStateCallback, this, std::placeholders::_1));
    cmd_sub_ = this->create_subscription<g1_control_msgs::msg::RobotCommand>(
        "/g1_control/robot_command", qos_best_effort, std::bind(&HardwareInterfaceNode::commandCallback, this, std::placeholders::_1));
    reset_sub_ = this->create_subscription<std_msgs::msg::Empty>(
        "/g1_control/reset", qos_reliable, std::bind(&HardwareInterfaceNode::resetCallback, this, std::placeholders::_1));
    estop_sub_ = this->create_subscription<std_msgs::msg::Empty>(
        "/g1_control/emergency_stop", qos_reliable, std::bind(&HardwareInterfaceNode::estopCallback, this, std::placeholders::_1));

    // Publishers
    state_pub_ = this->create_publisher<g1_control_msgs::msg::RobotState>("/g1_control/robot_state", qos_best_effort);
    cmd_pub_ = this->create_publisher<unitree_hg::msg::LowCmd>("/lowcmd", qos_best_effort);
    status_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticStatus>("/g1_control/operation_status", qos_reliable_transient);

    // Timers
    state_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(2), std::bind(&HardwareInterfaceNode::publishFilteredState, this));
    heartbeat_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100), std::bind(&HardwareInterfaceNode::publishStatus, this));

    // Initialize status messages
    ok_status_.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
    ok_status_.name = "Hardware Interface Status";
    ok_status_.message = "Robot active";

    error_status_.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
    error_status_.name = "Hardware Interface Status";
    error_status_.message = "Robot killed due to fall";

    initialized_ = false;
    is_ok_ = true;

    // Initialize low_cmd_
    for (size_t i = 0; i < NUM_JOINTS; ++i)
    {
      low_cmd_.motor_cmd[i].q = 0.0;
      low_cmd_.motor_cmd[i].dq = 0.0;
      low_cmd_.motor_cmd[i].tau = 0.0;
      low_cmd_.motor_cmd[i].kp = 0.0;
      low_cmd_.motor_cmd[i].kd = 0.0;
      // Note that the mode will remain constant for the lifetime of the node
      low_cmd_.motor_cmd[i].mode = 0x01; // Servo mode
    }
    // Ankle control in pitch/roll (PR) convention
    low_cmd_.mode_pr = 0;
  }

private:
  void resetCallback(const std_msgs::msg::Empty::SharedPtr)
  {
    RCLCPP_INFO(this->get_logger(), "Resetting hardware interface node");
    is_ok_ = true;
    initialized_ = false;
  }

  void estopCallback(const std_msgs::msg::Empty::SharedPtr)
  {
    RCLCPP_WARN(this->get_logger(), "Emergency stop command detected! Killing robot");
    is_ok_ = false;
    sendZeroTorque();
  }

  void lowStateCallback(const unitree_hg::msg::LowState::SharedPtr msg)
  {
    // The firmware rejects any LowCmd whose mode_machine doesn't match LowState
    mode_machine_ = msg->mode_machine;

    if (!initialized_)
    {
      for (size_t i = 0; i < NUM_JOINTS; ++i)
      {
        filtered_state_.q[i] = msg->motor_state[i].q;
        filtered_state_.dq[i] = msg->motor_state[i].dq;
        filtered_state_.tau_est[i] = msg->motor_state[i].tau_est;
        filtered_cmd_q_[i] = msg->motor_state[i].q;
      }

      current_quat_.w() = msg->imu_state.quaternion[0];
      current_quat_.x() = msg->imu_state.quaternion[1];
      current_quat_.y() = msg->imu_state.quaternion[2];
      current_quat_.z() = msg->imu_state.quaternion[3];

      filtered_quat_ = current_quat_;

      for (size_t i = 0; i < 3; ++i)
      {
        filtered_state_.gyroscope[i] = msg->imu_state.gyroscope[i];
        filtered_state_.accelerometer[i] = msg->imu_state.accelerometer[i];
      }
      initialized_ = true;
      return;
    }

    // EMA filter on joint state data
    for (size_t i = 0; i < NUM_JOINTS; ++i)
    {
      filtered_state_.q[i] = alpha_pos_ * msg->motor_state[i].q + (1.0 - alpha_pos_) * filtered_state_.q[i];
      filtered_state_.dq[i] = alpha_vel_ * msg->motor_state[i].dq + (1.0 - alpha_vel_) * filtered_state_.dq[i];
      filtered_state_.tau_est[i] = alpha_tau_ * msg->motor_state[i].tau_est + (1.0 - alpha_tau_) * filtered_state_.tau_est[i];
    }

    // SLERP for quaternion filtering
    current_quat_.w() = msg->imu_state.quaternion[0];
    current_quat_.x() = msg->imu_state.quaternion[1];
    current_quat_.y() = msg->imu_state.quaternion[2];
    current_quat_.z() = msg->imu_state.quaternion[3];
    // Ensure shortest path
    if (filtered_quat_.dot(current_quat_) < 0.0)
    {
      current_quat_.coeffs() *= -1.0;
    }
    filtered_quat_ = filtered_quat_.slerp(alpha_quat_, current_quat_);
    filtered_quat_.normalize();
    filtered_state_.quaternion[0] = filtered_quat_.w();
    filtered_state_.quaternion[1] = filtered_quat_.x();
    filtered_state_.quaternion[2] = filtered_quat_.y();
    filtered_state_.quaternion[3] = filtered_quat_.z();

    // EMA filter on IMU data
    for (size_t i = 0; i < 3; ++i)
    {
      filtered_state_.gyroscope[i] = alpha_omega_ * msg->imu_state.gyroscope[i] + (1.0 - alpha_omega_) * filtered_state_.gyroscope[i];
      filtered_state_.accelerometer[i] = alpha_accel_ * msg->imu_state.accelerometer[i] + (1.0 - alpha_accel_) * filtered_state_.accelerometer[i];
    }

    // Fall detection
    if (enable_fall_detection_ && is_ok_ && checkFall())
    {
      RCLCPP_WARN(this->get_logger(), "Fall detected! Killing robot.");
      is_ok_ = false;
      sendZeroTorque();
      status_pub_->publish(error_status_);
    }
  }

  void commandCallback(const g1_control_msgs::msg::RobotCommand::SharedPtr msg)
  {
    // Don't command until we've heard from the robot
    if (!is_ok_ || !initialized_)
    {
      return;
    }

    for (size_t i = 0; i < NUM_JOINTS; ++i)
    {
      filtered_cmd_q_[i] = alpha_cmd_ * msg->q[i] + (1.0 - alpha_cmd_) * filtered_cmd_q_[i];
      low_cmd_.motor_cmd[i].q = filtered_cmd_q_[i];
      low_cmd_.motor_cmd[i].dq = msg->dq[i];
      low_cmd_.motor_cmd[i].tau = msg->tau[i];
      low_cmd_.motor_cmd[i].kp = msg->kp[i];
      low_cmd_.motor_cmd[i].kd = msg->kd[i];
    }
    low_cmd_.mode_machine = mode_machine_;
    get_crc(low_cmd_);
    cmd_pub_->publish(low_cmd_);
  }

  bool checkFall()
  {
    Eigen::Vector3d world_z(0.0, 0.0, 1.0);
    Eigen::Vector3d body_z = filtered_quat_.inverse() * world_z;
    return body_z.z() < fall_tilt_threshold_;
  }

  void sendZeroTorque()
  {
    for (size_t i = 0; i < NUM_JOINTS; ++i)
    {
      low_cmd_.motor_cmd[i].q = 0.0;
      low_cmd_.motor_cmd[i].dq = 0.0;
      low_cmd_.motor_cmd[i].tau = 0.0;
      low_cmd_.motor_cmd[i].kp = 0.0;
      low_cmd_.motor_cmd[i].kd = 0.0;
    }
    low_cmd_.mode_machine = mode_machine_;
    get_crc(low_cmd_);
    cmd_pub_->publish(low_cmd_);
  }

  void publishStatus()
  {
    status_pub_->publish(is_ok_ ? ok_status_ : error_status_);
  }

  void publishFilteredState()
  {
    if (initialized_)
    {
      state_pub_->publish(filtered_state_);
    }
  }

  static constexpr size_t NUM_JOINTS = 29;

  rclcpp::Subscription<unitree_hg::msg::LowState>::SharedPtr state_sub_;
  rclcpp::Subscription<g1_control_msgs::msg::RobotCommand>::SharedPtr cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr estop_sub_;
  rclcpp::Publisher<g1_control_msgs::msg::RobotState>::SharedPtr state_pub_;
  rclcpp::Publisher<unitree_hg::msg::LowCmd>::SharedPtr cmd_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticStatus>::SharedPtr status_pub_;
  rclcpp::TimerBase::SharedPtr state_timer_;
  rclcpp::TimerBase::SharedPtr heartbeat_timer_;

  unitree_hg::msg::LowCmd low_cmd_;
  g1_control_msgs::msg::RobotState filtered_state_;
  diagnostic_msgs::msg::DiagnosticStatus ok_status_;
  diagnostic_msgs::msg::DiagnosticStatus error_status_;
  Eigen::Quaterniond filtered_quat_;
  Eigen::Quaterniond current_quat_;
  double alpha_pos_;
  double alpha_vel_;
  double alpha_tau_;
  double alpha_quat_;
  double alpha_omega_;
  double alpha_accel_;
  double alpha_cmd_;
  bool enable_fall_detection_;
  double fall_tilt_threshold_;
  std::array<double, NUM_JOINTS> filtered_cmd_q_{};
  bool initialized_;
  bool is_ok_;
  uint8_t mode_machine_ = 0;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HardwareInterfaceNode>());
  rclcpp::shutdown();
  return 0;
}
