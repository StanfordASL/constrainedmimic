#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "std_msgs/msg/header.hpp"
#include "std_msgs/msg/int32.hpp"
#include "g1_control_msgs/msg/frame_state.hpp"

class MocapNode : public rclcpp::Node
{
public:
    MocapNode() : Node("mocap_node")
    {
        // Declare and get parameters
        this->declare_parameter("filter_alpha", 0.5); // 0.0 to 1.0. Higher is more "recent"
        filter_alpha_ = this->get_parameter("filter_alpha").as_double();

        this->declare_parameter("left_foot_height_threshold", 0.02);
        this->declare_parameter("right_foot_height_threshold", 0.02);
        left_foot_height_threshold_ = this->get_parameter("left_foot_height_threshold").as_double();
        right_foot_height_threshold_ = this->get_parameter("right_foot_height_threshold").as_double();

        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();

        pelvis_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/vrpn_mocap/pelvis/pose",
            qos,
            std::bind(&MocapNode::pelvis_callback, this, std::placeholders::_1));

        left_foot_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/vrpn_mocap/left_foot/pose",
            qos,
            std::bind(&MocapNode::left_foot_callback, this, std::placeholders::_1));

        right_foot_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/vrpn_mocap/right_foot/pose",
            qos,
            std::bind(&MocapNode::right_foot_callback, this, std::placeholders::_1));

        pelvis_pub_ = this->create_publisher<g1_control_msgs::msg::FrameState>(
            "/g1_control/root_state", qos);

        contact_mode_pub_ = this->create_publisher<std_msgs::msg::Int32>(
            "/g1_control/contact_mode", qos);

        RCLCPP_INFO(this->get_logger(), "Mocap Node started");
        RCLCPP_INFO(this->get_logger(), "Filter alpha: %.2f", filter_alpha_);
        RCLCPP_INFO(this->get_logger(), "Foot height thresholds: L=%.3f, R=%.3f",
                    left_foot_height_threshold_, right_foot_height_threshold_);
    }

private:
    void pelvis_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        double current_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

        Eigen::Vector3d current_position(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
        Eigen::Quaterniond current_quat(
            msg->pose.orientation.w,
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z);

        if (last_time_ > 0.0)
        {
            double dt = current_time - last_time_;
            if (dt > 1e-6)
            {
                // Linear velocity
                Eigen::Vector3d instantaneous_velocity = (current_position - last_position_) / dt;

                // Angular velocity
                Eigen::Vector3d instantaneous_omega = compute_world_angular_velocity(
                    last_quat_, current_quat, dt);

                // EMA Filter
                if (first_step_)
                {
                    filtered_velocity_ = instantaneous_velocity;
                    filtered_omega_ = instantaneous_omega;
                    first_step_ = false;
                }
                else
                {
                    filtered_velocity_ = filter_alpha_ * instantaneous_velocity + (1.0 - filter_alpha_) * filtered_velocity_;
                    filtered_omega_ = filter_alpha_ * instantaneous_omega + (1.0 - filter_alpha_) * filtered_omega_;
                }
            }
        }

        // Update stored parameters
        last_time_ = current_time;
        last_position_ = current_position;
        last_quat_ = current_quat;

        auto out_msg = g1_control_msgs::msg::FrameState();
        for (int i = 0; i < 3; ++i)
        {
            out_msg.position[i] = current_position[i];
            out_msg.velocity[i] = filtered_velocity_[i];
            out_msg.omega[i] = filtered_omega_[i];
        }
        out_msg.quaternion[0] = current_quat.w();
        out_msg.quaternion[1] = current_quat.x();
        out_msg.quaternion[2] = current_quat.y();
        out_msg.quaternion[3] = current_quat.z();

        pelvis_pub_->publish(out_msg);
    }

    void left_foot_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        left_foot_z_ = msg->pose.position.z;
        publish_contact_mode();
    }

    void right_foot_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        right_foot_z_ = msg->pose.position.z;
        publish_contact_mode();
    }

    void publish_contact_mode()
    {
        bool left_contact = left_foot_z_ < left_foot_height_threshold_;
        bool right_contact = right_foot_z_ < right_foot_height_threshold_;

        int contact_mode = 0;
        if (left_contact && right_contact)
        {
            contact_mode = 3;
        }
        else if (right_contact)
        {
            contact_mode = 2;
        }
        else if (left_contact)
        {
            contact_mode = 1;
        }

        auto msg = std_msgs::msg::Int32();
        msg.data = contact_mode;
        contact_mode_pub_->publish(msg);
    }

    Eigen::Vector3d compute_world_angular_velocity(const Eigen::Quaterniond &q1, const Eigen::Quaterniond &q2, double dt)
    {
        Eigen::Quaterniond q1_norm = q1.normalized();
        Eigen::Quaterniond q2_norm = q2.normalized();

        // Ensure shortest path
        if (q1_norm.dot(q2_norm) < 0.0)
        {
            q2_norm.coeffs() *= -1.0;
        }

        Eigen::Quaterniond q_rel = q2_norm * q1_norm.conjugate();
        Eigen::AngleAxisd angle_axis(q_rel);
        double angle = angle_axis.angle();
        Eigen::Vector3d axis = angle_axis.axis();

        if (std::abs(angle) < 1e-8)
        {
            return Eigen::Vector3d::Zero();
        }

        return axis * (angle / dt);
    }

    double filter_alpha_;
    double left_foot_height_threshold_;
    double right_foot_height_threshold_;
    bool first_step_ = true;
    double last_time_ = -1.0;

    double left_foot_z_ = 1.0;
    double right_foot_z_ = 1.0;

    Eigen::Vector3d last_position_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond last_quat_ = Eigen::Quaterniond::Identity();

    Eigen::Vector3d filtered_velocity_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d filtered_omega_ = Eigen::Vector3d::Zero();

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pelvis_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr left_foot_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr right_foot_sub_;

    rclcpp::Publisher<g1_control_msgs::msg::FrameState>::SharedPtr pelvis_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr contact_mode_pub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MocapNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
