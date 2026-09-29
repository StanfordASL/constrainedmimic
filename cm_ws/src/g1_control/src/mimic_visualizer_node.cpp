#include <chrono>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "g1_control_msgs/msg/mimic_observation.hpp"
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

using namespace std::chrono_literals;

class MimicVisualizerNode : public rclcpp::Node
{
public:
    MimicVisualizerNode() : Node("mimic_visualizer")
    {
        std::string default_model_path = "../cm_control/cm_control/assets/unitree_g1/cm_g1.xml";
        this->declare_parameter("model_path", default_model_path);
        std::string model_path = this->get_parameter("model_path").as_string();

        char error[1000];
        m_ = mj_loadXML(model_path.c_str(), nullptr, error, 1000);
        if (!m_)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not load MuJoCo model: %s. Tried path: %s", error, model_path.c_str());
            exit(1);
        }
        d_ = mj_makeData(m_);

        keyframe_id_ = mj_name2id(m_, mjOBJ_KEY, "home");
        if (keyframe_id_ == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not find keyframe 'home'");
            exit(1);
        }
        // Perform an initial reset to the home keyframe
        mj_resetDataKeyframe(m_, d_, keyframe_id_);
        mj_forward(m_, d_);

        // Initialize state
        x_ = 0.0;
        y_ = 0.0;
        yaw_ = 0.0;
        last_time_ = this->now();

        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        sub_ = this->create_subscription<g1_control_msgs::msg::MimicObservation>(
            "/g1_control/mimic_obs", qos,
            std::bind(&MimicVisualizerNode::mimic_obs_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Mimic Visualizer Node Started");
    }

    ~MimicVisualizerNode()
    {
        if (d_)
        {
            mj_deleteData(d_);
        }
        if (m_)
        {
            mj_deleteModel(m_);
        }
    }

    std::mutex &mutex() { return mtx_; }
    mjModel *m() { return m_; }
    mjData *d() { return d_; }

private:
    void mimic_obs_callback(const g1_control_msgs::msg::MimicObservation::SharedPtr msg)
    {
        if (msg->data.size() < 35)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "Received mimic_obs with insufficient data size: %zu (expected >= 35)", msg->data.size());
            return;
        }

        double vx = msg->data[0];
        double vy = msg->data[1];
        double z = msg->data[2];
        double roll = msg->data[3];
        double pitch = msg->data[4];
        double yaw_vel = msg->data[5];

        rclcpp::Time now = this->now();
        double dt = (now - last_time_).seconds();
        last_time_ = now;

        if (dt > 0.1)
        {
            RCLCPP_INFO(this->get_logger(), "Large dt detected: %f (has it been a while since publishing mimic data?). Assuming 100 Hz", dt);
            dt = 0.01;
        }

        // Integrate to update state
        yaw_ += yaw_vel * dt;
        double cos_y = std::cos(yaw_);
        double sin_y = std::sin(yaw_);
        x_ += (vx * cos_y - vy * sin_y) * dt;
        y_ += (vx * sin_y + vy * cos_y) * dt;

        std::lock_guard<std::mutex> lock(mtx_);

        // Root position
        d_->qpos[0] = x_;
        d_->qpos[1] = y_;
        d_->qpos[2] = z;

        // Root orientation
        Eigen::Quaterniond q = Eigen::AngleAxisd(yaw_, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
        q.normalize();

        d_->qpos[3] = q.w();
        d_->qpos[4] = q.x();
        d_->qpos[5] = q.y();
        d_->qpos[6] = q.z();

        // Joint angles
        for (int i = 0; i < 29; ++i)
        {
            d_->qpos[7 + i] = msg->data[6 + i];
        }

        mj_forward(m_, d_);
    }

    mjModel *m_ = nullptr;
    mjData *d_ = nullptr;
    std::mutex mtx_;
    int keyframe_id_ = -1;
    rclcpp::Subscription<g1_control_msgs::msg::MimicObservation>::SharedPtr sub_;

    double x_ = 0, y_ = 0, yaw_ = 0;
    rclcpp::Time last_time_;
};

// Global MuJoCo objects for GLFW callbacks
mjModel *m = nullptr;
mjData *d = nullptr;
mjvScene scn;
mjvCamera cam;
mjvOption opt;
mjrContext con;

// mouse state
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

void mouse_button(GLFWwindow *window, int button, int act, int mods)
{
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    glfwGetCursorPos(window, &lastx, &lasty);
}

void mouse_move(GLFWwindow *window, double xpos, double ypos)
{
    if (!button_left && !button_middle && !button_right)
        return;

    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    int width, height;
    glfwGetWindowSize(window, &width, &height);

    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    mjtMouse action;
    if (button_right)
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if (button_left)
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

void scroll(GLFWwindow *window, double xoffset, double yoffset)
{
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MimicVisualizerNode>();
    m = node->m();
    d = node->d();

    if (!glfwInit())
        return 1;
    GLFWwindow *window = glfwCreateWindow(1200, 900, "Mimic Visualizer", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // Track pelvis
    int pelvis_id = mj_name2id(m, mjOBJ_BODY, "pelvis");
    if (pelvis_id != -1)
    {
        cam.type = mjCAMERA_TRACKING;
        cam.trackbodyid = pelvis_id;
        cam.distance = 2.0;
        cam.lookat[0] = 0;
        cam.lookat[1] = 0;
        cam.lookat[2] = 0.8;
    }

    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    std::thread spin_thread([&]()
                            { rclcpp::spin(node); });

    while (!glfwWindowShouldClose(window) && rclcpp::ok())
    {
        {
            std::lock_guard<std::mutex> lock(node->mutex());
            mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);

            mjrRect viewport = {0, 0, 0, 0};
            glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
            mjr_render(viewport, &scn, &con);
        }
        glfwSwapBuffers(window);
        glfwPollEvents();
        std::this_thread::sleep_for(20ms);
    }

    rclcpp::shutdown();
    if (spin_thread.joinable())
        spin_thread.join();
    glfwTerminate();

    return 0;
}
