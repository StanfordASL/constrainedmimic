#include <chrono>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

#include "rclcpp/rclcpp.hpp"
#include "g1_control_msgs/msg/teleop_state.hpp"

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

using namespace std::chrono_literals;

class TeleopVisualizerNode : public rclcpp::Node
{
public:
    TeleopVisualizerNode() : Node("teleop_visualizer")
    {
        std::string default_model_path = "../cm_control/cm_control/assets/unitree_g1/cm_g1.xml";
        this->declare_parameter("model_path", default_model_path);
        std::string model_path = this->get_parameter("model_path").as_string();
        char error[1000];
        m_ = mj_loadXML(model_path.c_str(), nullptr, error, 1000);
        if (!m_)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not load MuJoCo model: %s", error);
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

        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        sub_ = this->create_subscription<g1_control_msgs::msg::TeleopState>(
            "/g1_control/teleop/state", qos,
            [this](const g1_control_msgs::msg::TeleopState::SharedPtr msg)
            {
                std::lock_guard<std::mutex> lock(mtx_);
                for (int i = 0; i < 29; ++i)
                {
                    d_->qpos[7 + i] = msg->q[i];
                }
                for (int i = 0; i < 3; ++i)
                {
                    d_->qpos[i] = msg->position[i];
                }
                for (int i = 0; i < 4; ++i)
                {
                    d_->qpos[3 + i] = msg->quaternion[i];
                }
                mj_forward(m_, d_);
            });

        RCLCPP_INFO(this->get_logger(), "Teleop Visualizer Node Started");
    }

    ~TeleopVisualizerNode()
    {
        if (d_)
            mj_deleteData(d_);
        if (m_)
            mj_deleteModel(m_);
    }

    std::mutex &mutex() { return mtx_; }
    mjModel *m() { return m_; }
    mjData *d() { return d_; }

private:
    mjModel *m_ = nullptr;
    mjData *d_ = nullptr;
    std::mutex mtx_;
    int keyframe_id_ = -1;
    rclcpp::Subscription<g1_control_msgs::msg::TeleopState>::SharedPtr sub_;
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
    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // update previous mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}

void mouse_move(GLFWwindow *window, double xpos, double ypos)
{
    // no buttons pressed: nothing to do
    if (!button_left && !button_middle && !button_right)
    {
        return;
    }

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (button_right)
    {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    }
    else if (button_left)
    {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    }
    else
    {
        action = mjMOUSE_ZOOM;
    }

    // move camera
    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

void scroll(GLFWwindow *window, double xoffset, double yoffset)
{
    // emulate zoom with scroll wheel
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TeleopVisualizerNode>();
    m = node->m();
    d = node->d();

    if (!glfwInit())
        return 1;
    GLFWwindow *window = glfwCreateWindow(1200, 900, "Teleop Visualizer", NULL, NULL);
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
        std::this_thread::sleep_for(20ms); // 50Hz Viewer
    }

    rclcpp::shutdown();
    if (spin_thread.joinable())
        spin_thread.join();
    glfwTerminate();

    return 0;
}
