# VSCode tips

Here's a copy of my current `settings.json` file to make VSCode recognize any Python imports:

```json
{
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "charliermarsh.ruff",
    "python.analysis.extraPaths": [
        "${workspaceFolder}/cm_control",
        "/opt/ros/jazzy/lib/python3.12/site-packages",
        "${workspaceFolder}/cm_ws/install/g1_control/lib/python3.12/site-packages",
        "${workspaceFolder}/cm_ws/install/g1_control_msgs/lib/python3.12/site-packages",
        "${workspaceFolder}/cm_ws/install/xrobo_ros2/lib/python3.12/site-packages",
    ],
    "python.autoComplete.extraPaths": [
        "${workspaceFolder}/cm_control",
        "/opt/ros/jazzy/lib/python3.12/site-packages",
        "${workspaceFolder}/cm_ws/install/g1_control/lib/python3.12/site-packages",
        "${workspaceFolder}/cm_ws/install/g1_control_msgs/lib/python3.12/site-packages",
        "${workspaceFolder}/cm_ws/install/xrobo_ros2/lib/python3.12/site-packages",
    ]
}
```

I always open VSCode in the top-level `constrainedmimic` directory.