SERVICE_TEMPLATE = """
[Unit]
Description = {description}
After = network.target 
 
[Service]
Type = simple
User = {service_user}
Group = {service_group}
ExecStart = {executable} {service_entry_point_path}
Restart = {restart} 
SyslogIdentifier = {service_name}
RestartSec = 5
TimeoutStartSec = infinity
 
[Install]
WantedBy = {service_target}
"""

# .config/systemd/user/{app}.service
# sudo systemctl --user start {app}.service
# sudo systemctl --user restart {app}.service
# sudo systemctl --user enable {app}.service
# sudo systemctl --user status {app}.service
# sudo systemctl --user disable {app}.service
SERVICE_TEMPLATE_USER_SIMPLE = """
[Unit]
Description={app} Service

[Service]
Type=simple
ExecStart={app_path}

[Install]
WantedBy=default.target
"""

SERVICE_TEMPLATE_SIMPLE = """
[Unit]
Description={service_name} Service
After=multi-user.target

[Service]
Type=simple
ExecStart={executable} {service_entry_point_path}

[Install]
WantedBy=multi-user.target
"""

if __name__ == "__main__":
    print(
        SERVICE_TEMPLATE.format(
            description=1,
            python_path=2,
            script_path=3,
            service_name=4,
            executable=5,
            service_target=6,
            service_entry_point_path=7,
            service_user=8,
            service_group=9,
        )
    )
    print(
        SERVICE_TEMPLATE_SIMPLE.format(
            service_name_a=1, executable=2, service_entry_point_path=3
        )
    )
    print(SERVICE_TEMPLATE_USER_SIMPLE)
