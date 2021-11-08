if __name__ == "__main__":
    from scripts import busy_script

    # print()

    # print(sh.ls("/home/heider"))
    # print(sys.executable)

    # print(sh.systemctl('status', 'lightdm.service'))

    from pathlib import Path

    from draugr.os_utilities.linux_utilities.systemd_utilities.service_management import (
        install_service,
    )

    # remove_service("busy_script")
    install_service(Path(busy_script.__file__), "busy_script")
