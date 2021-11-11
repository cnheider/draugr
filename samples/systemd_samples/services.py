if __name__ == "__main__":
    from scripts import busy_script

    # print()

    # print(sh.ls("/home/heider"))
    # print(sys.executable)

    # print(sh.systemctl('status', 'lightdm.service'))

    from pathlib import Path

    from draugr.os_utilities.linux_utilities.systemd_utilities.service_management import (
        install_service,
        RunAsEnum,
        remove_service,
    )

    remove_service("busy_script", run_as=RunAsEnum.root)
    # install_service(Path(busy_script.__file__), "busy_script",run_as=RunAsEnum.root)
