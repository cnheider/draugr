#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17/03/2020
           """

import getpass
import sys
from enum import Enum
from pathlib import Path

import sh  # pip install sh
from sorcery import assigned_names

from draugr import PROJECT_NAME
from draugr.os_utilities.linux_utilities.user_utilities import make_user, remove_user

__all__ = [
    "install_service",
    "remove_service",
    "enable_service",
    "disable_service",
    "start_service",
    "stop_service",
    "restart_service",
    "status_service",
    "RunAsEnum",
]

from draugr.os_utilities.linux_utilities.systemd_utilities.service_template import (
    SERVICE_TEMPLATE,
)
from warg import ContextWrapper


class RunAsEnum(Enum):
    """ """

    user, app_user, root = assigned_names()


def target_service_path(service_name, run_as: RunAsEnum = RunAsEnum.user):
    """ """
    if run_as == RunAsEnum.user:
        return Path.home() / ".config" / "systemd" / "user" / f"{service_name}.service"
    elif run_as == run_as.root:
        return Path("/") / "etc" / "systemd" / "system" / f"{service_name}.service"
    return Path("/") / "lib" / "systemd" / "system" / f"{service_name}.service"


class RestartServiceEnum(Enum):
    """
    Restart settings\Exit causes	 no	  always      on-success	    on-failure	    on-abnormal	on-abort	on-watchdog
    Clean exit code or signal	 	        X	          X
    Unclean exit code	 	                X	 	                        X
    Unclean signal	 	                  X	 	                        X	              X	          X
    Timeout	 	                          X	 	                        X	              X
    Watchdog	 	                        X	 	                        X	              X	 	                  X
    """

    no, always = assigned_names()
    on_failure = "on-failure"
    on_success = "on-success"
    on_abnormal = "on-abnormal"
    on_abort = "on-abort"
    on_watchdog = "on-watchdog"


class ServiceTargetEnum(Enum):  # TODO: UTILISE!
    """
    Default is default

    SysV 	      systemd Target	                                                Notes
    0	          runlevel0.target, poweroff.target	                              Halt the system.
    1, s,       single	runlevel1.target, rescue.target	                        Single user mode.
    2, 4	      runlevel2.target, runlevel4.target, multi-user.target	          User-defined/Site-specific runlevels. By default, identical to 3.
    3	          runlevel3.target, multi-user.target	                            Multi-user, non-graphical. Users can usually login via multiple consoles or via the network.
    5	          runlevel5.target, graphical.target	                            Multi-user, graphical. Usually has all the services of runlevel 3 plus a graphical login.
    6	          runlevel6.target, reboot.target	                                Reboot
    emergency	  emergency.target	                                              Emergency shell
    """

    default = "default.target"
    multi_user = "multi-user.target"


def install_service(
    service_entry_point_path: Path,
    service_name: str,
    *,
    description: str = None,
    auto_enable: bool = True,
    run_as: RunAsEnum = RunAsEnum.user,
    # get_sudo: bool = False,
    restart: RestartServiceEnum = RestartServiceEnum.on_failure,
) -> None:
    """
    Args:
        :param restart:
        :param service_entry_point_path:
        :param service_name:
        :param description:
        :param auto_enable:
        :param run_as:
    """
    assert (
        service_entry_point_path.is_file()
        and service_entry_point_path.name.endswith(".py")
    )
    project_service_name = f"{PROJECT_NAME}_service_{service_name}"
    user = getpass.getuser()

    systemd_service_file_path = target_service_path(project_service_name, run_as=run_as)
    print(f"Installing {systemd_service_file_path}")
    get_sudo = run_as != RunAsEnum.user
    with ContextWrapper(
        sh.contrib.sudo,
        construction_kwargs=dict(
            password=(
                getpass.getpass(prompt=f"[sudo] password for {user}: ")
                if get_sudo
                else None
            ),
            _with=True,
        ),
        enabled=get_sudo,
    ):
        if run_as == RunAsEnum.app_user:
            service_user = service_name + "_user"
            make_user(service_user, get_sudo=False)
            service_target = "default.target"
            service_group = service_user
        elif run_as == RunAsEnum.root:
            service_user = "root"
            service_target = "multi-user.target"
            service_group = service_user
        elif run_as == RunAsEnum.user:
            service_user = user
            service_target = "default.target"
            service_group = service_user
        else:
            raise ValueError

        sh.touch(systemd_service_file_path)
        group_name = str(sh.id(["-g", "-n", service_user])).strip("\n")
        assert service_group == group_name
        current_owner = sh.ls("-l", systemd_service_file_path).split(" ")[2]
        if current_owner != service_user:  # SETTING UP PERMISSIONS
            print(
                f"Changing owner of service file from {current_owner} to {service_user}"
            )
            if run_as == RunAsEnum.root:
                group_name = ""
            else:
                print(f"with common group {group_name}")
                # group_id = sh.id(["-g", service_user])
                sh.usermod(
                    ["-a", "-G", group_name, user]
                )  # TODO: Polluting groups of user
        sh.chown(
            [f"{user}:{group_name}", service_entry_point_path]
        )  # If a colon but no group name follows the user name, that user is made the owner of the files and the group of the files is changed to that user's login group.
        sh.chown(
            [f"{user}:{group_name}", systemd_service_file_path]
        )  # If a colon but no group name follows the user name, that user is made the owner of the files and the group of the files is changed to that user's login group.

        print("writing service file")
        if not description:
            description = f"heimdallr service for {service_name}"
        with open(systemd_service_file_path, "w") as f:
            f.writelines(
                SERVICE_TEMPLATE.format(
                    service_name=project_service_name,
                    service_user=service_user,
                    executable=sys.executable,
                    description=description,
                    service_entry_point_path=service_entry_point_path,
                    service_target=service_target,
                    service_group=service_group,
                    restart=restart.value,
                )
            )
        sh.chown(
            [f"{service_user}:{group_name}", systemd_service_file_path]
        )  # If a colon but no group name follows the user name, that user is made the owner of the files and the group of the files is changed to that user's login group.
        sh.chmod(["664", systemd_service_file_path])
        sh.chmod(["774", service_entry_point_path])
        sh.systemctl("daemon-reload")  # TODO: Requires sudo?

        if auto_enable:
            enable_service(service_name, get_sudo=False, run_as=run_as)


def remove_service(
    service_name: str,
    *,
    remove_app_user: bool = True,
    get_sudo: bool = False,
    run_as: RunAsEnum = RunAsEnum.user,
) -> None:
    """

    Args:
        :param get_sudo:
        :param service_name:
        :param remove_app_user:
        :param run_as:
    """

    try:
        # get_sudo = not run_as == RunAsEnum.user
        with ContextWrapper(
            sh.contrib.sudo,
            construction_kwargs=dict(
                password=(
                    getpass.getpass(prompt=f"[sudo] password for {getpass.getuser()}: ")
                    if get_sudo
                    else None
                ),
                _with=True,
            ),
            enabled=get_sudo,
        ):
            disable_service(service_name, get_sudo=False, run_as=run_as)
            project_service_name = f"{PROJECT_NAME}_service_{service_name}"
            target_service_file_path = target_service_path(
                project_service_name, run_as=run_as
            )
            print(f"Removing {target_service_file_path}")

            sh.rm(target_service_file_path)
            sh.systemctl("daemon-reload")

            if run_as == RunAsEnum.app_user and remove_app_user:
                # DO CLEAN UP!
                remove_user(service_name + "_user", get_sudo=False, run_as=run_as)
    except sh.ErrorReturnCode_1:
        pass


def enable_service(
    service_name: str, *, get_sudo: bool = False, run_as: RunAsEnum = RunAsEnum.user
) -> None:
    """

    Args:
        service_name:
        :param service_name:
        :param run_as:
        :param get_sudo:
    """
    project_service_name = f"{PROJECT_NAME}_service_{service_name}"
    print(f"Enabling {project_service_name}")
    with ContextWrapper(
        sh.contrib.sudo,
        construction_kwargs=dict(
            password=(
                getpass.getpass(prompt=f"[sudo] password for {getpass.getuser()}: ")
                if get_sudo
                else None
            ),
            _with=True,
        ),
        enabled=get_sudo,
    ):
        sh.systemctl(
            (["--user"] if run_as == RunAsEnum.user else [])
            + [f"enable", f"{project_service_name}.service"]
        )
        start_service(service_name, get_sudo=False, run_as=run_as)


def disable_service(
    service_name: str, *, get_sudo: bool = False, run_as: RunAsEnum = RunAsEnum.user
) -> None:
    """

    Args:
        service_name:
        :param service_name:
        :param run_as:
        :param get_sudo:
    """
    project_service_name = f"{PROJECT_NAME}_service_{service_name}"
    print(f"Disabling {project_service_name}")
    try:
        with ContextWrapper(
            sh.contrib.sudo,
            construction_kwargs=dict(
                password=(
                    getpass.getpass(prompt=f"[sudo] password for {getpass.getuser()}: ")
                    if get_sudo
                    else None
                ),
                _with=True,
            ),
            enabled=get_sudo,
        ):
            stop_service(service_name, get_sudo=False, run_as=run_as)
            sh.systemctl(
                (["--user"] if run_as == RunAsEnum.user else [])
                + ["disable", f"{project_service_name}.service"]
            )
    except sh.ErrorReturnCode_5:
        pass


def stop_service(
    service_name: str, *, get_sudo: bool = False, run_as: RunAsEnum = RunAsEnum.user
) -> None:
    """

    Args:
        service_name:
        :param service_name:
        :param run_as:
        :param get_sudo:
    """
    project_service_name = f"{PROJECT_NAME}_service_{service_name}"
    print(f"Stopping {project_service_name}")
    try:
        with ContextWrapper(
            sh.contrib.sudo,
            construction_kwargs=dict(
                password=(
                    getpass.getpass(prompt=f"[sudo] password for {getpass.getuser()}: ")
                    if get_sudo
                    else None
                ),
                _with=True,
            ),
            enabled=get_sudo,
        ):
            sh.systemctl(
                (["--user"] if run_as == RunAsEnum.user else [])
                + ["stop", f"{project_service_name}.service"]
            )
    except sh.ErrorReturnCode_5:
        pass


def start_service(
    service_name: str, *, get_sudo: bool = False, run_as: RunAsEnum = RunAsEnum.user
) -> None:
    """

    Args:
        service_name:
        :param service_name:
        :param run_as:
        :param get_sudo:
    """
    project_service_name = f"{PROJECT_NAME}_service_{service_name}"
    print(f"Starting {project_service_name}")
    try:
        with ContextWrapper(
            sh.contrib.sudo,
            construction_kwargs=dict(
                password=(
                    getpass.getpass(prompt=f"[sudo] password for {getpass.getuser()}: ")
                    if get_sudo
                    else None
                ),
                _with=True,
            ),
            enabled=get_sudo,
        ):
            sh.systemctl(
                (["--user"] if run_as == RunAsEnum.user else [])
                + ["start", f"{project_service_name}.service"]
            )
    except sh.ErrorReturnCode_5:
        pass


def restart_service(
    service_name: str, *, get_sudo: bool = False, run_as: RunAsEnum = RunAsEnum.user
) -> None:
    """

    Args:
        service_name:
        :param service_name:
        :param run_as:
        :param get_sudo:
    """
    project_service_name = f"{PROJECT_NAME}_service_{service_name}"
    print(f"Restarting {project_service_name}")
    try:
        with ContextWrapper(
            sh.contrib.sudo,
            construction_kwargs=dict(
                password=(
                    getpass.getpass(prompt=f"[sudo] password for {getpass.getuser()}: ")
                    if get_sudo
                    else None
                ),
                _with=True,
            ),
            enabled=get_sudo,
        ):
            sh.systemctl(
                (["--user"] if run_as == RunAsEnum.user else [])
                + ["restart", f"{project_service_name}.service"]
            )
    except sh.ErrorReturnCode_5:
        pass


def status_service(
    service_name: str, *, get_sudo: bool = False, run_as: RunAsEnum = RunAsEnum.user
) -> None:
    """

    Args:
        service_name:
        :param service_name:
        :param run_as:
        :param get_sudo:
    """
    project_service_name = f"{PROJECT_NAME}_service_{service_name}"
    print(f"Status for {project_service_name}")
    try:
        with ContextWrapper(
            sh.contrib.sudo,
            construction_kwargs=dict(
                password=(
                    getpass.getpass(prompt=f"[sudo] password for {getpass.getuser()}: ")
                    if get_sudo
                    else None
                ),
                _with=True,
            ),
            enabled=get_sudo,
        ):
            sh.systemctl(
                (["--user"] if run_as == RunAsEnum.user else [])
                + ["status", f"{project_service_name}.service"]
            )
    except sh.ErrorReturnCode_3 as e:
        print(e, e.stdout)


if __name__ == "__main__":
    pass
    print(RunAsEnum.user.value)
    # remove_service("busy_script")
    # install_service('busy_script')
    # status_service('busy_script')
    # restart_service('busy_script')
