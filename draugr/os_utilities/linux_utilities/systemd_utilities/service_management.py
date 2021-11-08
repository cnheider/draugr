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

import sh

from draugr import PROJECT_NAME
from draugr.os_utilities.linux_utilities.user_utilities import make_user, remove_user

__all__ = [
    "install_service",
    "remove_service",
    "enable_service",
    "disable_service",
]

from draugr.os_utilities.linux_utilities.systemd_utilities.service_template import (
    SERVICE_TEMPLATE,
)
from warg import ContextWrapper


class RunAsEnum(Enum):
    """ """

    user = "user"
    app_user = "app_user"
    root = "root"


def target_service_path(service_name):
    """ """
    return f"/lib/systemd/system/{service_name}.service"
    # ~/.config/systemd/user/ for user

    # /etc/systemd/system/ also an option
    # sudo chown root:root /etc/systemd/system/python_demo_service.service
    # sudo chmod 644 /etc/systemd/system/python_demo_service.service


def install_service(
    service_entry_point_path: Path,
    service_name: str,
    description: str = None,
    auto_enable: bool = True,
    run_as: RunAsEnum = RunAsEnum.app_user,
) -> None:
    """
    Args:
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

    systemd_service_file_path = target_service_path(project_service_name)
    print(f"Installing {systemd_service_file_path}")
    with sh.contrib.sudo(
        password=getpass.getpass(prompt=f"[sudo] password for {user}: "), _with=True
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
        sh.chown(
            f"{user}:", systemd_service_file_path
        )  # If a colon but no group name follows the user name, that user is made the owner of the files and the group of the files is changed to that user's login group.
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
                )
            )
        sh.chmod("644", systemd_service_file_path)
        sh.systemctl("daemon-reload")

        if auto_enable:
            enable_service(service_name, False)


def remove_service(
    service_name: str,
    remove_app_user: bool = True,
    run_as: RunAsEnum = RunAsEnum.app_user,
) -> None:
    """

    Args:

        :param service_name:
        :param remove_app_user:
        :param run_as:
    """

    try:
        with sh.contrib.sudo(
            password=getpass.getpass(
                prompt=f"[sudo] password for {getpass.getuser()}: "
            ),
            _with=True,
        ):
            disable_service(service_name, get_sudo=False)
            project_service_name = f"{PROJECT_NAME}_service_{service_name}"
            target_service_file_path = (
                f"/lib/systemd/system/{project_service_name}.service"
            )
            print(f"Removing {target_service_file_path}")

            sh.rm(target_service_file_path)
            sh.systemctl("daemon-reload")

            if run_as == RunAsEnum.app_user and remove_app_user:
                # DO CLEAN UP!
                remove_user(service_name + "_user", get_sudo=False)
    except sh.ErrorReturnCode_1:
        pass


def enable_service(service_name: str, get_sudo: bool = True) -> None:
    """

    Args:
        service_name:
        :param get_sudo:
    """
    project_service_name = f"{PROJECT_NAME}_service_{service_name}"
    print(f"Enabling {project_service_name}")
    with ContextWrapper(
        sh.contrib.sudo,
        construction_kwargs=dict(
            password=getpass.getpass(
                prompt=f"[sudo] password for {getpass.getuser()}: "
            )
            if get_sudo
            else None,
            _with=True,
        ),
        enabled=get_sudo,
    ):
        sh.systemctl(f"enable", f"{project_service_name}.service")
        start_service(service_name, False)


def disable_service(service_name: str, get_sudo: bool = True) -> None:
    """

    Args:
        service_name:
        :param get_sudo:
    """
    project_service_name = f"{PROJECT_NAME}_service_{service_name}"
    print(f"Disabling {project_service_name}")
    try:
        with ContextWrapper(
            sh.contrib.sudo,
            construction_kwargs=dict(
                password=getpass.getpass(
                    prompt=f"[sudo] password for {getpass.getuser()}: "
                )
                if get_sudo
                else None,
                _with=True,
            ),
            enabled=get_sudo,
        ):
            stop_service(service_name, False)
            sh.systemctl("disable", f"{project_service_name}.service")
    except sh.ErrorReturnCode_5:
        pass


def stop_service(service_name: str, get_sudo: bool = True) -> None:
    """

    Args:
        service_name:
        :param get_sudo:
    """
    project_service_name = f"{PROJECT_NAME}_service_{service_name}"
    print(f"Stopping {project_service_name}")
    try:
        with ContextWrapper(
            sh.contrib.sudo,
            construction_kwargs=dict(
                password=getpass.getpass(
                    prompt=f"[sudo] password for {getpass.getuser()}: "
                )
                if get_sudo
                else None,
                _with=True,
            ),
            enabled=get_sudo,
        ):
            sh.systemctl("stop", f"{project_service_name}.service")
    except sh.ErrorReturnCode_5:
        pass


def start_service(service_name: str, get_sudo: bool = True) -> None:
    """

    Args:
        service_name:
        :param get_sudo:
    """
    project_service_name = f"{PROJECT_NAME}_service_{service_name}"
    print(f"Starting {project_service_name}")
    try:
        with ContextWrapper(
            sh.contrib.sudo,
            construction_kwargs=dict(
                password=getpass.getpass(
                    prompt=f"[sudo] password for {getpass.getuser()}: "
                )
                if get_sudo
                else None,
                _with=True,
            ),
            enabled=get_sudo,
        ):
            sh.systemctl("start", f"{project_service_name}.service")
    except sh.ErrorReturnCode_5:
        pass


def restart_service(service_name: str, get_sudo: bool = True) -> None:
    """

    Args:
        service_name:
        :param get_sudo:
    """
    project_service_name = f"{PROJECT_NAME}_service_{service_name}"
    print(f"Restarting {project_service_name}")
    try:
        with ContextWrapper(
            sh.contrib.sudo,
            construction_kwargs=dict(
                password=getpass.getpass(
                    prompt=f"[sudo] password for {getpass.getuser()}: "
                )
                if get_sudo
                else None,
                _with=True,
            ),
            enabled=get_sudo,
        ):
            sh.systemctl("restart", f"{project_service_name}.service")
    except sh.ErrorReturnCode_5:
        pass


def status_service(service_name: str, get_sudo: bool = False) -> None:
    """

    Args:
        service_name:
        :param get_sudo:
    """
    project_service_name = f"{PROJECT_NAME}_service_{service_name}"
    print(f"Status for {project_service_name}")
    try:
        with ContextWrapper(
            sh.contrib.sudo,
            construction_kwargs=dict(
                password=getpass.getpass(
                    prompt=f"[sudo] password for {getpass.getuser()}: "
                )
                if get_sudo
                else None,
                _with=True,
            ),
            enabled=get_sudo,
        ):
            sh.systemctl("status", f"{project_service_name}.service")
    except sh.ErrorReturnCode_3 as e:
        print(e)


if __name__ == "__main__":
    remove_service("busy_script")
    # install_service('busy_script')
    # status_service('busy_script')
    # restart_service('busy_script')
