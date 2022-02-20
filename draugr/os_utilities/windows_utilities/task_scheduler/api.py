import datetime
import getpass
import socket

try:
    import win32com.client
except ModuleNotFoundError as e:
    raise ModuleNotFoundError((e, "missing win32com.client"))

from draugr.os_utilities.windows_utilities.task_scheduler.enums import (
    TaskActionTypeEnum,
    TaskCreationEnum,
    TaskLogonTypeEnum,
    TaskTriggerEnum,
)

__all__ = ["new_user_logon_execute_task", "delete_task", "set_task_activity"]


def get_scheduler() -> win32com.client.CDispatch:
    scheduler = win32com.client.Dispatch("Schedule.Service")
    scheduler.Connect()
    return scheduler


def register_task(
    task_folder: win32com.client.CDispatch,
    task_definition: win32com.client.CDispatch,
    name: str,
    *,
    user: str = "",
    password: str = "",
    task_creation_type=TaskCreationEnum.TASK_CREATE_OR_UPDATE.value,
    task_logon_type=TaskLogonTypeEnum.TASK_LOGON_NONE.value,
) -> None:
    task_folder.RegisterTaskDefinition(
        name, task_definition, task_creation_type, user, password, task_logon_type
    )


def new_execute_action(
    task_def: win32com.client.CDispatch,
    action_path: str,
    action_arguments: str,
    *,
    action_id: str = "action0",
) -> None:
    action = task_def.Actions.Create(TaskActionTypeEnum.TASK_ACTION_EXEC.value)
    action.ID = action_id
    action.Path = action_path
    action.Arguments = action_arguments


def new_time_trigger(
    task_def: win32com.client.CDispatch,
    start_time: datetime.datetime = datetime.datetime.now()
    + datetime.timedelta(minutes=5),
    end_time: datetime.datetime = datetime.datetime.now()
    + datetime.timedelta(minutes=10),
) -> None:
    trigger = task_def.Triggers.Create(TaskTriggerEnum.TASK_TRIGGER_TIME.value)
    trigger.StartBoundary = start_time.isoformat()
    trigger.EndBoundary = end_time.isoformat()
    trigger.ExecutionTimeLimit = "PT5M"  # Five minutes


def new_logon_trigger(
    task_def: win32com.client.CDispatch,
    domain: str = socket.gethostname(),
    username: str = getpass.getuser(),
) -> None:
    trigger = task_def.Triggers.Create(TaskTriggerEnum.TASK_TRIGGER_LOGON.value)
    trigger.Id = "LogonTriggerId"
    trigger.UserId = f"{domain}\{username}"  # Must be a valid user account


def new_boot_trigger(task_def: win32com.client.CDispatch) -> None:
    trigger = task_def.Triggers.Create(TaskTriggerEnum.TASK_TRIGGER_BOOT.value)
    trigger.Id = "BootTriggerId"
    # trigger.Delay = "PT30S"  # 30 Seconds


def new_user_logon_execute_task(
    name: str,
    desc: str = "No description",
    action_path: str = "cmd.exe",
    action_arguments: str = '/c "exit"',
    *,
    task_folder: str = "\\User",
    stop_if_on_battery: bool = False,
) -> None:
    scheduler = get_scheduler()
    root_folder = scheduler.GetFolder(task_folder)
    task_def = scheduler.NewTask(0)
    task_def.RegistrationInfo.Description = desc
    task_def.RegistrationInfo.Author = getpass.getuser()
    task_def.Settings.Enabled = True
    task_def.Settings.StartWhenAvailable = True
    task_def.Settings.StopIfGoingOnBatteries = stop_if_on_battery

    new_logon_trigger(task_def)
    # new_boot_trigger(task_def)
    new_execute_action(task_def, action_path, action_arguments)
    register_task(root_folder, task_def, name)


def delete_task(task_name: str, *, task_folder: str = "\\User") -> None:
    scheduler = get_scheduler()
    task_folder = scheduler.GetFolder(task_folder)
    task_folder.DeleteTask(task_name, 0)


def set_task_activity(
    task_name: str, enable: bool, *, task_folder: str = "\\User"
) -> None:
    scheduler = get_scheduler()
    task_folder = scheduler.GetFolder(task_folder)
    task = task_folder.GetTask(task_name)
    task.Enabled = enable


if __name__ == "__main__":

    print(f"{socket.gethostname()}\{getpass.getuser()}")

    if True:
        delete_task("test_task")
        new_user_logon_execute_task("test_task")
        set_task_activity("test_task", False)
