from pathlib import Path

from warg import ContextWrapper


def make_user(
    username: str = "demo_user",
    password: str = None,
    *,
    add_home: bool = True,
    home_dir: Path = None,
    allow_existing_user: bool = True,
    get_sudo: bool = True,
) -> None:
    """ """
    import crypt
    import sh
    import getpass

    query = []

    if add_home:
        query += [f"-m", f"-d"]
        if home_dir:
            query += [str(home_dir)]
        else:
            query += [f"/home/{username}"]

    try:
        user_id = sh.id(["-u", username])
        if int(user_id):
            if not allow_existing_user:
                raise FileExistsError
            group_id = sh.id(["-g", username])
            print(f"user {username} exists with id {user_id} and {group_id}")
    except (ValueError, sh.ErrorReturnCode_1):
        pass
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
            try:
                sh.useradd(
                    query
                    + [
                        f"-p",
                        f"{crypt.crypt(password if password else input(f'new password for user {username}: '), '22')}",
                        f"{username}",
                    ]
                )
            except sh.ErrorReturnCode_9:
                pass


def remove_user(
    username: str = "demo_user", *, remove_home: bool = True, get_sudo: bool = True
) -> None:
    """ """
    import sh
    import getpass

    try:
        user_id = sh.id(["-u", username])
        if int(user_id):
            print(f"User {username} exists with id {user_id}")
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
                sh.userdel((["-r"] if remove_home else []) + [f"{username}"])
                print(f"Removed user {username}")
    except (ValueError, sh.ErrorReturnCode_1):
        pass


def change_passwd(
    username: str = "demo_user",
    password: str = None,
) -> None:
    """

    :param username:
    :param password:
    """
    raise NotImplementedError
    pass  # ./passwd


def change_home_dir(username: str = "demo_user", new_home: str = None) -> None:
    """

    :param username:
    :param new_home:
    """
    raise NotImplementedError
    pass  # ./mkhomedir_helper username


if __name__ == "__main__":
    make_user()
    remove_user()
