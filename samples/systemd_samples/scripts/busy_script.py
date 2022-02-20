def asd() -> None:
    """
    :rtype: None
    """
    import time

    from draugr import IgnoreInterruptSignal, busy_indicator

    next_reading = time.time()
    with IgnoreInterruptSignal():
        print("Publisher started")

        for _ in busy_indicator():
            next_reading += 3
            sleep_time = next_reading - time.time()

            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":

    if False:
        from pathlib import Path

        from draugr.os_utilities.linux_utilities.systemd_utilities.service_management import (
            install_service,
        )

        install_service(Path(__file__), "busy_script")
    else:
        asd()
