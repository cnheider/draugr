#  Copyright (c) 2021. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.


def asd():
    """ """
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
