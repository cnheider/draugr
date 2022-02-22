#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.


def a():
    import subprocess
    import sys

    reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    installed_packages = [r.decode().split("==")[0] for r in reqs.split()]


def b():
    import importlib.util
    import sys

    name = "itertools"  # For illustrative purposes.

    if name in sys.modules:
        print(f"{name!r} already in sys.modules")
    elif (spec := importlib.util.find_spec(name)) is not None:
        # If you choose to perform the actual import ...
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        print(f"{name!r} has been imported")
    else:
        print(f"can't find the {name!r} module")


def c():
    import pkg_resources
    from pkg_resources import DistributionNotFound, VersionConflict

    # dependencies can be any iterable with strings,
    # e.g. file line-by-line iterator
    dependencies = [
        "Werkzeug>=0.6.1",
        "Flask>=0.9",
    ]

    # here, if a dependency is not met, a DistributionNotFound or VersionConflict
    # exception is thrown.
    pkg_resources.require(dependencies)
