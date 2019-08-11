#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def python_version_check():
    import sys

    assert sys.version_info.major == 3 and sys.version_info.minor >= 6, (
        f"This project is utilises language features only present Python 3.6 and greater. "
        f"You are running {sys.version_info}."
    )


python_version_check()

import pathlib
import re

from setuptools import find_packages, setup

with open(
    pathlib.Path(__file__).parent / "draugr" / "__init__.py", "r"
) as project_init_file:
    content = project_init_file.read()
    # get version string from module
    version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", content, re.M).group(1)

    project_name = re.search(
        r"PROJECT_NAME = ['\"]([^'\"]*)['\"]", content, re.M
    ).group(1)

__author__ = "cnheider"


class DraugrPackage:
    @property
    def test_dependencies(self) -> list:
        path = pathlib.Path(__file__).parent
        requirements_tests = []
        with open(path / "requirements_tests.txt") as f:
            requirements = f.readlines()

            for requirement in requirements:
                requirements_tests.append(requirement.strip())

        return requirements_tests

    @property
    def setup_dependencies(self) -> list:
        path = pathlib.Path(__file__).parent
        requirements_setup = []
        with open(path / "requirements_setup.txt") as f:
            requirements = f.readlines()

            for requirement in requirements:
                requirements_setup.append(requirement.strip())

        return requirements_setup

    @property
    def package_name(self) -> str:
        return project_name

    @property
    def url(self) -> str:
        return "https://github.com/cnheider/draugr"

    @property
    def download_url(self) -> str:
        return self.url + "/releases"

    @property
    def readme_type(self) -> str:
        return "text/markdown"

    @property
    def packages(self):
        return find_packages(
            exclude=[
                # 'neodroid/environments'
            ]
        )

    @property
    def author_name(self):
        return "Christian Heider Nielsen"

    @property
    def author_email(self):
        return "christian.heider@alexandra.dk"

    @property
    def maintainer_name(self):
        return self.author_name

    @property
    def maintainer_email(self):
        return self.author_email

    @property
    def package_data(self):
        # data = glob.glob('environments/mab/**', recursive=True)
        return {
            # 'neodroid':[
            # *data
            # 'environments/mab/**',
            # 'environments/mab/**_Data/*',
            # 'environments/mab/windows/*'
            # 'environments/mab/windows/*_Data/*'
            #  ]
        }

    @property
    def entry_points(self):
        return {
            "console_scripts": [
                # "name_of_executable = module.with:function_to_execute"
            ]
        }

    @property
    def extras(self):

        path = pathlib.Path(__file__).parent
        requirements_writers = []
        with open(path / "requirements_writers.txt") as f:
            requirements = f.readlines()

            for requirement in requirements:
                requirements_writers.append(requirement.strip())

        these_extras = {
            # 'ExtraGroupName':['package-name; platform_system == "System(Linux,Windows)"'
            "writers": requirements_writers
        }

        all_dependencies = []

        for group_name in these_extras:
            all_dependencies += these_extras[group_name]
        these_extras["all"] = all_dependencies

        return these_extras

    @property
    def requirements(self) -> list:
        requirements_out = []
        with open("requirements.txt") as f:
            requirements = f.readlines()

            for requirement in requirements:
                requirements_out.append(requirement.strip())

        return requirements_out

    @property
    def description(self):
        return "A package for plotting directly in your terminal"

    @property
    def readme(self):
        with open("README.md") as f:
            return f.read()

    @property
    def keyword(self):
        with open("KEYWORDS.md") as f:
            return f.read()

    @property
    def license(self):
        return "Apache License, Version 2.0"

    @property
    def classifiers(self):
        return [
            "Development Status :: 4 - Beta",
            "Environment :: Console",
            "Intended Audience :: End Users/Desktop",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Natural Language :: English",
            # 'Topic :: Scientific/Engineering :: Artificial Intelligence'
            # 'Topic :: Software Development :: Bug Tracking',
        ]

    @property
    def version(self):
        return version


if __name__ == "__main__":

    pkg = DraugrPackage()

    setup(
        name=pkg.package_name,
        version=pkg.version,
        packages=pkg.packages,
        package_data=pkg.package_data,
        author=pkg.author_name,
        author_email=pkg.author_email,
        maintainer=pkg.maintainer_name,
        maintainer_email=pkg.maintainer_email,
        description=pkg.description,
        license=pkg.license,
        keywords=pkg.keyword,
        url=pkg.url,
        download_url=pkg.download_url,
        install_requires=pkg.requirements,
        extras_require=pkg.extras,
        setup_requires=pkg.setup_dependencies,
        entry_points=pkg.entry_points,
        classifiers=pkg.classifiers,
        long_description_content_type=pkg.readme_type,
        long_description=pkg.readme,
        tests_require=pkg.test_dependencies,
        include_package_data=True,
        python_requires=">=3",
    )
