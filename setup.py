#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from DraugrPackage import DraugrPackage

__author__ = 'cnheider'

from setuptools import setup

if __name__ == '__main__':

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
      entry_points=pkg.entry_points,
      classifiers=pkg.classifiers,
      long_description_content_type=pkg.readme_type,
      long_description=pkg.readme,
      tests_require=pkg.test_dependencies,
      include_package_data=True,
      python_requires='>=3'
      )
