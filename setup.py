from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='project-biomassters',
      version="0.0.1",
      description="Project BioMassters Model (api_pred)",
      #license="MIT",
      author="Le Wagon group",
      author_email="sofia.m.matias@gmail.com",
      url="https://github.com/sofiammatias/project-biomassters",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
