from setuptools import setup, find_packages

setup(
    name="ip_camera",
    packages=['ip_camera'],
    version="1.00",
    description="ip camera read with a new thread",
    author="Nimrod Zhai",
    url="http://www.csdn.net",
    license="LGPL",
    scripts=["ip_camera/ip_camera_reader.py"]
)
