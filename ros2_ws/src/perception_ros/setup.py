from glob import glob

from setuptools import setup

package_name = "perception_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "carla_bridge = perception_ros.carla_bridge_node:main",
            "perception = perception_ros.perception_node:main",
            "aeb = perception_ros.aeb_node:main",
        ],
    },
)
