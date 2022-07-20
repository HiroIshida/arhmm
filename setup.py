from setuptools import setup

setup_requires = []

install_requires = ["numpy", "scipy", 'dataclasses; python_version < "3.7"']

setup(
    name="arhmm",
    version="0.0.2",
    description="",
    license="MIT",
    install_requires=install_requires,
    package_data={"arhmm": ["py.typed"]},
)
