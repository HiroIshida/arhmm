from setuptools import find_packages, setup

setup_requires = []

install_requires = ["numpy", "scipy", 'dataclasses; python_version < "3.7"']

setup(
    name="arhmm",
    version="0.0.4",
    description="",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    license="MIT",
    install_requires=install_requires,
    packages=find_packages(exclude=("tests", "docs")),
    package_data={"arhmm": ["py.typed"]},
)
