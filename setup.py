from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.1'
DESCRIPTION = 'Poisoning Functionalities for your dataset'
LONG_DESCRIPTION = 'Long Description: Poisoning Functionalities for your dataset'

# Setting up
setup(
    name="seedpoisoner",
    version=VERSION,
    author="Prabhanjan Vinoda Bharadwaj",
    author_email="prabhanjan.vb03@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    # install_requires=['opencv-python', 'pyautogui', 'pyaudio'],
    keywords=['python', 'poison', 'badcode', 'codesearch', 'defense', 'attack'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)