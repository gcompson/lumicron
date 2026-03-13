from setuptools import setup, find_packages

setup(
    name='lumicron',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'opencv-python',
    ],
    entry_points={
        'console_scripts': [
            'lumicron = lumicron:main',
        ],
    },
)
