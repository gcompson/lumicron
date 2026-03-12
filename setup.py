from setuptools import setup

setup(
    name='lumicron',
    version='1.0.0',
    # Points to the 'core' folder as the source of the package
    package_dir={'': 'core'},
    py_modules=['lumicron'],
    install_requires=[
        'click',
        'pandas',
        'opencv-python',
        'numpy',
        'pyyaml',
        'tqdm',
        'matplotlib',
        'fpdf2'
    ],
    entry_points={
        'console_scripts': [
            'lumicron = lumicron:cli',
        ],
    },
    python_requires='>=3.9',
)
