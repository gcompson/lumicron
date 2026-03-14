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
        'pandas',      # Required for smear_audit.csv 
        'matplotlib',  # Required for light curves 
        'fpdf2',       # Required for forensic briefs 
        'tqdm',        # Required for progress bars 
        'pyyaml',      # Required for config management 
    ],
    entry_points={
        'console_scripts': [
            'lumicron = lumicron:main',
        ],
    },
    python_requires='>=3.9',
)
