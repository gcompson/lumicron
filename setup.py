from setuptools import setup, find_packages

setup(
    name='lumicron',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'opencv-contrib-python-headless', # Optimized for Apple Silicon / OpenCL
        'pandas',      
        'matplotlib',  
        'fpdf2',       
        'tqdm',        
        'pyyaml',      
        'streamlit',   
    ],
    entry_points={
        'console_scripts': [
            'lumicron = lumicron:main', # Points directly to def main() in __init__.py
        ],
    },
    python_requires='>=3.9',
)
