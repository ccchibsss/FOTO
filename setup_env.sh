from setuptools import setup, find_packages

setup(
    name='watermark_remover',
    version='0.1',
    description='Watermark removal with Mask R-CNN',
    author='Ваше Имя',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0,<2.1',
        'torchvision>=0.8.0',
        'streamlit==1.24.0',
        'opencv-python==4.7.0.68',
        'numpy==1.23.5',
        'pillow',
        'moviepy',
        'rich==14.2.0',
        'pygments==2.19.2',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
