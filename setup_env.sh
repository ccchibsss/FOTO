from setuptools import setup, find_packages

setup(
    name='watermark_remover',       # Название вашего проекта
    version='0.1',
    description='Watermark removal with Mask R-CNN',
    author='Ваше Имя',              # Укажите ваше имя
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'streamlit',
        'opencv-python',
        'numpy',
        'pillow',
        'moviepy',
        'torchvision',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
