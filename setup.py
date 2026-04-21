"""
setup.py
Quick installer — run: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="multimodal_emotion_recognition",
    version="1.0.0",
    description="Multimodal Negative Emotion Recognition using EEG and Facial Analysis",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.2",
        "opencv-python>=4.8.0",
        "Pillow>=10.1.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "plotly>=5.18.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "tensorboard>=2.15.0",
        "loguru>=0.7.2",
        "streamlit>=1.29.0",
        "optuna>=3.4.0",
        "shap>=0.44.0",
        "omegaconf>=2.3.0",
        "einops>=0.7.0",
    ],
    entry_points={
        "console_scripts": [
            "emotion-recognize=main:main",
        ]
    },
)
