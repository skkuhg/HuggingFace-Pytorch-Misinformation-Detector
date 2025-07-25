"""
Setup script for Hugging Face PyTorch Misinformation Detector
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="huggingface-pytorch-misinformation-detector",
    version="1.0.0",
    author="skkuhg",
    author_email="ahczhg@gmail.com",
    description="AI-powered misinformation detection and counter-narrative generation using Hugging Face Transformers and PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/skkuhg/huggingface-pytorch-misinformation-detector",
    project_urls={
        "Bug Tracker": "https://github.com/skkuhg/huggingface-pytorch-misinformation-detector/issues",
        "Documentation": "https://github.com/skkuhg/huggingface-pytorch-misinformation-detector#readme",
        "Source Code": "https://github.com/skkuhg/huggingface-pytorch-misinformation-detector",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "gpu": [
            "torch>=1.13.0+cu118",
            "torchvision>=0.14.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "misinformation-detector=graphics_generator:run_demo",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    keywords=[
        "misinformation",
        "fact-checking",
        "transformers",
        "pytorch",
        "huggingface",
        "nlp",
        "ai",
        "machine-learning",
        "deep-learning",
        "counter-narratives",
        "graphics-generation",
        "stable-diffusion",
    ],
    zip_safe=False,
)