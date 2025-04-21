#!/usr/bin/env python3
"""
Setup script for trading_llm package
"""

import os
from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), "r") as f:
    requirements = f.read().splitlines()

# Read long description from README
with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as f:
    long_description = f.read()

setup(
    name="trading_llm",
    version="0.1.0",
    description="Generate natural language explanations for RL trading decisions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/trading-llm",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    keywords="reinforcement-learning, language-models, trading, explainability, finance",
    entry_points={
        "console_scripts": [
            "trading-llm=trading_llm.train_llm:main",
        ],
    },
) 