"""
CoT-Evo: Evolutionary Distillation of Chain-of-Thought for Scientific Reasoning
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="cot-evo",
    version="0.1.0",
    author="CoT-Evo Team",
    author_email="your-email@example.com",
    description="Evolutionary Distillation of Chain-of-Thought for Scientific Reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/CoT-Evo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.11.0",
            "mypy>=1.5.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
        ],
        "embedding": [
            "vllm>=0.8.3",
            "transformers>=4.56.1",
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cot-evo=cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cot_evo": ["config/*.yaml", "prompts/*.txt"],
    },
)
