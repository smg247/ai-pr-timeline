from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-pr-timeline",
    version="0.1.0",
    author="AI PR Timeline Team",
    author_email="",
    description="AI plugin for predicting pull request merge times using machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ai-pr-timeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Version Control :: Git",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ai-pr-timeline-train=examples.train_model:main",
            "ai-pr-timeline-predict=examples.predict_pr:main",
            "ai-pr-timeline-batch=examples.batch_predict:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 