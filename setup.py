from setuptools import setup, find_packages

setup(
    name="anpr",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "ultralytics==8.0.114",
        "opencv-python==4.7.0.72",
        "numpy==1.24.3",
        "easyocr==1.7.0",
        "filterpy==1.4.5",
        "pandas==2.0.2",
        "scipy==1.10.1",
        "matplotlib==3.7.1",
        "seaborn==0.12.2"
    ],
    extras_require={
        "dev": [
            "pytest==7.3.1",
            "pytest-cov==4.0.0",
            "black==23.3.0",
            "flake8==6.0.0",
            "mypy==1.3.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "anpr=anpr.core.main:main"
        ]
    },
    python_requires=">=3.8"
)
