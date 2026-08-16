from setuptools import setup, find_packages

setup(
    name="epd_research",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "xgboost",
        "joblib",
        "numpy"
    ],
)
