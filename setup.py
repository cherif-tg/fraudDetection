from setuptools import find_packages, setup


setup(
	name="fraud_detection",
	version="0.1.0",
	description="Fraud detection starter project",
	packages=find_packages(exclude=("tests", "notebooks")),
	python_requires=">=3.10",
)

