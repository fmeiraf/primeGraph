from setuptools import find_packages, setup  # type: ignore

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

setup(
  name="primeGraph",
  version="0.1.0",
  author="Fernando Meira Filho",
  author_email="fmeira.filho@gmail.com",
  description="A lightweight graph approach to LLM workflows.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/fmeiraf/primeGraph",
  packages=find_packages(exclude=["tests*"]),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  python_requires=">=3.11",
  install_requires=[
    "graphviz",
    "pydantic",
    "pytest",
    "rich",
    "psycopg2-binary",
    "pytest-asyncio",
    "fastapi",
    "uvicorn",
    "websockets",
  ],
)
