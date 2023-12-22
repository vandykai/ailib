import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ailib",
    version="0.0.1",
    author="vandykai",
    author_email="vandykai@gmail.com",
    description="AI lib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved ::Apache License::Version 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
        "PyYAML >= 5.4.1",
        "beautifulsoup4 >= 4.10.0",
        "dask >= 2021.6.2",
        "dill >= 0.3.4",
        "einops >= 0.3.2",
        "fastai >= 2.5.3",
        "fastprogress >= 1.0.0",
        "graphviz >= 0.17",
        "hyperopt >= 0.2.7",
        "ipython >= 7.29.0",
        "jieba >= 0.42.1",
        "jieba_fast >= 0.53",
        "matplotlib >= 3.4.3",
        "nltk >= 3.6.5",
        "numpy >= 1.20.3",
        "opencv_python >= 4.5.4.58",
        "pandas >= 1.3.4",
        "pytest >= 6.2.5",
        "requests >= 2.26.0",
        "retrying >= 1.3.3",
        "sacremoses >= 0.0.46",
        "scikit_learn >= 0.24.2",
        "scipy >= 1.7.2",
        "seaborn >= 0.11.2",
        "setuptools >= 58.0.4",
        "spacy >= 3.1.4",
        "tabulate >= 0.8.9",
        "torch >= 1.10.0",
        "tqdm >= 4.62.3",
        "transformers >= 4.12.5",
        "treelib >= 1.6.1",
        "xgboost >= 1.3.3"
    ]
)
