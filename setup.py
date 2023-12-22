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
        "PyYAML",
        "beautifulsoup4",
        "dask",
        "dill",
        "einops",
        "fastai",
        "fastprogress",
        "graphviz",
        "hyperopt",
        "ipython",
        "jieba",
        "jieba_fast",
        "matplotlib",
        "nltk",
        "numpy",
        "opencv_python",
        "pandas",
        "pytest",
        "requests",
        "retrying",
        "sacremoses",
        "scikit_learn",
        "scipy",
        "seaborn",
        "setuptools",
        "spacy",
        "tabulate",
        "torch",
        "tqdm",
        "transformers",
        "treelib",
        "xgboost"
    ]
)
