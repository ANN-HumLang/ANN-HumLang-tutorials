from setuptools import find_packages, setup


requirements = [
    'torch',
    'scipy',
    'sparsemax',
    'numpy',
    'nltk',
    'jupyterlab',
    'transformers',
    'datasets',
    'ipywidgets',
    'pandas',
    'seaborn',
    'matplotlib',
    
]


setup(
    name="h",
    version="0.1.1",
    url="https://github.com/hcoxec/h_tensor",
    author="hcoxec",
    author_email="hcoxec@gmail.com",
    description="Entropy Estimation for Continuous Representations",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={"console_scripts": ["h=h.run:main"]},
)
