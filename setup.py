from setuptools import setup
import os

# HERE = pathlib.Path(__file__).parent
# README = (HERE / "README.md").read_text()
with open('README.md') as README:
    long_description = README.read()

setup(
    name="nrpylatex",
    version="1.0.3",
    description="LaTeX Interface to SymPy (CAS) for General Relativity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zachetienne/nrpylatex",
    author="Ken Sible",
    author_email="ksible@outlook.com",
    license="BSD License (BSD)",
    packages=["nrpylatex", "nrpylatex.tests"],
    install_requires=["sympy"],
    keywords=['General Relativity', 'LaTeX', 'CAS'],
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing :: Markup :: LaTeX',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Code Generators'
    ]
)
