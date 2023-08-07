# MLtools_BOULDERING

(last updated the 7th of August 2023).

MLtools is a package that gathers a collection of tools/function to 

This GitHub repository is written following the functional programming paradigm. 

## To do

- [ ] Provide Zenodo link to inputs and MLtools repos. 

## Installation

Create a new environment if wanted. In order for MLtools to work you need to install rastertools, shptools, Pytorch, Detectron2 (https://github.com/facebookresearch/detectron2) and then all the dependencies in the MLtools repository (found in requirements.txt). 

### Install rastertools

```bash
git clone https://github.com/astroNils/rastertools.git
cd rastertools
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps rastertools_BOULDERING
pip install -r requirements.txt
```

### Install shptools

```bash
git clone https://github.com/astroNils/shptools.git
cd shptools
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps shptools_BOULDERING
pip install -r requirements.txt
```

### Install Pytorch





### Install Detectron2





You should now have access to this module in Python. Note that if you have a windows machine 

```bash
python
```

```python
import MLtools
```

## Getting Started

A jupyter notebook is provided as a tutorial ([GET STARTED HERE](./resources/nb/GETTING_STARTED.ipynb)).

## Citing BOULDERING

If you use MLtools in your research or wish to refer to the baseline results published in TBD, please use the following citation entry.





