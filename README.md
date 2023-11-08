# MLtools_BOULDERING

(last updated the 8th of November 2023).

MLtools is a package that gathers a collection of tools/function to 

This GitHub repository is written following the functional programming paradigm. 

## To do

- [ ] The `is_tta` option in the prediction function is not working at the moment (ongoing debugging).
- [ ] A new training will be soon done (including 1000+ new image patches of the lunar surface). This should hopefully improve the performances of the model at detecting boulders. Many of those image patches are originating around cold spots on the lunar surface and therefore contains secondary craters. I hope adding this data will also help in decreasing the mixing-up of craters as boulders. 
- [ ] Several semantic segmentation models (and other instance segmentation models) will be tested in the next coming months. 
- [ ] A "leaderboard" of the performances of the different algorithms will be gathered in a markdown file. 

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

It's better to use Pytorch version 1.13.1 with Detectron2 (newer versions lead to some problem, at least this what I have experienced.). Go to the [Pytorch website, and the page where you have previous versions of Pytorch][https://pytorch.org/get-started/previous-versions/]. Here, be careful, you need to copy and paste the commands which corresponds to your OS, packaging system, if you have a graphical card or not and so forth... If you are lost, please check the "install pytorch" section on the Pytorch website. If you have a CPU, and you are installing pytorch with pip install, you need to paste the following command: 

```bash
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

If you use a server or a workstation with a GPU installed on it, I would write:

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

### Install Detectron2

```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
```

#### Install MLtools

```bash
git clone https://github.com/astroNils/MLtools.git
cd MLtools
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps MLtools_BOULDERING
pip install -r requirements.txt
```

You should now have access to this module in Python. Note that if you have a windows machine 

```bash
python
```

```python
from MLtools import inference
```

## Getting Started

A jupyter notebook is provided as a tutorial ([GET STARTED HERE](./resources/nb/GETTING_STARTED.ipynb)).

## Citing BOULDERING

If you use MLtools in your research or wish to refer to the baseline results published in TBD, please use the following citation entry.





