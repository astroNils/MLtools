# rastertools_BOULDERING

(last updated the 19th of June 2023).

Rastertools is a package that gathers a collection of tools to manipulate rasters and extract metadata from them. A large portion of this package was written for  the pre- and post-processing of planetary images so that it can be easily ingested in deep learning algorithms. Because of that, some of the functions are a bit, sometimes, too specific and repetitive (sorry for that!). I will try over time to improve this GitHub repository. Please contribute if you are interested. 

This GitHub repository is written following the functional programming paradigm. 

## To do

- [ ] Define coordinate systems (in `crs.py`) with different lonlat range (-180, 180) and (0, 360). Right now, there are some issues due to this problem. 
- [ ] `raster.projection` is currently not working for projection from Equirectangular to Moon2000. 
- [ ] Clean functions in `convert.py`.
- [ ] Global cleaning of the documentation of functions.

## Installation

Create a new environment if wanted. Then you can install the rastertools by writing the following in your terminal. 

```bash
git clone https://github.com/yellowchocobo/rastertools.git
cd rastertools
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps rastertools_BOULDERING
pip install -r requirements.txt
```

You should now have access to this module in Python.

```bash
python
```

```python
import rastertools_BOULDERING.raster as raster
import rastertools_BOULDERING.metadata as raster_metadata
import rastertools_BOULDERING.convert as raster_convert
import rastertools_BOULDERING.misc as raster_misc
```

## Getting Started

A jupyter notebook is provided as a tutorial ([GET STARTED HERE](./resources/nb/GETTING_STARTED.ipynb)).





