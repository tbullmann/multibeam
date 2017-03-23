# Multibeam

This repository contains the source code to play around with ZEISS multibeam data.
At the current state it is mostly for testing ideas.

## Get started

### Source code

Clone 'multibeam' from github and and change to the folder:

```bash
git clone http://github.com/tbullmann/multibeam/
cd multibeam
```
### Install requirements

Using conda to create an environment ```multibeam``` and install the requirements:
```bash
conda create -n multibeam python=2.7 pillow scipy h5py pyyaml scikit-image pandas pyramid
source activate multibeam
pip install tifffile
```

### Folders and data

Now make symlinks to the data:
``` bash
ln -s ~/iclound/data/mbeam data
ln -s ~/iclound/temp/mbeam temp
```

### PyCarm

Change project interpreter to ```multibeam``` environment.

