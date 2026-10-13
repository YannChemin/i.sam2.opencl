# i.sam2.opencl

GRASS GIS module that segments a 3-band imagery group with Meta's Segment
Anything Model 2 (SAM 2 and SAM 2.1) on an OpenCL device (GPU or CPU) and
writes the segments as a clean vector area map. The whole image path of
the network (Hiera image encoder, FPN neck, prompt encoder and mask
decoder) runs in OpenCL and reads the original PyTorch checkpoint
directly: neither Python nor PyTorch is needed at run time. See
`i.sam2.opencl.md` for the full manual.

This is the SAM 2 counterpart of *i.sam.opencl*, which runs the original
SAM (ViT-H/L/B). The two modules are independent.

## SAM 2 weights

The weights are not shipped with the module: even the smallest
checkpoint exceeds GitHub's 100 MB file limit. The module downloads the
SAM 2.1 checkpoint it needs on first use into the central folder
`$SAM_MODELS_DIR/sam2` (by default `~/models/sam/sam2`), or into
**weights_dir**; `-n` disables this. They can also be fetched beforehand
from Meta (Apache 2.0 licence). SAM 2.1 is the improved release; the SAM 2
files work too.

| Model | SAM 2.1 file | Size (bytes) | SAM 2 file | Size (bytes) |
| ----- | ------------ | ------------ | ---------- | ------------ |
| Hiera-L (best quality) | [`sam2.1_hiera_large.pt`](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) | 898 083 611 | [`sam2_hiera_large.pt`](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt) | 897 952 466 |
| Hiera-B+ | [`sam2.1_hiera_base_plus.pt`](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt) | 323 606 802 | [`sam2_hiera_base_plus.pt`](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt) | 323 493 298 |
| Hiera-S | [`sam2.1_hiera_small.pt`](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) | 184 416 285 | [`sam2_hiera_small.pt`](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt) | 184 309 650 |
| Hiera-T (fastest) | [`sam2.1_hiera_tiny.pt`](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) | 156 008 466 | [`sam2_hiera_tiny.pt`](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt) | 155 906 050 |

For example, all SAM 2.1 models into the central folder:

```sh
mkdir -p ~/models/sam/sam2
cd ~/models/sam/sam2
for v in large base_plus small tiny; do
    curl -L -O https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_$v.pt
done
```

The checkpoint is given to the module either as a file or as the folder
holding the downloaded files:

```sh
# One checkpoint file:
i.sam2.opencl group=rgb checkpoint=~/models/sam/sam2/sam2.1_hiera_large.pt output=segments

# The central folder: the best model found there is used (large, then
# base_plus, small, tiny), or the one chosen with model=, downloaded if
# missing
i.sam2.opencl group=rgb output=segments
i.sam2.opencl group=rgb model=tiny output=segments

# Another folder, never downloading
i.sam2.opencl group=rgb weights_dir=/data/sam2 output=segments -n
```

In a folder, a checkpoint is a `.pt` (or `.pth`) file recognised by
`hiera_large`, `hiera_base_plus`, `hiera_small` or `hiera_tiny` in its
name; SAM 2.1 files sort before SAM 2 files of the same model. The model
is always checked against the tensor shapes inside the file.

## Building

```sh
make MODULE_TOPDIR=$HOME/dev/grass
```

or install into the user's addon directory with
`g.extension extension=i.sam2.opencl url=/path/to/i.sam2.opencl`.
Requires an OpenCL 1.2 platform and its development headers (e.g.
`ocl-icd-opencl-dev`), libcurl (`libcurl4-openssl-dev`) and a GRASS GIS
build with OpenCL support. With Mesa on AMD GPUs, export `RUSTICL_ENABLE=radeonsi` (or the matching driver) to
expose the GPU through rusticl.

## Tests

```sh
I_SAM2_OPENCL_CHECKPOINT=~/models/sam/sam2/sam2.1_hiera_tiny.pt pytest tests
```

Without `I_SAM2_OPENCL_CHECKPOINT`, only the input validation tests run.

`tests/reference_compare.py` compares the module's debug dumps (written
when `I_SAM2_OPENCL_DUMP` names a directory) with the PyTorch `sam2`
package; it needs `torch` and `sam2`, which the module itself does not.
