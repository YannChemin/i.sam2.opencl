## DESCRIPTION

*i.sam2.opencl* segments a 3-band imagery group with the Segment
Anything Model 2 (SAM 2) and writes the segments as a clean vector area
map. The whole image path of the network, Hiera image encoder, feature
pyramid neck, prompt encoder and mask decoder, runs on an OpenCL device
(GPU or CPU), with weights read directly from the original PyTorch
checkpoint file: neither Python nor PyTorch is needed at run time.

The module implements SAM 2's automatic mask generation: the model is
prompted with a regular grid of points, and the masks it returns are
kept according to their predicted quality and stability, de-duplicated
and cleaned. The overlapping masks are then turned into a
non-overlapping, topologically clean vector map with one attribute row
per segment. SAM 2's video memory (memory encoder and memory attention)
is not used for single images and is not run.

The **group** (or its **subgroup**) must contain exactly three raster
maps, which are used as the red, green and blue channels in the order
they are listed in the group (see *i.group*). Each band is stretched
linearly between its lower and upper **percentile** to the 0-255 range
SAM 2 was trained on. Cells that are NULL in any band are not segmented.
Processing is done in the current computational region.

### Checkpoint

The **checkpoint** is one of the original model files published with
SAM 2.1 or SAM 2; the model is detected from the tensor shapes:

- Hiera-L:
  [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
  (857 MiB, best quality),
- Hiera-B+:
  [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
  (309 MiB),
- Hiera-S:
  [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
  (176 MiB),
- Hiera-T:
  [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
  (149 MiB, fastest).

The SAM 2 files of the same models (`sam2_hiera_large.pt` etc., in
`https://dl.fbaipublicfiles.com/segment_anything_2/072824/`) are
supported too. Checkpoints of the original SAM (`sam_vit_h_4b8939.pth`
etc.) are refused: they belong to *i.sam.opencl*.

The weights are too large to be installed with the module. Without a
**checkpoint** file, the module looks for one in the weights folder:
**weights_dir**, by default the central folder `$SAM_MODELS_DIR/sam2`, or
`~/models/sam/sam2` when the environment variable `SAM_MODELS_DIR` is not
set. If the model is not there, its SAM 2.1 checkpoint is downloaded into
that folder (created if needed) on first use and reused afterwards: the
model chosen with **model**, or Hiera-L when the folder holds no
checkpoint at all. The download is checked against the published size,
resumes where an interrupted one stopped, and waits for another run
downloading the same file. The **-n** flag forbids downloading: a missing
checkpoint is then an error.

A checkpoint is recognised in the weights folder by
`hiera_large`, `hiera_base_plus`, `hiera_small` or `hiera_tiny` in its
`.pt` (or `.pth`) file name, so renamed copies such as
`my_sam2.1_hiera_large.pt` are found too. The best model present is used
(large, then base_plus, small, tiny), or the one chosen with **model**;
the module checks that the file really holds that model. If a folder
holds several files of the chosen model, the first in alphabetical order
is used, with a warning; SAM 2.1 files (`sam2.1_*`) sort before SAM 2
files (`sam2_*`).

### Mask generation

The region is processed in crops. A region no larger than **tile_size**
cells is one crop, a larger region is split into tiles of **tile_size**
cells overlapping by **tile_overlap** cells. With **crop_layers** greater
than 0, each tile is additionally segmented in zoomed-in sub-crops (4 in
layer 1, 16 in layer 2, ...), prompted with **points_per_side** divided
by **crop_points_downscale** per layer.

Each crop is resized to 1024 x 1024 pixels, regardless of its aspect
ratio, as SAM 2 does, and passed through the image encoder. The mask
decoder is then prompted with a grid of **points_per_side** x
**points_per_side** points, **batch** prompts at a time, returning three
candidate masks per point. A mask is kept if its predicted IoU exceeds
**iou_threshold** and its stability score (the IoU of the mask
thresholded at plus and minus **stability_offset**) is at least
**stability_threshold**. Duplicates whose bounding boxes overlap by more
than **nms_threshold** (IoU) are suppressed, keeping the best one. Masks
touching an inner tile or crop edge are discarded, since the neighbouring
tile sees them whole. Holes and islands smaller than **minsize** cells
are finally removed from each mask.

These steps and their defaults are those of the reference
`SAM2AutomaticMaskGenerator`, with one difference: SAM 2's generator no
longer removes small holes and islands at full resolution (its
`min_mask_region_area` cleans the low-resolution logits with a CUDA-only
kernel instead, skipped without CUDA). *i.sam2.opencl* keeps the original
SAM full-resolution cleaning, controlled by **minsize**. With
**minsize**=0, it returns the same masks as the reference generator on
identical input.

### Vectorization and cleaning

SAM 2 masks overlap: a field and the parcels inside it can both be
masks. The **overlap** rule chooses which mask a cell belongs to: the
smallest covering mask (*small*, finest partition, the default), the
largest (*large*) or the one with the highest predicted IoU (*score*).
Masks larger than **maxsize** cells are discarded first; this is the way
to drop the single mask SAM 2 often returns for the whole background of
a scene.

The resulting label raster is cleaned: every connected piece smaller
than **minsize** cells, including small unsegmented gaps, is merged into
the neighbouring piece it shares the longest border with. Boundaries are
traced along cell edges into shared topological boundaries, simplified
with the Douglas-Peucker algorithm using the **simplify** tolerance (one
cell by default; 0 keeps the raster staircase), and the topology is
cleaned (boundaries broken at intersections, duplicates removed,
pseudo-nodes merged, slivers smaller than one cell removed). Each segment
gets one centroid, placed at the cell farthest from its border.

Areas without a centroid are gaps that no mask covers. The optional
**raster** output holds the segment categories of the cleaned label
raster before simplification.

### Attribute table

| Column        | Description                                        |
| ------------- | -------------------------------------------------- |
| cat           | Segment category                                   |
| mask          | Number of the SAM 2 mask the segment comes from    |
| predicted_iou | Mask quality predicted by SAM 2                    |
| stability     | Stability score of the mask                        |
| cells         | Segment size in cells                              |
| area          | Area of the simplified polygon in square map units |

A mask split into several pieces by overlapping masks yields several
segments with the same **mask** number.

## NOTES

### Differences from SAM

Compared with the original SAM (*i.sam.opencl*), SAM 2 has a
hierarchical Hiera image encoder (windowed attention at four
resolutions, with query pooling between them) and a feature pyramid
whose two finest levels are fed to the mask decoder to sharpen mask
edges. Its encoders are smaller and faster than SAM's ViT models of
similar quality. SAM 2 squashes each crop to 1024 x 1024 instead of
padding it, and its predicted IoU goes through a sigmoid, hence the lower
default **iou_threshold** (0.8 instead of 0.88).

### OpenCL device

With **device**=*auto*, the first GPU is used, else the first CPU
device. When several OpenCL platforms expose the same device (for
example Mesa's Clover and rusticl drivers for AMD GPUs), the first
platform on which the kernels compile is used; **platform** restricts
the choice to platforms whose name contains the given text. Mesa's
rusticl only exposes a GPU when enabled in the environment, e.g.
`RUSTICL_ENABLE=radeonsi`. A CPU OpenCL implementation such as PoCL
gives the same result, much more slowly.

The largest work buffers are those of the first stage of the image
encoder, at 256 x 256 tokens. With **batch**=32, the work buffers take
2330 MiB for Hiera-L and 1870 MiB for Hiera-T, in addition to
the weights; a smaller **batch** reduces the mask decoder's share. Buffers
are kept below the device's maximum allocation size.

### Device profile and precision

The **profile** chooses how buffers are allocated and which matrix
product kernels run, to suit the device:

- *cpu*: a device working in host memory, such as PoCL. Buffers are
  mapped instead of copied, and the matrix products use kernels without
  local memory that stream rows with 8-wide vector loads, which PoCL
  runs far better than the GPU kernels.
- *igpu*: a GPU sharing host memory (integrated graphics, APU). Buffers
  are allocated in host memory and written and read by mapping them
  (zero-copy), so no transfer crosses the memory bus twice.
- *dgpu*: a GPU with its own memory on internal PCI Express. Buffers
  live in device memory.
- *egpu*: a GPU behind a slow link, such as a Thunderbolt or USB4
  enclosure. Buffers live in device memory and large transfers go
  through a pinned staging buffer. The weights stay on the device for
  the whole run.

With **profile**=*auto* the profile is detected: *cpu* for a CPU
device, *igpu* for a GPU that reports unified memory, *egpu* for a GPU
whose PCI function Linux marks as removable (when the driver reports
its PCI address), *dgpu* otherwise. The chosen profile and the reason
are printed. A profile can be forced to compare them; every profile
gives the same results, only the speed differs.

**precision**=*fp16* stores the weight matrices of the linear layers
(most of the model's memory) as IEEE half-precision numbers, read with
the core OpenCL `vload_half` functions, so no half-precision extension
is needed; the arithmetic stays in single precision. It halves their
memory and the bytes read by the matrix products, which matters on
devices limited by memory bandwidth (integrated GPUs, CPUs). Rounding
the weights to half precision changes the masks slightly; check the
result against **precision**=*fp32* (the default) on a sample before
relying on it.

### Performance

On a CPU (Intel Core i7-1165G7, 4 cores, PoCL), the image encoder took
46 s for Hiera-T, 116 s for Hiera-B+ and 331 s for Hiera-L per crop, and
the mask decoder about 31 s per 64 prompts. A GPU is expected to be one
to two orders of magnitude faster. The number of crops, and so the
run time, grows with the region size divided by **tile_size**, and with
**crop_layers**.

### Choosing the resolution

SAM 2 sees every crop at 1024 x 1024 pixels. Objects should span at
least ten or so pixels at that scale: set the region resolution and
**tile_size** accordingly. A smaller **tile_size** zooms in (crops are
upsampled to 1024 pixels), which finds smaller objects at the cost of
more crops. Square tiles avoid distorting objects: a crop much longer
than wide is squashed to a square.

### Debugging aids

When the environment variable `I_SAM2_OPENCL_DUMP` names a directory,
the 8-bit crops, the encoder inputs, the image embeddings, the
high-resolution features and the list of masks are written there. The
script `tests/reference_compare.py` in the source tree compares them
with the PyTorch reference implementation.
`I_SAM2_OPENCL_PROFILE=1` prints the time spent in each OpenCL kernel. Each line gives the event time and the wall-clock time (some drivers, such as rusticl on Intel iris, report wrong event times), and for the matrix products the achieved GFLOP/s, which shows whether a kernel is limited by compute or by memory bandwidth.
`I_SAM2_OPENCL_DOWNLOAD_URL` replaces the download location
(`https://dl.fbaipublicfiles.com/segment_anything_2/092824`), for a local
mirror.

## EXAMPLES

### Landsat scene (North Carolina sample dataset)

Segment a true-colour composite of Landsat 7 bands 3, 2 and 1, dropping
the background mask (the region has 475 x 527 cells):

```sh
g.region raster=lsat7_2002_30 -p
i.group group=lsat_rgb input=lsat7_2002_30,lsat7_2002_20,lsat7_2002_10
i.sam2.opencl group=lsat_rgb checkpoint=sam2.1_hiera_large.pt \
    output=lsat_segments maxsize=50000
v.db.select lsat_segments | head
```

### Weights folder

Use the fastest model for a quick look; it is downloaded into
`~/models/sam/sam2` the first time:

```sh
i.sam2.opencl group=lsat_rgb model=tiny output=lsat_quick maxsize=50000
```

### Agricultural parcels

Delineate fields in a 2.5 m VNIR image with bands 1-3 grouped as RGB,
keeping only segments of at least 0.1 ha (160 cells) and a label raster
alongside the vector map:

```sh
i.group group=vnir_rgb input=vnir.1,vnir.2,vnir.3
g.region raster=vnir.1 -p
i.sam2.opencl group=vnir_rgb checkpoint=sam2.1_hiera_large.pt \
    output=fields raster=fields minsize=160
```

## REFERENCES

Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H.,
Radle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K. V.,
Carion, N., Wu, C.-Y., Girshick, R., Dollar, P., Feichtenhofer, C., 2024.
SAM 2: Segment Anything in Images and Videos.
[arXiv:2408.00714](https://arxiv.org/abs/2408.00714)

Ryali, C., Hu, Y.-T., Bolya, D., Wei, C., Fan, H., Huang, P.-Y.,
Aggarwal, V., Chowdhury, A., Poursaeed, O., Hoffman, J., Malik, J., Li,
Y., Feichtenhofer, C., 2023. Hiera: A Hierarchical Vision Transformer
without the Bells-and-Whistles. International Conference on Machine
Learning (ICML). [arXiv:2306.00989](https://arxiv.org/abs/2306.00989)

## SEE ALSO

*[i.group](i.group.md), [i.sam.opencl](i.sam.opencl.md),
[i.segment](i.segment.md), [r.to.vect](r.to.vect.md),
[v.clean](v.clean.md), [v.generalize](v.generalize.md)*

## AUTHORS

Yann Chemin
