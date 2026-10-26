/* SPDX-License-Identifier: Unlicense */

#ifndef I_SAM2_OPENCL_SAM2_MODEL_H
#define I_SAM2_OPENCL_SAM2_MODEL_H

#include "ocl_backend.h"
#include "pth_loader.h"

#define SAM_IMG 1024   /* Encoder input side. */
#define SAM_PATCH 256  /* Patch grid side after the stride-4 stem. */
#define SAM_GRID 64    /* Image embedding side (stride 16). */
#define SAM_DIM 256    /* FPN, prompt and decoder embedding width. */
#define SAM_LOWRES 256 /* Low-resolution mask side. */
#define SAM_STAGES 4   /* Hiera stages. */

struct sam_attn_w {
    cl_mem qw, qb, kw, kb, vw, vb, ow, ob;
    int dint;
};

/* One Hiera MultiScaleBlock. */
struct hiera_block {
    int dim, dim_out, heads;
    int window; /* Window side, 0 for global attention. */
    int pool;   /* 2x2 max pooling of the queries and the shortcut. */
    int stage;
    int side; /* Token grid side at the block input. */
    cl_mem n1w, n1b, qkvw, qkvb, projw, projb, n2w, n2b;
    cl_mem l1w, l1b, l2w, l2b;
    cl_mem dw, db; /* Shortcut projection when dim != dim_out. */
};

struct sam_dec_layer {
    struct sam_attn_w self, t2i, i2t;
    cl_mem n1w, n1b, n2w, n2b, n3w, n3b, n4w, n4b;
    cl_mem m1w, m1b, m2w, m2b;
};

struct sam_model {
    struct ocl_backend *ocl;
    const char *variant;
    int embed, depth;
    int stage_end[SAM_STAGES], stage_dim[SAM_STAGES];

    /* Hiera trunk. */
    cl_mem pe_w, pe_b, pos;
    struct hiera_block *blocks;

    /* FPN neck lateral convolutions (index = stage), the decoder's
     * high-resolution projections and the no-memory embedding added to
     * the image embedding. */
    cl_mem fpn_w[SAM_STAGES], fpn_b[SAM_STAGES];
    cl_mem s0w, s0b, s1w, s1b;
    cl_mem no_mem;

    /* Prompt encoder (host side, tiny). */
    float gauss[2 * 128];
    float point_emb[4][SAM_DIM];
    float not_a_point[SAM_DIM];
    float no_mask[SAM_DIM];

    /* Mask decoder. */
    struct sam_dec_layer layers[2];
    struct sam_attn_w final_t2i;
    cl_mem nfw, nfb;
    cl_mem up0w, up0b, up1w, up1b, up3w, up3b;
    float obj_token[SAM_DIM], iou_token[SAM_DIM], mask_tokens[4 * SAM_DIM];
    float *hyper_w[4][3], *hyper_b[4][3];
    float *iou_w[3], *iou_b[3];

    cl_kernel k_gemm, k_gemm_big, k_attn_few, k_attn_many, k_ln, k_softmax,
        k_stem, k_add, k_wpart, k_wunpart, k_pool, k_up2, k_convt, k_logits,
        k_post;
    /* CPU GEMMs (profile=cpu), and the GEMMs reading half-precision
     * weights (precision=fp16; NULL otherwise). */
    cl_kernel k_gemm_cpu_t, k_gemm_cpu_n;
    cl_kernel k_gemm_h, k_gemm_big_h, k_gemm_cpu_t_h, k_gemm_cpu_n_h;

    /* Encoder work buffers. */
    long score_cap; /* Floats of the attention score buffer. */
    cl_mem img, x, xn, xw, qkv, qp, scores, attn, tmp, mlpbuf;
    cl_mem emb, feat_s0, feat_s1;

    /* Per-image decoder caches: src = embedding + no-mask embedding,
     * dense positional encoding, and the layer-0 projections that do
     * not depend on the prompt. */
    cl_mem src, pe_img, srcpe, kp0, vp0, qp0;

    /* Decoder batch buffers, sized for batch prompts. */
    int batch;
    cl_mem qpe, qry, qin, sq, sk, sv, so, sh;
    cl_mem keys, kin, bq, bk, bv, bo;
    cl_mem u1, g2, u2, hyper, lowres;
    float *h_queries;

    /* Postprocessed binary masks and their statistics. */
    cl_mem mask, sel, stats;
    size_t mask_cap;
};

/* Hiera configurations of the published SAM 2 / 2.1 checkpoints. Head
 * counts, window sizes and global attention blocks are not stored in a
 * checkpoint; a variant is identified by its trunk width and depth. */
struct sam_variant {
    const char *name;
    int embed, depth, heads;
    int global[3];
    int window[SAM_STAGES];
};

extern const struct sam_variant sam_variants[];
extern const int sam_nvariants;

/* Variant of a SAM 2 checkpoint; NULL if pf holds no Hiera trunk, fatal
 * if it holds one of an unknown configuration. */
const struct sam_variant *sam_detect_variant(const struct pth_file *pf);

/* Load weights from a SAM 2 or SAM 2.1 checkpoint (Hiera tiny, small,
 * base_plus or large, detected from tensor shapes) onto the device and
 * allocate work buffers for decoding batch prompts at a time. */
void sam_load(struct sam_model *m, struct ocl_backend *ocl,
              const struct pth_file *pf, int batch);

/* Encode a normalized 3 x 1024 x 1024 CHW image and prepare the
 * prompt-independent decoder caches. */
void sam_encode(struct sam_model *m, const float *img);

/* Copy the encoder outputs to host, token-major: the 64 x 64 x 256
 * image embedding (no-memory embedding included), and the 256 x 256 x
 * 32 and 128 x 128 x 64 high-resolution features. */
void sam_read_features(struct sam_model *m, float *emb, float *s0, float *s1);

/* Decode P <= batch single-point prompts, points given in encoder
 * input pixel coordinates (x, y pairs). Writes the three multimask IoU
 * predictions per prompt to iou[P * 3] and leaves the low-resolution
 * logits on the device. */
void sam_decode(struct sam_model *m, int P, const float *points, float *iou);

/* Postprocess n selected low-resolution masks (index p * 3 + k) to an
 * out_h x out_w crop. stats receives 6 ints per mask: count(logit >
 * offset), count(logit > -offset), xmin, ymin, xmax, ymax (xmax < 0 if
 * empty). */
void sam_postprocess(struct sam_model *m, int n, const int *sel, int out_h,
                     int out_w, float offset, int *stats);

/* Read rows y0..y1 and columns x0..x1 (inclusive) of postprocessed mask
 * k into dst (row-major, (x1 - x0 + 1) wide). */
void sam_read_mask(struct sam_model *m, int k, int out_h, int out_w, int x0,
                   int y0, int x1, int y1, unsigned char *dst);

void sam_free(struct sam_model *m);

#endif /* I_SAM2_OPENCL_SAM2_MODEL_H */
