/****************************************************************************
 *
 * MODULE:       i.sam2.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Segment Anything 2 (SAM 2) image encoder (Hiera trunk and
 *               FPN neck) and mask decoder forward passes on OpenCL,
 *               weights read directly from the original PyTorch
 *               checkpoint.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "progress.h"
#include "pth_loader.h"
#include "sam2_model.h"

#define ACT_NONE 0
#define ACT_GELU 1
#define ACT_RELU 2

#define ENC_EPS 1e-6f /* Hiera LayerNorm(eps=1e-6) and LayerNorm2d. */
#define DEC_EPS 1e-5f /* nn.LayerNorm default, decoder transformer. */

/* Object score token, IoU token, 4 mask tokens, point, padding point. */
#define NTOK 8
#define DEC_HEADS 8

#define STEM_K 147 /* 3 x 7 x 7 stem patch. */

const struct sam_variant sam_variants[] = {
    {"large", 144, 48, 2, {23, 33, 43}, {8, 4, 16, 8}},
    {"base_plus", 112, 24, 2, {12, 16, 20}, {8, 4, 14, 7}},
    {"small", 96, 16, 1, {7, 10, 13}, {8, 4, 14, 7}},
    {"tiny", 96, 12, 1, {5, 7, 9}, {8, 4, 14, 7}},
};
const int sam_nvariants = sizeof(sam_variants) / sizeof(sam_variants[0]);

static int count_blocks(const struct pth_file *pf)
{
    char name[128];
    int n;

    for (n = 0;; n++) {
        snprintf(name, sizeof(name),
                 "image_encoder.trunk.blocks.%d.norm1.weight", n);
        if (!pth_find(pf, name))
            return n;
    }
}

const struct sam_variant *sam_detect_variant(const struct pth_file *pf)
{
    const struct pth_tensor *t = pth_find(pf, "image_encoder.trunk.pos_embed");
    int depth, v;

    if (!t || t->ndim != 4)
        return NULL;
    depth = count_blocks(pf);
    for (v = 0; v < sam_nvariants; v++)
        if (sam_variants[v].embed == t->shape[1] &&
            sam_variants[v].depth == depth)
            return &sam_variants[v];
    G_fatal_error(_("Unsupported Hiera trunk in <%s>: width %ld, %d blocks "
                    "(SAM 2 tiny, small, base_plus and large are supported)"),
                  pf->path, t->shape[1], depth);
    return NULL;
}

/* Weight loading. */

static size_t loaded_bytes;

static cl_mem load_dev(struct sam_model *m, const struct pth_file *pf,
                       const char *name, int ndim, const long *shape)
{
    const struct pth_tensor *t = pth_require(pf, name, ndim, shape);
    float *h = pth_read_f32(pf, t);
    size_t n = pth_numel(t) * sizeof(float);
    cl_mem b = ocl_upload(m->ocl, h, n, name);

    loaded_bytes += n;
    G_free(h);
    return b;
}

static float *load_host(const struct pth_file *pf, const char *name, int ndim,
                        const long *shape)
{
    return pth_read_f32(pf, pth_require(pf, name, ndim, shape));
}

static void load_host_into(const struct pth_file *pf, const char *name,
                           int ndim, const long *shape, float *dst)
{
    const struct pth_tensor *t = pth_require(pf, name, ndim, shape);
    float *h = pth_read_f32(pf, t);

    memcpy(dst, h, pth_numel(t) * sizeof(float));
    G_free(h);
}

static cl_mem load_dev_fmt(struct sam_model *m, const struct pth_file *pf,
                           int ndim, const long *shape, const char *fmt, ...)
{
    char name[256];
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    return load_dev(m, pf, name, ndim, shape);
}

/* Linear weight matrices, used only as GEMM operand B: half with
 * precision=fp16. */
static cl_mem load_dev_w(struct sam_model *m, const struct pth_file *pf,
                         const char *name, int ndim, const long *shape)
{
    const struct pth_tensor *t = pth_require(pf, name, ndim, shape);
    float *h = pth_read_f32(pf, t);
    size_t n = pth_numel(t);
    cl_mem b = ocl_upload_weight(m->ocl, h, n, name);

    loaded_bytes += n * (m->ocl->fp16w ? 2 : sizeof(float));
    G_free(h);
    return b;
}

static cl_mem load_dev_w_fmt(struct sam_model *m, const struct pth_file *pf,
                             int ndim, const long *shape, const char *fmt, ...)
{
    char name[256];
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    return load_dev_w(m, pf, name, ndim, shape);
}

static void load_attn(struct sam_model *m, const struct pth_file *pf,
                      const char *prefix, int dint, struct sam_attn_w *a)
{
    long w_in[2] = {dint, SAM_DIM}, w_out[2] = {SAM_DIM, dint};
    long b_in[1] = {dint}, b_out[1] = {SAM_DIM};

    a->dint = dint;
    a->qw = load_dev_w_fmt(m, pf, 2, w_in, "%s.q_proj.weight", prefix);
    a->qb = load_dev_fmt(m, pf, 1, b_in, "%s.q_proj.bias", prefix);
    a->kw = load_dev_w_fmt(m, pf, 2, w_in, "%s.k_proj.weight", prefix);
    a->kb = load_dev_fmt(m, pf, 1, b_in, "%s.k_proj.bias", prefix);
    a->vw = load_dev_w_fmt(m, pf, 2, w_in, "%s.v_proj.weight", prefix);
    a->vb = load_dev_fmt(m, pf, 1, b_in, "%s.v_proj.bias", prefix);
    a->ow = load_dev_w_fmt(m, pf, 2, w_out, "%s.out_proj.weight", prefix);
    a->ob = load_dev_fmt(m, pf, 1, b_out, "%s.out_proj.bias", prefix);
}

static void release_attn(struct sam_attn_w *a)
{
    ocl_release(&a->qw);
    ocl_release(&a->qb);
    ocl_release(&a->kw);
    ocl_release(&a->kb);
    ocl_release(&a->vw);
    ocl_release(&a->vb);
    ocl_release(&a->ow);
    ocl_release(&a->ob);
}

/* Kernel launch helpers. */

#define SETARG(k, i, v) ocl_setarg(k, i, sizeof(v), &(v), #k)

/* Accumulated kernel times when profiling is enabled. */
static struct {
    const char *what;
    double ms, wall_ms, flops;
    long calls;
} prof[64];
static int nprof;
/* FLOPs of the next launch, set by gemm() for the profile. */
static double pending_flops;

static double now_ms(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static void run(struct sam_model *m, cl_kernel k, int dim, const size_t *g,
                const size_t *l, const char *what)
{
    cl_event ev;
    cl_ulong t0, t1;
    double w0 = 0.0, w1;
    int i;

    /* Wall time as well as event time: some drivers (rusticl on iris,
     * Mesa 25.0) return wrong event timestamps. */
    if (m->ocl->profile) {
        clFinish(m->ocl->queue);
        w0 = now_ms();
    }
    ocl_check(clEnqueueNDRangeKernel(m->ocl->queue, k, dim, NULL, g, l, 0, NULL,
                                     m->ocl->profile ? &ev : NULL),
              what);
    if (!m->ocl->profile) {
        pending_flops = 0.0;
        return;
    }
    clWaitForEvents(1, &ev);
    w1 = now_ms();
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0,
                            NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1,
                            NULL);
    clReleaseEvent(ev);
    for (i = 0; i < nprof && strcmp(prof[i].what, what) != 0; i++)
        ;
    if (i == nprof && nprof < 64)
        prof[nprof++].what = what;
    if (i < 64) {
        prof[i].ms += (t1 - t0) * 1e-6;
        prof[i].wall_ms += w1 - w0;
        prof[i].flops += pending_flops;
        prof[i].calls++;
    }
    pending_flops = 0.0;
}

static void run1d(struct sam_model *m, cl_kernel k, size_t n, const char *what)
{
    size_t l = 256, g = (n + l - 1) / l * l;

    run(m, k, 1, &g, &l, what);
}

/* One GEMM operand: buffer, element offset, leading dimension and the
 * two batch strides. */
struct opnd {
    cl_mem buf;
    cl_long off;
    cl_int ld;
    cl_long s0, s1;
};

static struct opnd mat(cl_mem buf, long off, int ld)
{
    struct opnd o = {buf, off, ld, 0, 0};

    return o;
}

static struct opnd bmat(cl_mem buf, long off, int ld, long s0, long s1)
{
    struct opnd o = {buf, off, ld, s0, s1};

    return o;
}

static void gemm(struct sam_model *m, int M, int N, int K, struct opnd A,
                 struct opnd B, int transb, struct opnd C, int nb, int nb1,
                 cl_mem bias, cl_mem R, long roff, float alpha, int act)
{
    cl_int cM = M, cN = N, cK = K, ct = transb, cnb1 = nb1, cact = act;
    cl_long croff = roff;
    cl_float calpha = alpha;
    size_t g[3], l[3] = {16, 16, 1};
    /* The 128 x 128 float4 kernel needs every A/B row start 16-byte
     * aligned; small or misaligned products use the generic kernel. */
    int big = M >= 128 && N >= 64 && K % 4 == 0 && A.ld % 4 == 0 &&
              A.off % 4 == 0 && A.s0 % 4 == 0 && A.s1 % 4 == 0 &&
              B.ld % 4 == 0 && B.off % 4 == 0 && B.s0 % 4 == 0 && B.s1 % 4 == 0;
    int tile = big ? 128 : 64;
    int half = m->ocl->fp16w && ocl_is_half(B.buf);
    int cpu = m->ocl->arch == OCL_ARCH_CPU;
    cl_kernel k;
    const char *what;

    if (cpu) {
        k = transb ? (half ? m->k_gemm_cpu_t_h : m->k_gemm_cpu_t)
                   : (half ? m->k_gemm_cpu_n_h : m->k_gemm_cpu_n);
        what = transb ? "gemm_cpu_t" : "gemm_cpu_n";
    }
    else if (big) {
        k = half ? m->k_gemm_big_h : m->k_gemm_big;
        what = "gemm_big";
    }
    else {
        k = half ? m->k_gemm_h : m->k_gemm;
        what = "gemm";
    }

    SETARG(k, 0, cM);
    SETARG(k, 1, cN);
    SETARG(k, 2, cK);
    SETARG(k, 3, A.buf);
    SETARG(k, 4, A.off);
    SETARG(k, 5, A.ld);
    SETARG(k, 6, A.s0);
    SETARG(k, 7, A.s1);
    SETARG(k, 8, B.buf);
    SETARG(k, 9, B.off);
    SETARG(k, 10, B.ld);
    SETARG(k, 11, B.s0);
    SETARG(k, 12, B.s1);
    SETARG(k, 13, ct);
    SETARG(k, 14, C.buf);
    SETARG(k, 15, C.off);
    SETARG(k, 16, C.ld);
    SETARG(k, 17, C.s0);
    SETARG(k, 18, C.s1);
    SETARG(k, 19, cnb1);
    SETARG(k, 20, bias);
    SETARG(k, 21, R);
    SETARG(k, 22, croff);
    SETARG(k, 23, calpha);
    SETARG(k, 24, cact);
    pending_flops = 2.0 * M * N * K * nb;
    if (cpu) {
        /* One work-item per 2 x 4 (transb) or 4 x 8 output block; the
         * work-group size is left to the driver. */
        g[0] = transb ? (size_t)(N + 3) / 4 : (size_t)(N + 7) / 8;
        g[1] = transb ? (size_t)(M + 1) / 2 : (size_t)(M + 3) / 4;
        g[2] = nb;
        run(m, k, 3, g, NULL, what);
        return;
    }
    g[0] = (size_t)((N + tile - 1) / tile) * 16;
    g[1] = (size_t)((M + tile - 1) / tile) * 16;
    g[2] = nb;
    run(m, k, 3, g, l, what);
}

/* Y[rows, out] = act(X[rows, in] W^T + b) (+ Y when residual). */
static void linear(struct sam_model *m, int rows, int in, int out, cl_mem X,
                   long xoff, cl_mem W, cl_mem b, cl_mem Y, long yoff, int act,
                   int residual)
{
    gemm(m, rows, out, in, mat(X, xoff, in), mat(W, 0, in), 1,
         mat(Y, yoff, out), 1, 1, b, residual ? Y : NULL, yoff, 1.0f, act);
}

/* Row LayerNorm: 256 items per row for wide rows, one 64-item wavefront
 * per row for the narrow ones. */
static void layernorm(struct sam_model *m, long rows, int C, cl_mem X, cl_mem Y,
                      cl_mem g, cl_mem b, float eps, int act)
{
    cl_kernel k = m->k_ln;
    cl_int cC = C, cact = act;
    cl_float ceps = eps;
    size_t ls = C >= 1024 ? 256 : 64, gs = rows * ls;

    SETARG(k, 0, X);
    SETARG(k, 1, Y);
    SETARG(k, 2, g);
    SETARG(k, 3, b);
    SETARG(k, 4, cC);
    SETARG(k, 5, ceps);
    SETARG(k, 6, cact);
    run(m, k, 1, &gs, &ls, "layernorm");
}

static void softmax_rows(struct sam_model *m, cl_mem S, long rows, int L)
{
    cl_kernel k = m->k_softmax;
    cl_int cL = L;
    size_t ls = L >= 256 ? 256 : 64, gs = rows * ls;

    SETARG(k, 0, S);
    SETARG(k, 1, cL);
    run(m, k, 1, &gs, &ls, "softmax_rows");
}

static void add_bcast(struct sam_model *m, cl_mem out, cl_mem a, cl_mem b,
                      long n, long bmod)
{
    cl_kernel k = m->k_add;
    cl_int cn = (cl_int)n, cb = (cl_int)bmod;

    SETARG(k, 0, out);
    SETARG(k, 1, a);
    SETARG(k, 2, b);
    SETARG(k, 3, cn);
    SETARG(k, 4, cb);
    run1d(m, k, n, "add_bcast");
}

/* 2x2 max pooling of nwin s x s grids, C channels from column xoff of
 * rows ldx floats wide, into out (C floats per row). */
static void maxpool2(struct sam_model *m, cl_mem X, int xoff, int ldx,
                     cl_mem out, int C, int s, int nwin)
{
    cl_kernel k = m->k_pool;
    cl_int cxoff = xoff, cldx = ldx, cC = C, cs = s;
    cl_int total = nwin * (s / 2) * (s / 2) * C;

    SETARG(k, 0, X);
    SETARG(k, 1, cxoff);
    SETARG(k, 2, cldx);
    SETARG(k, 3, out);
    SETARG(k, 4, cC);
    SETARG(k, 5, cs);
    SETARG(k, 6, total);
    run1d(m, k, total, "maxpool2");
}

static void window_partition(struct sam_model *m, cl_mem X, cl_mem out, int H,
                             int C, int ws, int nwx)
{
    cl_kernel k = m->k_wpart;
    cl_int cH = H, cC = C, cws = ws, cnwx = nwx;
    cl_int total = nwx * nwx * ws * ws * C;

    SETARG(k, 0, X);
    SETARG(k, 1, out);
    SETARG(k, 2, cH);
    SETARG(k, 3, cH);
    SETARG(k, 4, cC);
    SETARG(k, 5, cws);
    SETARG(k, 6, cnwx);
    SETARG(k, 7, total);
    run1d(m, k, total, "window_partition");
}

static void window_unpartition_add(struct sam_model *m, cl_mem X, cl_mem Wd,
                                   int H, int C, int ws, int nwx)
{
    cl_kernel k = m->k_wunpart;
    cl_int cH = H, cC = C, cws = ws, cnwx = nwx;

    SETARG(k, 0, X);
    SETARG(k, 1, Wd);
    SETARG(k, 2, cH);
    SETARG(k, 3, cH);
    SETARG(k, 4, cC);
    SETARG(k, 5, cws);
    SETARG(k, 6, cnwx);
    run1d(m, k, (size_t)H * H * C, "window_unpartition_add");
}

static void convt_scatter(struct sam_model *m, cl_mem G, cl_mem out,
                          cl_mem bias, cl_mem add, int P, int H, int W, int Co,
                          int act)
{
    cl_kernel k = m->k_convt;
    cl_int cH = H, cW = W, cCo = Co, cact = act;
    cl_int total = P * 4 * H * W * Co;

    SETARG(k, 0, G);
    SETARG(k, 1, out);
    SETARG(k, 2, bias);
    SETARG(k, 3, add);
    SETARG(k, 4, cH);
    SETARG(k, 5, cW);
    SETARG(k, 6, cCo);
    SETARG(k, 7, total);
    SETARG(k, 8, cact);
    run1d(m, k, total, "convt_scatter");
}

/* Multi-head attention core on already projected tensors (decoder, 8
 * heads). qp/kp/vp hold P prompts of Nq/Nk rows of dint channels with
 * per-prompt strides sq/sk/sv (0 = shared by all prompts); the
 * recombined heads are written to o as P * Nq rows of dint. Scores,
 * softmax and weighting are fused: per query for the 8 prompt-token
 * keys, per work-group for the 4096 image-token keys. */
static void mha(struct sam_model *m, int P, int Nq, int Nk, int dint, cl_mem qp,
                long sq, cl_mem kp, long sk, cl_mem vp, long sv, cl_mem o)
{
    cl_int c = dint / DEC_HEADS, cNq = Nq, cNk = Nk, cd = dint;
    cl_int csq = (cl_int)sq, csk = (cl_int)sk, csv = (cl_int)sv;
    cl_int total = P * DEC_HEADS * Nq;
    cl_float scale = 1.0f / sqrtf((float)c);
    int many = Nk > 64;
    cl_kernel k = many ? m->k_attn_many : m->k_attn_few;

    if (many && (c != 16 || Nk > 4096))
        G_fatal_error("mha: unsupported shape");
    SETARG(k, 0, qp);
    SETARG(k, 1, csq);
    SETARG(k, 2, kp);
    SETARG(k, 3, csk);
    SETARG(k, 4, vp);
    SETARG(k, 5, csv);
    SETARG(k, 6, o);
    SETARG(k, 7, cNq);
    SETARG(k, 8, cNk);
    SETARG(k, 9, cd);
    SETARG(k, 10, c);
    SETARG(k, 11, scale);
    SETARG(k, 12, total);
    if (many) {
        size_t gs = (size_t)total * 256, ls = 256;

        run(m, k, 1, &gs, &ls, "attn_manykeys");
    }
    else
        run1d(m, k, total, "attn_fewkeys");
}

/* Model loading. */

/* PyTorch's bicubic convolution coefficients (A = -0.75) for the
 * fractional offset t. */
static void cubic_coeffs(double t, double c[4])
{
    const double A = -0.75;
    double x;

    x = t + 1.0;
    c[0] = ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    x = t;
    c[1] = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    x = 1.0 - t;
    c[2] = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    x = 2.0 - t;
    c[3] = ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
}

/* Hiera._get_pos_embed for the 256 x 256 patch grid: the background
 * embedding (E x bh x bw) resized with F.interpolate(mode="bicubic",
 * align_corners=False), plus the window embedding (E x ws x ws) tiled
 * over the grid. Returned token-major, 256 * 256 x E. */
static float *hiera_pos_embed(const float *bg, int E, int bh, int bw,
                              const float *win, int ws)
{
    const int G = SAM_PATCH;
    float *out = G_malloc((size_t)G * G * E * sizeof(float));
    int *iy = G_malloc(4 * G * sizeof(int)),
        *ix = G_malloc(4 * G * sizeof(int));
    double *cy = G_malloc(4 * G * sizeof(double));
    double *cx = G_malloc(4 * G * sizeof(double));
    int y, x, c, i, j;

    for (i = 0; i < G; i++) {
        double ry = (double)bh / G * (i + 0.5) - 0.5;
        double rx = (double)bw / G * (i + 0.5) - 0.5;
        int fy = (int)floor(ry), fx = (int)floor(rx);

        cubic_coeffs(ry - fy, cy + 4 * i);
        cubic_coeffs(rx - fx, cx + 4 * i);
        for (j = 0; j < 4; j++) {
            int a = fy - 1 + j, b = fx - 1 + j;

            iy[4 * i + j] = a < 0 ? 0 : a > bh - 1 ? bh - 1 : a;
            ix[4 * i + j] = b < 0 ? 0 : b > bw - 1 ? bw - 1 : b;
        }
    }

#pragma omp parallel for private(x, c, i, j)
    for (y = 0; y < G; y++) {
        for (x = 0; x < G; x++) {
            float *o = out + ((size_t)y * G + x) * E;

            for (c = 0; c < E; c++) {
                const float *p = bg + (size_t)c * bh * bw;
                double v = 0.0;

                for (i = 0; i < 4; i++) {
                    const float *row = p + (size_t)iy[4 * y + i] * bw;
                    double h = 0.0;

                    for (j = 0; j < 4; j++)
                        h += cx[4 * x + j] * row[ix[4 * x + j]];
                    v += cy[4 * y + i] * h;
                }
                o[c] = (float)v + win[((size_t)c * ws + y % ws) * ws + x % ws];
            }
        }
    }
    G_free(iy);
    G_free(ix);
    G_free(cy);
    G_free(cx);
    return out;
}

static void load_encoder(struct sam_model *m, const struct pth_file *pf)
{
    const struct sam_variant *v = sam_detect_variant(pf);
    const char *p = "image_encoder.trunk.blocks";
    const struct pth_tensor *t;
    long E, s1[1], s2[2];
    char name[160];
    int i, j, stage, side, bh, bw, ws0;
    float *bg, *win, *pos;

    if (!v)
        G_fatal_error(_("<%s> is not a SAM 2 checkpoint"), pf->path);
    m->variant = v->name;
    m->embed = v->embed;
    m->depth = v->depth;
    E = v->embed;

    /* Stage layout: a block with a shortcut projection starts a stage,
     * doubles the width and pools the grid. */
    m->blocks = G_calloc(m->depth, sizeof(struct hiera_block));
    stage = 0;
    side = SAM_PATCH;
    for (i = 0; i < m->depth; i++) {
        struct hiera_block *b = &m->blocks[i];
        int prev_dim = i ? m->blocks[i - 1].dim_out : (int)E;

        snprintf(name, sizeof(name), "%s.%d.attn.qkv.weight", p, i);
        t = pth_require(pf, name, 2, (long[]){-1, prev_dim});
        snprintf(name, sizeof(name), "%s.%d.proj.weight", p, i);
        b->dim = prev_dim;
        b->dim_out = (int)(t->shape[0] / 3);
        b->pool = pth_find(pf, name) != NULL;
        if (b->pool != (b->dim_out != b->dim) ||
            (b->pool && b->dim_out != 2 * b->dim) || (b->pool && i == 0))
            G_fatal_error(_("Unexpected Hiera block %d in <%s>: width %d -> "
                            "%d"),
                          i, pf->path, b->dim, b->dim_out);
        if (b->pool) {
            m->stage_end[stage] = i - 1;
            m->stage_dim[stage] = b->dim;
            if (++stage >= SAM_STAGES)
                G_fatal_error(_("Hiera trunk in <%s> has more than %d stages"),
                              pf->path, SAM_STAGES);
        }
        b->stage = stage;
        b->side = side;
        if (b->pool)
            side /= 2;
        b->heads = v->heads << stage;
        if (b->dim_out % b->heads != 0)
            G_fatal_error(_("Hiera block %d: width %d not divisible by %d "
                            "heads"),
                          i, b->dim_out, b->heads);
        /* The first block of a stage still uses the window size of the
         * previous stage (on the grid before pooling). */
        b->window = v->window[b->pool ? stage - 1 : stage];
        for (j = 0; j < 3; j++)
            if (v->global[j] == i)
                b->window = 0;
    }
    if (stage != SAM_STAGES - 1)
        G_fatal_error(_("Hiera trunk in <%s> has %d stages, expected %d"),
                      pf->path, stage + 1, SAM_STAGES);
    m->stage_end[stage] = m->depth - 1;
    m->stage_dim[stage] = m->blocks[m->depth - 1].dim_out;
    G_message(_("SAM 2 Hiera %s image encoder: width %d, %d blocks, stages "
                "%d/%d/%d/%d"),
              m->variant, m->embed, m->depth, m->stage_end[0] + 1,
              m->stage_end[1] - m->stage_end[0],
              m->stage_end[2] - m->stage_end[1],
              m->stage_end[3] - m->stage_end[2]);

    m->pe_w = load_dev(m, pf, "image_encoder.trunk.patch_embed.proj.weight", 4,
                       (long[]){E, 3, 7, 7});
    m->pe_b = load_dev(m, pf, "image_encoder.trunk.patch_embed.proj.bias", 1,
                       (long[]){E});

    t = pth_require(pf, "image_encoder.trunk.pos_embed", 4,
                    (long[]){1, E, -1, -1});
    bh = (int)t->shape[2];
    bw = (int)t->shape[3];
    bg = pth_read_f32(pf, t);
    t = pth_require(pf, "image_encoder.trunk.pos_embed_window", 4,
                    (long[]){1, E, -1, -1});
    ws0 = (int)t->shape[2];
    if (t->shape[3] != ws0 || SAM_PATCH % ws0 != 0)
        G_fatal_error(_("Unsupported window position embedding in <%s>"),
                      pf->path);
    win = pth_read_f32(pf, t);
    pos = hiera_pos_embed(bg, (int)E, bh, bw, win, ws0);
    m->pos = ocl_upload(m->ocl, pos, (size_t)SAM_PATCH * SAM_PATCH * E * 4,
                        "position embedding");
    G_free(bg);
    G_free(win);
    G_free(pos);

    for (i = 0; i < m->depth; i++) {
        struct hiera_block *b = &m->blocks[i];
        long d = b->dim, o = b->dim_out, h;

        s1[0] = d;
        b->n1w = load_dev_fmt(m, pf, 1, s1, "%s.%d.norm1.weight", p, i);
        b->n1b = load_dev_fmt(m, pf, 1, s1, "%s.%d.norm1.bias", p, i);
        s2[0] = 3 * o;
        s2[1] = d;
        b->qkvw = load_dev_w_fmt(m, pf, 2, s2, "%s.%d.attn.qkv.weight", p, i);
        s1[0] = 3 * o;
        b->qkvb = load_dev_fmt(m, pf, 1, s1, "%s.%d.attn.qkv.bias", p, i);
        s2[0] = o;
        s2[1] = o;
        b->projw = load_dev_w_fmt(m, pf, 2, s2, "%s.%d.attn.proj.weight", p, i);
        s1[0] = o;
        b->projb = load_dev_fmt(m, pf, 1, s1, "%s.%d.attn.proj.bias", p, i);
        b->n2w = load_dev_fmt(m, pf, 1, s1, "%s.%d.norm2.weight", p, i);
        b->n2b = load_dev_fmt(m, pf, 1, s1, "%s.%d.norm2.bias", p, i);
        snprintf(name, sizeof(name), "%s.%d.mlp.layers.0.weight", p, i);
        h = pth_require(pf, name, 2, (long[]){-1, o})->shape[0];
        if (h != 4 * o)
            G_fatal_error(_("Hiera block %d: MLP width %ld, expected %ld"), i,
                          h, 4 * o);
        s2[0] = h;
        s2[1] = o;
        b->l1w = load_dev_w_fmt(m, pf, 2, s2, "%s.%d.mlp.layers.0.weight", p, i);
        s1[0] = h;
        b->l1b = load_dev_fmt(m, pf, 1, s1, "%s.%d.mlp.layers.0.bias", p, i);
        s2[0] = o;
        s2[1] = h;
        b->l2w = load_dev_w_fmt(m, pf, 2, s2, "%s.%d.mlp.layers.1.weight", p, i);
        s1[0] = o;
        b->l2b = load_dev_fmt(m, pf, 1, s1, "%s.%d.mlp.layers.1.bias", p, i);
        if (b->pool) {
            s2[0] = o;
            s2[1] = d;
            b->dw = load_dev_w_fmt(m, pf, 2, s2, "%s.%d.proj.weight", p, i);
            b->db = load_dev_fmt(m, pf, 1, s1, "%s.%d.proj.bias", p, i);
        }
    }

    /* FPN lateral convolutions: convs[0] reads the last (coarsest)
     * stage. */
    for (i = 0; i < SAM_STAGES; i++) {
        long c = m->stage_dim[i];

        m->fpn_w[i] = load_dev_fmt(m, pf, 4, (long[]){SAM_DIM, c, 1, 1},
                                   "image_encoder.neck.convs.%d.conv.weight",
                                   SAM_STAGES - 1 - i);
        m->fpn_b[i] = load_dev_fmt(m, pf, 1, (long[]){SAM_DIM},
                                   "image_encoder.neck.convs.%d.conv.bias",
                                   SAM_STAGES - 1 - i);
    }
    m->s0w = load_dev(m, pf, "sam_mask_decoder.conv_s0.weight", 4,
                      (long[]){SAM_DIM / 8, SAM_DIM, 1, 1});
    m->s0b = load_dev(m, pf, "sam_mask_decoder.conv_s0.bias", 1,
                      (long[]){SAM_DIM / 8});
    m->s1w = load_dev(m, pf, "sam_mask_decoder.conv_s1.weight", 4,
                      (long[]){SAM_DIM / 4, SAM_DIM, 1, 1});
    m->s1b = load_dev(m, pf, "sam_mask_decoder.conv_s1.bias", 1,
                      (long[]){SAM_DIM / 4});
    m->no_mem = load_dev(m, pf, "no_mem_embed", 3, (long[]){1, 1, SAM_DIM});
}

static void load_decoder(struct sam_model *m, const struct pth_file *pf)
{
    const long v256[1] = {SAM_DIM}, e1[2] = {1, SAM_DIM};
    const char *dec = "sam_mask_decoder";
    char name[160];
    int i, j;

    load_host_into(pf,
                   "sam_prompt_encoder.pe_layer.positional_encoding_"
                   "gaussian_matrix",
                   2, (long[]){2, SAM_DIM / 2}, m->gauss);
    for (i = 0; i < 4; i++) {
        snprintf(name, sizeof(name),
                 "sam_prompt_encoder.point_embeddings.%d.weight", i);
        load_host_into(pf, name, 2, e1, m->point_emb[i]);
    }
    load_host_into(pf, "sam_prompt_encoder.not_a_point_embed.weight", 2, e1,
                   m->not_a_point);
    load_host_into(pf, "sam_prompt_encoder.no_mask_embed.weight", 2, e1,
                   m->no_mask);
    load_host_into(pf, "sam_mask_decoder.obj_score_token.weight", 2, e1,
                   m->obj_token);
    load_host_into(pf, "sam_mask_decoder.iou_token.weight", 2, e1,
                   m->iou_token);
    load_host_into(pf, "sam_mask_decoder.mask_tokens.weight", 2,
                   (long[]){4, SAM_DIM}, m->mask_tokens);

    for (i = 0; i < 2; i++) {
        struct sam_dec_layer *L = &m->layers[i];
        const char *p = "sam_mask_decoder.transformer.layers";

        snprintf(name, sizeof(name), "%s.%d.self_attn", p, i);
        load_attn(m, pf, name, SAM_DIM, &L->self);
        snprintf(name, sizeof(name), "%s.%d.cross_attn_token_to_image", p, i);
        load_attn(m, pf, name, SAM_DIM / 2, &L->t2i);
        snprintf(name, sizeof(name), "%s.%d.cross_attn_image_to_token", p, i);
        load_attn(m, pf, name, SAM_DIM / 2, &L->i2t);
        L->n1w = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm1.weight", p, i);
        L->n1b = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm1.bias", p, i);
        L->n2w = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm2.weight", p, i);
        L->n2b = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm2.bias", p, i);
        L->n3w = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm3.weight", p, i);
        L->n3b = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm3.bias", p, i);
        L->n4w = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm4.weight", p, i);
        L->n4b = load_dev_fmt(m, pf, 1, v256, "%s.%d.norm4.bias", p, i);
        L->m1w = load_dev_w_fmt(m, pf, 2, (long[]){2048, SAM_DIM},
                              "%s.%d.mlp.layers.0.weight", p, i);
        L->m1b = load_dev_fmt(m, pf, 1, (long[]){2048},
                              "%s.%d.mlp.layers.0.bias", p, i);
        L->m2w = load_dev_w_fmt(m, pf, 2, (long[]){SAM_DIM, 2048},
                              "%s.%d.mlp.layers.1.weight", p, i);
        L->m2b = load_dev_fmt(m, pf, 1, v256, "%s.%d.mlp.layers.1.bias", p, i);
    }
    load_attn(m, pf, "sam_mask_decoder.transformer.final_attn_token_to_image",
              SAM_DIM / 2, &m->final_t2i);
    m->nfw = load_dev_fmt(m, pf, 1, v256,
                          "%s.transformer.norm_final_attn.weight", dec);
    m->nfb = load_dev_fmt(m, pf, 1, v256, "%s.transformer.norm_final_attn.bias",
                          dec);

    m->up0w = load_dev_fmt(m, pf, 4, (long[]){SAM_DIM, 64, 2, 2},
                           "%s.output_upscaling.0.weight", dec);
    m->up0b =
        load_dev_fmt(m, pf, 1, (long[]){64}, "%s.output_upscaling.0.bias", dec);
    m->up1w = load_dev_fmt(m, pf, 1, (long[]){64},
                           "%s.output_upscaling.1.weight", dec);
    m->up1b =
        load_dev_fmt(m, pf, 1, (long[]){64}, "%s.output_upscaling.1.bias", dec);
    m->up3w = load_dev_fmt(m, pf, 4, (long[]){64, 32, 2, 2},
                           "%s.output_upscaling.3.weight", dec);
    m->up3b =
        load_dev_fmt(m, pf, 1, (long[]){32}, "%s.output_upscaling.3.bias", dec);

    for (i = 0; i < 4; i++) {
        for (j = 0; j < 3; j++) {
            long ws[2] = {j == 2 ? 32 : SAM_DIM, SAM_DIM};
            long bs[1] = {ws[0]};

            snprintf(name, sizeof(name),
                     "%s.output_hypernetworks_mlps.%d.layers.%d.weight", dec, i,
                     j);
            m->hyper_w[i][j] = load_host(pf, name, 2, ws);
            snprintf(name, sizeof(name),
                     "%s.output_hypernetworks_mlps.%d.layers.%d.bias", dec, i,
                     j);
            m->hyper_b[i][j] = load_host(pf, name, 1, bs);
        }
    }
    for (j = 0; j < 3; j++) {
        long ws[2] = {j == 2 ? 4 : SAM_DIM, SAM_DIM};
        long bs[1] = {ws[0]};

        snprintf(name, sizeof(name), "%s.iou_prediction_head.layers.%d.weight",
                 dec, j);
        m->iou_w[j] = load_host(pf, name, 2, ws);
        snprintf(name, sizeof(name), "%s.iou_prediction_head.layers.%d.bias",
                 dec, j);
        m->iou_b[j] = load_host(pf, name, 1, bs);
    }
}

/* PositionEmbeddingRandom encoding of normalized (x, y) in [0, 1]. */
static void pe_encode(const struct sam_model *m, float x, float y, float *out)
{
    int i;

    x = 2.0f * x - 1.0f;
    y = 2.0f * y - 1.0f;
    for (i = 0; i < SAM_DIM / 2; i++) {
        float v = x * m->gauss[i] + y * m->gauss[SAM_DIM / 2 + i];

        v *= 2.0f * (float)M_PI;
        out[i] = sinf(v);
        out[SAM_DIM / 2 + i] = cosf(v);
    }
}

/* Attention geometry of a Hiera block: windows of side s (the whole
 * grid for global attention), nwx windows per side, queries pooled to
 * side sq. */
struct geom {
    int s, nwx, nwin, Nk, sq, Nq, out_side;
};

static struct geom block_geom(const struct hiera_block *b)
{
    struct geom g;

    g.s = b->window ? b->window : b->side;
    g.nwx = b->window ? (b->side + b->window - 1) / b->window : 1;
    g.nwin = g.nwx * g.nwx;
    g.Nk = g.s * g.s;
    g.sq = b->pool ? g.s / 2 : g.s;
    g.Nq = g.sq * g.sq;
    g.out_side = b->pool ? b->side / 2 : b->side;
    return g;
}

/* Heads whose score matrices fit at once in the score buffer. */
static int head_chunk(const struct sam_model *m, const struct hiera_block *b,
                      const struct geom *g)
{
    long per_head = (long)g->nwin * g->Nq * g->Nk;
    long hc = m->score_cap / per_head;

    return hc > b->heads ? b->heads : (int)hc;
}

#define GROW(v, n)                                                             \
    do {                                                                       \
        long n_ = (n);                                                         \
        if (n_ > (v))                                                          \
            (v) = n_;                                                          \
    } while (0)

static size_t work_bytes;

static cl_mem walloc(struct ocl_backend *ocl, size_t n, const char *what)
{
    work_bytes += n;
    return ocl_alloc(ocl, n, what);
}

static void alloc_buffers(struct sam_model *m)
{
    struct ocl_backend *ocl = m->ocl;
    const size_t f = sizeof(float);
    const long T = SAM_GRID * SAM_GRID, T0 = (long)SAM_PATCH * SAM_PATCH;
    const long P = m->batch;
    long nx = 0, nxn = 0, nxw = 0, nqkv = 0, nqp = 0, nattn = 0, ntmp = 0;
    long nmlp = T0 * STEM_K, nscores = 0;
    size_t cap;
    float *pe;
    int i, y, x;

    cap = ocl->max_alloc < ((size_t)512 << 20) ? ocl->max_alloc
                                               : ((size_t)512 << 20);
    m->score_cap = (long)(cap / f);

    GROW(nx, T0 * m->embed);
    for (i = 0; i < m->depth; i++) {
        const struct hiera_block *b = &m->blocks[i];
        const struct geom g = block_geom(b);
        const long Tin = (long)b->side * b->side;
        const long Tout = (long)g.out_side * g.out_side;
        const long rows = (long)g.nwin * g.Nk, qrows = (long)g.nwin * g.Nq;
        int hc = head_chunk(m, b, &g);

        if (hc < 1)
            G_fatal_error(
                _("OpenCL device cannot hold the attention scores "
                  "of one head of Hiera block %d (%lu MiB)"),
                i, (unsigned long)((size_t)g.nwin * g.Nq * g.Nk * f >> 20));
        GROW(nx, Tout * b->dim_out);
        GROW(nxn, Tin * b->dim);
        GROW(nxn, Tout * b->dim_out);
        if (b->window)
            GROW(nxw, rows * b->dim);
        GROW(nqkv, rows * 3 * b->dim_out);
        if (b->pool) {
            GROW(nqp, qrows * b->dim_out);
            GROW(ntmp, Tin * b->dim_out);
        }
        GROW(nattn, qrows * b->dim_out);
        GROW(ntmp, qrows * b->dim_out);
        GROW(nscores, (long)hc * g.nwin * g.Nq * g.Nk);
        GROW(nmlp, Tout * 4 * b->dim_out);
    }
    /* FPN scratch for the stage-1, stage-2 and stage-4 lateral maps. */
    GROW(nmlp, T0 * SAM_DIM);

    m->img = walloc(ocl, 3L * SAM_IMG * SAM_IMG * f, "image");
    m->x = walloc(ocl, nx * f, "tokens");
    m->xn = walloc(ocl, nxn * f, "normalized tokens");
    m->xw = walloc(ocl, (nxw ? nxw : 1) * f, "window partition");
    m->qkv = walloc(ocl, nqkv * f, "qkv");
    m->qp = walloc(ocl, (nqp ? nqp : 1) * f, "pooled queries");
    m->scores = walloc(ocl, nscores * f, "attention scores");
    m->attn = walloc(ocl, nattn * f, "attention output");
    m->tmp = walloc(ocl, ntmp * f, "projection");
    m->mlpbuf = walloc(ocl, nmlp * f, "mlp");
    m->emb = walloc(ocl, T * SAM_DIM * f, "embedding");
    m->feat_s0 = walloc(ocl, T0 * (SAM_DIM / 8) * f, "features s0");
    m->feat_s1 = walloc(ocl, T0 / 4 * (SAM_DIM / 4) * f, "features s1");

    m->src = walloc(ocl, T * SAM_DIM * f, "decoder src");
    m->srcpe = walloc(ocl, T * SAM_DIM * f, "decoder src+pe");
    m->kp0 = walloc(ocl, T * 128 * f, "decoder k0");
    m->vp0 = walloc(ocl, T * 128 * f, "decoder v0");
    m->qp0 = walloc(ocl, T * 128 * f, "decoder q0");

    /* Dense positional encoding of the 64 x 64 embedding grid. */
    pe = G_malloc(T * SAM_DIM * f);
    for (y = 0; y < SAM_GRID; y++)
        for (x = 0; x < SAM_GRID; x++)
            pe_encode(m, (x + 0.5f) / SAM_GRID, (y + 0.5f) / SAM_GRID,
                      pe + ((long)y * SAM_GRID + x) * SAM_DIM);
    m->pe_img = ocl_upload(ocl, pe, T * SAM_DIM * f, "decoder pe");
    G_free(pe);

    m->qpe = walloc(ocl, P * NTOK * SAM_DIM * f, "query pe");
    m->qry = walloc(ocl, P * NTOK * SAM_DIM * f, "queries");
    m->qin = walloc(ocl, P * NTOK * SAM_DIM * f, "queries+pe");
    m->sq = walloc(ocl, P * NTOK * SAM_DIM * f, "token q");
    m->sk = walloc(ocl, P * NTOK * SAM_DIM * f, "token k");
    m->sv = walloc(ocl, P * NTOK * SAM_DIM * f, "token v");
    m->so = walloc(ocl, P * NTOK * SAM_DIM * f, "token attention");
    m->sh = walloc(ocl, P * NTOK * 2048 * f, "token mlp");
    m->keys = walloc(ocl, P * T * SAM_DIM * f, "keys");
    m->kin = walloc(ocl, P * T * SAM_DIM * f, "keys+pe");
    m->bq = walloc(ocl, P * T * 128 * f, "image q");
    m->bk = walloc(ocl, P * T * 128 * f, "image k");
    m->bv = walloc(ocl, P * T * 128 * f, "image v");
    m->bo = walloc(ocl, P * T * 128 * f, "image attention");
    m->u1 = walloc(ocl, P * 4 * T * 64 * f, "upscaling 1");
    m->g2 = walloc(ocl, P * 4 * T * 128 * f, "upscaling 2 gemm");
    m->u2 = walloc(ocl, P * 16 * T * 32 * f, "upscaling 2");
    m->hyper = walloc(ocl, P * 3 * 32 * f, "hypernetwork");
    m->lowres =
        walloc(ocl, P * 3 * SAM_LOWRES * SAM_LOWRES * f, "low-res masks");
    m->h_queries = G_malloc(P * NTOK * SAM_DIM * f);
    m->sel = walloc(ocl, P * 3 * sizeof(cl_int), "mask selection");
    m->stats = walloc(ocl, P * 3 * 6 * sizeof(cl_int), "mask statistics");
    G_verbose_message(_("Allocated %lu MiB of work buffers"),
                      (unsigned long)(work_bytes >> 20));
}

void sam_load(struct sam_model *m, struct ocl_backend *ocl,
              const struct pth_file *pf, int batch)
{
    memset(m, 0, sizeof(*m));
    m->ocl = ocl;
    m->batch = batch;

    m->k_gemm = ocl_kernel(ocl, "gemm");
    m->k_gemm_big = ocl_kernel(ocl, "gemm_big");
    m->k_gemm_cpu_t = ocl_kernel(ocl, "gemm_cpu_t");
    m->k_gemm_cpu_n = ocl_kernel(ocl, "gemm_cpu_n");
    if (ocl->fp16w) {
        m->k_gemm_h = ocl_kernel_h(ocl, "gemm");
        m->k_gemm_big_h = ocl_kernel_h(ocl, "gemm_big");
        m->k_gemm_cpu_t_h = ocl_kernel_h(ocl, "gemm_cpu_t");
        m->k_gemm_cpu_n_h = ocl_kernel_h(ocl, "gemm_cpu_n");
    }
    m->k_ln = ocl_kernel(ocl, "layernorm");
    m->k_softmax = ocl_kernel(ocl, "softmax_rows");
    m->k_attn_few = ocl_kernel(ocl, "attn_fewkeys");
    m->k_attn_many = ocl_kernel(ocl, "attn_manykeys");
    m->k_stem = ocl_kernel(ocl, "stem_im2col");
    m->k_add = ocl_kernel(ocl, "add_bcast");
    m->k_wpart = ocl_kernel(ocl, "window_partition");
    m->k_wunpart = ocl_kernel(ocl, "window_unpartition_add");
    m->k_pool = ocl_kernel(ocl, "maxpool2");
    m->k_up2 = ocl_kernel(ocl, "upsample2_add");
    m->k_convt = ocl_kernel(ocl, "convt_scatter");
    m->k_logits = ocl_kernel(ocl, "mask_logits");
    m->k_post = ocl_kernel(ocl, "mask_post");

    G_message(_("Loading SAM 2 checkpoint <%s>..."), pf->path);
    loaded_bytes = 0;
    load_encoder(m, pf);
    load_decoder(m, pf);
    G_verbose_message(_("Uploaded %lu MiB of weights"),
                      (unsigned long)(loaded_bytes >> 20));

    alloc_buffers(m);
}

/* Image encoder forward pass. */

/* One Hiera MultiScaleBlock, x updated in place. */
static void hiera_block(struct sam_model *m, const struct hiera_block *b)
{
    const struct geom g = block_geom(b);
    const int H = b->side, T = H * H, Tout = g.out_side * g.out_side;
    const int d = b->dim, o = b->dim_out, hd = o / b->heads;
    const long rows = (long)g.nwin * g.Nk, qrows = (long)g.nwin * g.Nq;
    const float scale = 1.0f / sqrtf((float)hd);
    const int hc = head_chunk(m, b, &g);
    cl_mem A = m->xn, Q = m->qkv;
    int ldq = 3 * o, h0;
    long qstride = (long)g.Nk * 3 * o;

    layernorm(m, T, d, m->x, m->xn, b->n1w, b->n1b, ENC_EPS, ACT_NONE);

    /* Stage transition: the shortcut is the pooled projection of the
     * normalized input. */
    if (b->pool) {
        linear(m, T, d, o, m->xn, 0, b->dw, b->db, m->tmp, 0, ACT_NONE, 0);
        maxpool2(m, m->tmp, 0, o, m->x, o, H, 1);
    }

    if (b->window) {
        window_partition(m, m->xn, m->xw, H, d, g.s, g.nwx);
        A = m->xw;
    }
    linear(m, (int)rows, d, 3 * o, A, 0, b->qkvw, b->qkvb, m->qkv, 0, ACT_NONE,
           0);
    if (b->pool) {
        maxpool2(m, m->qkv, 0, 3 * o, m->qp, o, g.s, g.nwin);
        Q = m->qp;
        ldq = o;
        qstride = (long)g.Nq * o;
    }

    for (h0 = 0; h0 < b->heads; h0 += hc) {
        int hcc = b->heads - h0 < hc ? b->heads - h0 : hc;
        long plane = (long)g.Nq * g.Nk, kstride = (long)g.Nk * 3 * o;

        gemm(m, g.Nq, g.Nk, hd, bmat(Q, (long)h0 * hd, ldq, qstride, hd),
             bmat(m->qkv, o + (long)h0 * hd, 3 * o, kstride, hd), 1,
             bmat(m->scores, 0, g.Nk, hcc * plane, plane), g.nwin * hcc, hcc,
             NULL, NULL, 0, scale, ACT_NONE);
        softmax_rows(m, m->scores, (long)g.nwin * hcc * g.Nq, g.Nk);
        gemm(m, g.Nq, hd, g.Nk, bmat(m->scores, 0, g.Nk, hcc * plane, plane),
             bmat(m->qkv, 2L * o + (long)h0 * hd, 3 * o, kstride, hd), 0,
             bmat(m->attn, (long)h0 * hd, o, (long)g.Nq * o, hd), g.nwin * hcc,
             hcc, NULL, NULL, 0, 1.0f, ACT_NONE);
    }

    if (b->window) {
        linear(m, (int)qrows, o, o, m->attn, 0, b->projw, b->projb, m->tmp, 0,
               ACT_NONE, 0);
        window_unpartition_add(m, m->x, m->tmp, g.out_side, o, g.sq, g.nwx);
    }
    else
        linear(m, Tout, o, o, m->attn, 0, b->projw, b->projb, m->x, 0, ACT_NONE,
               1);

    layernorm(m, Tout, o, m->x, m->xn, b->n2w, b->n2b, ENC_EPS, ACT_NONE);
    linear(m, Tout, o, 4 * o, m->xn, 0, b->l1w, b->l1b, m->mlpbuf, 0, ACT_GELU,
           0);
    linear(m, Tout, 4 * o, o, m->mlpbuf, 0, b->l2w, b->l2b, m->x, 0, ACT_NONE,
           1);
}

/* FPN neck and the decoder's high-resolution projections, run at the end
 * of each stage on its output x (side x side x C). Levels 0 and 1 have
 * no top-down path (fpn_top_down_levels = [2, 3]); level 2, the image
 * embedding, adds the nearest-upsampled level 3, which is then dropped
 * (scalp = 1). */
static void neck_stage(struct sam_model *m, int stage, int side)
{
    const int T = side * side, C = m->stage_dim[stage];

    switch (stage) {
    case 0:
        linear(m, T, C, SAM_DIM, m->x, 0, m->fpn_w[0], m->fpn_b[0], m->mlpbuf,
               0, ACT_NONE, 0);
        linear(m, T, SAM_DIM, SAM_DIM / 8, m->mlpbuf, 0, m->s0w, m->s0b,
               m->feat_s0, 0, ACT_NONE, 0);
        break;
    case 1:
        linear(m, T, C, SAM_DIM, m->x, 0, m->fpn_w[1], m->fpn_b[1], m->mlpbuf,
               0, ACT_NONE, 0);
        linear(m, T, SAM_DIM, SAM_DIM / 4, m->mlpbuf, 0, m->s1w, m->s1b,
               m->feat_s1, 0, ACT_NONE, 0);
        break;
    case 2:
        linear(m, T, C, SAM_DIM, m->x, 0, m->fpn_w[2], m->fpn_b[2], m->emb, 0,
               ACT_NONE, 0);
        break;
    default: {
        cl_kernel k = m->k_up2;
        cl_int cW = SAM_GRID, cC = SAM_DIM,
               total = SAM_GRID * SAM_GRID * SAM_DIM;

        linear(m, T, C, SAM_DIM, m->x, 0, m->fpn_w[3], m->fpn_b[3], m->mlpbuf,
               0, ACT_NONE, 0);
        SETARG(k, 0, m->emb);
        SETARG(k, 1, m->mlpbuf);
        SETARG(k, 2, cW);
        SETARG(k, 3, cC);
        SETARG(k, 4, total);
        run1d(m, k, total, "upsample2_add");
        /* SAM2ImagePredictor adds no_mem_embed to the image embedding
         * (directly_add_no_mem_embed). */
        add_bcast(m, m->emb, m->emb, m->no_mem, total, SAM_DIM);
    }
    }
}

void sam_encode(struct sam_model *m, const float *img)
{
    const long T0 = (long)SAM_PATCH * SAM_PATCH, T = SAM_GRID * SAM_GRID;
    const int E = m->embed;
    int i, stage = 0;

    ocl_write(m->ocl, m->img, 0, img, 3L * SAM_IMG * SAM_IMG * sizeof(float));

    /* Stem (7x7/stride-4 convolution) and position embedding. */
    {
        cl_kernel k = m->k_stem;
        cl_int side = SAM_IMG, grid = SAM_PATCH;

        SETARG(k, 0, m->img);
        SETARG(k, 1, m->mlpbuf);
        SETARG(k, 2, side);
        SETARG(k, 3, grid);
        run1d(m, k, (size_t)T0 * STEM_K, "stem_im2col");
        linear(m, (int)T0, STEM_K, E, m->mlpbuf, 0, m->pe_w, m->pe_b, m->x, 0,
               ACT_NONE, 0);
        add_bcast(m, m->x, m->x, m->pos, T0 * E, T0 * E);
    }

    for (i = 0; i < m->depth; i++) {
        /* Report the blocks the device has finished, not the ones
         * queued: enqueueing all of them takes milliseconds. */
        clFinish(m->ocl->queue);
        progress_part(i, m->depth);
        hiera_block(m, &m->blocks[i]);
        if (i == m->stage_end[stage]) {
            neck_stage(m, stage, SAM_PATCH >> stage);
            stage++;
        }
    }

    /* Prompt-independent decoder inputs. Without a mask prompt the
     * dense embedding is no_mask_embed broadcast over the grid. */
    {
        cl_mem nm = ocl_upload(m->ocl, m->no_mask, SAM_DIM * sizeof(float),
                               "no-mask embedding");
        const struct sam_dec_layer *L0 = &m->layers[0];

        add_bcast(m, m->src, m->emb, nm, T * SAM_DIM, SAM_DIM);
        add_bcast(m, m->srcpe, m->src, m->pe_img, T * SAM_DIM, T * SAM_DIM);
        linear(m, (int)T, SAM_DIM, 128, m->srcpe, 0, L0->t2i.kw, L0->t2i.kb,
               m->kp0, 0, ACT_NONE, 0);
        linear(m, (int)T, SAM_DIM, 128, m->src, 0, L0->t2i.vw, L0->t2i.vb,
               m->vp0, 0, ACT_NONE, 0);
        linear(m, (int)T, SAM_DIM, 128, m->srcpe, 0, L0->i2t.qw, L0->i2t.qb,
               m->qp0, 0, ACT_NONE, 0);
        clFinish(m->ocl->queue);
        clReleaseMemObject(nm);
    }
}

void sam_read_features(struct sam_model *m, float *emb, float *s0, float *s1)
{
    const size_t f = sizeof(float), T0 = (size_t)SAM_PATCH * SAM_PATCH;

    ocl_read(m->ocl, m->emb, 0, emb, (size_t)SAM_GRID * SAM_GRID * SAM_DIM * f);
    ocl_read(m->ocl, m->feat_s0, 0, s0, T0 * (SAM_DIM / 8) * f);
    ocl_read(m->ocl, m->feat_s1, 0, s1, T0 / 4 * (SAM_DIM / 4) * f);
}

/* Mask decoder forward pass. */

/* y = W x + b for a PyTorch [nout, nin] weight, optional ReLU. */
static void host_linear(const float *W, const float *b, const float *x, int nin,
                        int nout, int relu, float *y)
{
    int o, i;

    for (o = 0; o < nout; o++) {
        const float *w = W + (long)o * nin;
        double acc = b[o];

        for (i = 0; i < nin; i++)
            acc += (double)w[i] * x[i];
        y[o] = relu && acc < 0.0 ? 0.0f : (float)acc;
    }
}

static void host_mlp3(float *const W[3], float *const B[3], const float *x,
                      int nout, float *y)
{
    float h1[SAM_DIM], h2[SAM_DIM];

    host_linear(W[0], B[0], x, SAM_DIM, SAM_DIM, 1, h1);
    host_linear(W[1], B[1], h1, SAM_DIM, SAM_DIM, 1, h2);
    host_linear(W[2], B[2], h2, SAM_DIM, nout, 0, y);
}

/* Token -> image cross attention on per-prompt keys, residual into the
 * queries (layers >= 1 and the final attention). */
static void t2i_attention(struct sam_model *m, int P,
                          const struct sam_attn_w *a)
{
    const long T = SAM_GRID * SAM_GRID, n = (long)P * NTOK * SAM_DIM;

    add_bcast(m, m->qin, m->qry, m->qpe, n, n);
    add_bcast(m, m->kin, m->keys, m->pe_img, P * T * SAM_DIM, T * SAM_DIM);
    linear(m, P * NTOK, SAM_DIM, 128, m->qin, 0, a->qw, a->qb, m->sq, 0,
           ACT_NONE, 0);
    linear(m, P * T, SAM_DIM, 128, m->kin, 0, a->kw, a->kb, m->bk, 0, ACT_NONE,
           0);
    linear(m, P * T, SAM_DIM, 128, m->keys, 0, a->vw, a->vb, m->bv, 0, ACT_NONE,
           0);
    mha(m, P, NTOK, T, 128, m->sq, NTOK * 128, m->bk, T * 128, m->bv, T * 128,
        m->so);
    linear(m, P * NTOK, 128, SAM_DIM, m->so, 0, a->ow, a->ob, m->qry, 0,
           ACT_NONE, 1);
}

static void token_mlp(struct sam_model *m, int P, const struct sam_dec_layer *L)
{
    linear(m, P * NTOK, SAM_DIM, 2048, m->qry, 0, L->m1w, L->m1b, m->sh, 0,
           ACT_RELU, 0);
    linear(m, P * NTOK, 2048, SAM_DIM, m->sh, 0, L->m2w, L->m2b, m->qry, 0,
           ACT_NONE, 1);
}

void sam_decode(struct sam_model *m, int P, const float *points, float *iou)
{
    const long T = SAM_GRID * SAM_GRID, n = (long)P * NTOK * SAM_DIM;
    const struct sam_dec_layer *L0 = &m->layers[0], *L1 = &m->layers[1];
    float *tok = m->h_queries, *hyper;
    int p, i;

    if (P > m->batch)
        G_fatal_error("sam_decode: batch overflow");

    /* Prompt tokens: object score token, IoU token, 4 mask tokens, the
     * positive point and the padding point SAM appends when no box is
     * given. */
    for (p = 0; p < P; p++) {
        float *t = tok + (long)p * NTOK * SAM_DIM;

        memcpy(t, m->obj_token, SAM_DIM * sizeof(float));
        memcpy(t + SAM_DIM, m->iou_token, SAM_DIM * sizeof(float));
        memcpy(t + 2 * SAM_DIM, m->mask_tokens, 4 * SAM_DIM * sizeof(float));
        pe_encode(m, (points[2 * p] + 0.5f) / SAM_IMG,
                  (points[2 * p + 1] + 0.5f) / SAM_IMG, t + 6 * SAM_DIM);
        for (i = 0; i < SAM_DIM; i++) {
            t[6 * SAM_DIM + i] += m->point_emb[1][i];
            t[7 * SAM_DIM + i] = m->not_a_point[i];
        }
    }
    ocl_write(m->ocl, m->qpe, 0, tok, n * sizeof(float));
    ocl_check(clEnqueueCopyBuffer(m->ocl->queue, m->qpe, m->qry, 0, 0,
                                  n * sizeof(float), 0, NULL, NULL),
              "clEnqueueCopyBuffer");

    /* Layer 0: self attention without positional encoding and without
     * residual (skip_first_layer_pe). */
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qry, 0, L0->self.qw, L0->self.qb,
           m->sq, 0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qry, 0, L0->self.kw, L0->self.kb,
           m->sk, 0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qry, 0, L0->self.vw, L0->self.vb,
           m->sv, 0, ACT_NONE, 0);
    mha(m, P, NTOK, NTOK, SAM_DIM, m->sq, NTOK * SAM_DIM, m->sk, NTOK * SAM_DIM,
        m->sv, NTOK * SAM_DIM, m->so);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->so, 0, L0->self.ow, L0->self.ob,
           m->qry, 0, ACT_NONE, 0);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L0->n1w, L0->n1b, DEC_EPS,
              ACT_NONE);

    /* Layer 0 token -> image: keys are still the shared image embedding,
     * so the cached projections serve every prompt (stride 0). */
    add_bcast(m, m->qin, m->qry, m->qpe, n, n);
    linear(m, P * NTOK, SAM_DIM, 128, m->qin, 0, L0->t2i.qw, L0->t2i.qb, m->sq,
           0, ACT_NONE, 0);
    mha(m, P, NTOK, T, 128, m->sq, NTOK * 128, m->kp0, 0, m->vp0, 0, m->so);
    linear(m, P * NTOK, 128, SAM_DIM, m->so, 0, L0->t2i.ow, L0->t2i.ob, m->qry,
           0, ACT_NONE, 1);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L0->n2w, L0->n2b, DEC_EPS,
              ACT_NONE);
    token_mlp(m, P, L0);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L0->n3w, L0->n3b, DEC_EPS,
              ACT_NONE);

    /* Layer 0 image -> token, queries from the shared embedding. */
    add_bcast(m, m->qin, m->qry, m->qpe, n, n);
    linear(m, P * NTOK, SAM_DIM, 128, m->qin, 0, L0->i2t.kw, L0->i2t.kb, m->sk,
           0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, 128, m->qry, 0, L0->i2t.vw, L0->i2t.vb, m->sv,
           0, ACT_NONE, 0);
    mha(m, P, T, NTOK, 128, m->qp0, 0, m->sk, NTOK * 128, m->sv, NTOK * 128,
        m->bo);
    linear(m, P * T, 128, SAM_DIM, m->bo, 0, L0->i2t.ow, L0->i2t.ob, m->keys, 0,
           ACT_NONE, 0);
    add_bcast(m, m->keys, m->keys, m->src, P * T * SAM_DIM, T * SAM_DIM);
    layernorm(m, P * T, SAM_DIM, m->keys, m->keys, L0->n4w, L0->n4b, DEC_EPS,
              ACT_NONE);

    /* Layer 1. */
    add_bcast(m, m->qin, m->qry, m->qpe, n, n);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qin, 0, L1->self.qw, L1->self.qb,
           m->sq, 0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qin, 0, L1->self.kw, L1->self.kb,
           m->sk, 0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->qry, 0, L1->self.vw, L1->self.vb,
           m->sv, 0, ACT_NONE, 0);
    mha(m, P, NTOK, NTOK, SAM_DIM, m->sq, NTOK * SAM_DIM, m->sk, NTOK * SAM_DIM,
        m->sv, NTOK * SAM_DIM, m->so);
    linear(m, P * NTOK, SAM_DIM, SAM_DIM, m->so, 0, L1->self.ow, L1->self.ob,
           m->qry, 0, ACT_NONE, 1);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L1->n1w, L1->n1b, DEC_EPS,
              ACT_NONE);
    t2i_attention(m, P, &L1->t2i);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L1->n2w, L1->n2b, DEC_EPS,
              ACT_NONE);
    token_mlp(m, P, L1);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, L1->n3w, L1->n3b, DEC_EPS,
              ACT_NONE);

    /* Layer 1 image -> token (kin = keys + pe from t2i_attention). */
    add_bcast(m, m->qin, m->qry, m->qpe, n, n);
    linear(m, P * T, SAM_DIM, 128, m->kin, 0, L1->i2t.qw, L1->i2t.qb, m->bq, 0,
           ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, 128, m->qin, 0, L1->i2t.kw, L1->i2t.kb, m->sk,
           0, ACT_NONE, 0);
    linear(m, P * NTOK, SAM_DIM, 128, m->qry, 0, L1->i2t.vw, L1->i2t.vb, m->sv,
           0, ACT_NONE, 0);
    mha(m, P, T, NTOK, 128, m->bq, T * 128, m->sk, NTOK * 128, m->sv,
        NTOK * 128, m->bo);
    linear(m, P * T, 128, SAM_DIM, m->bo, 0, L1->i2t.ow, L1->i2t.ob, m->keys, 0,
           ACT_NONE, 1);
    layernorm(m, P * T, SAM_DIM, m->keys, m->keys, L1->n4w, L1->n4b, DEC_EPS,
              ACT_NONE);

    /* Final token -> image attention. */
    t2i_attention(m, P, &m->final_t2i);
    layernorm(m, P * NTOK, SAM_DIM, m->qry, m->qry, m->nfw, m->nfb, DEC_EPS,
              ACT_NONE);

    /* Output upscaling with the high-resolution features: ConvT 256->64
     * + feat_s1, LayerNorm2d, GELU, ConvT 64->32 + feat_s0, GELU. The kin
     * buffer is free again and holds the first GEMM. */
    gemm(m, P * T, 256, SAM_DIM, mat(m->keys, 0, SAM_DIM), mat(m->up0w, 0, 256),
         0, mat(m->kin, 0, 256), 1, 1, NULL, NULL, 0, 1.0f, ACT_NONE);
    convt_scatter(m, m->kin, m->u1, m->up0b, m->feat_s1, P, SAM_GRID, SAM_GRID,
                  64, ACT_NONE);
    layernorm(m, P * 4 * T, 64, m->u1, m->u1, m->up1w, m->up1b, ENC_EPS,
              ACT_GELU);
    gemm(m, P * 4 * T, 128, 64, mat(m->u1, 0, 64), mat(m->up3w, 0, 128), 0,
         mat(m->g2, 0, 128), 1, 1, NULL, NULL, 0, 1.0f, ACT_NONE);
    convt_scatter(m, m->g2, m->u2, m->up3b, m->feat_s0, P, 2 * SAM_GRID,
                  2 * SAM_GRID, 32, ACT_GELU);

    /* Hypernetwork MLPs and IoU head are tiny: run them on the host. The
     * multimask outputs are mask tokens 1..3 (rows 3..5), the IoU head
     * reads row 1 and ends in a sigmoid (iou_prediction_use_sigmoid). */
    ocl_read(m->ocl, m->qry, 0, tok, n * sizeof(float));
    hyper = G_malloc((size_t)P * 96 * sizeof(float));
#pragma omp parallel for private(i)
    for (p = 0; p < P; p++) {
        const float *t = tok + (long)p * NTOK * SAM_DIM;
        float out[4];

        for (i = 1; i <= 3; i++)
            host_mlp3(m->hyper_w[i], m->hyper_b[i], t + (2 + i) * SAM_DIM, 32,
                      hyper + (long)p * 96 + (i - 1) * 32);
        host_mlp3(m->iou_w, m->iou_b, t + SAM_DIM, 4, out);
        for (i = 0; i < 3; i++)
            iou[p * 3 + i] = 1.0f / (1.0f + expf(-out[1 + i]));
    }
    ocl_write(m->ocl, m->hyper, 0, hyper, (size_t)P * 96 * sizeof(float));
    G_free(hyper);

    {
        cl_kernel k = m->k_logits;
        cl_int npix = SAM_LOWRES * SAM_LOWRES;
        cl_int total = P * npix;

        SETARG(k, 0, m->u2);
        SETARG(k, 1, m->hyper);
        SETARG(k, 2, m->lowres);
        SETARG(k, 3, npix);
        SETARG(k, 4, total);
        run1d(m, k, total, "mask_logits");
    }
}

void sam_postprocess(struct sam_model *m, int n, const int *sel, int out_h,
                     int out_w, float offset, int *stats)
{
    cl_kernel k = m->k_post;
    size_t need = (size_t)n * out_h * out_w, g[3], l[3] = {16, 16, 1};
    cl_int coh = out_h, cow = out_w;
    cl_float coff = offset;
    int i;

    if (n <= 0)
        return;
    if (need > m->mask_cap) {
        ocl_release(&m->mask);
        m->mask = ocl_alloc(m->ocl, need, "postprocessed masks");
        m->mask_cap = need;
    }
    for (i = 0; i < n; i++) {
        stats[6 * i + 0] = 0;
        stats[6 * i + 1] = 0;
        stats[6 * i + 2] = 0x7fffffff;
        stats[6 * i + 3] = 0x7fffffff;
        stats[6 * i + 4] = -1;
        stats[6 * i + 5] = -1;
    }
    ocl_write(m->ocl, m->sel, 0, sel, n * sizeof(cl_int));
    ocl_write(m->ocl, m->stats, 0, stats, 6 * n * sizeof(cl_int));

    SETARG(k, 0, m->lowres);
    SETARG(k, 1, m->sel);
    SETARG(k, 2, coh);
    SETARG(k, 3, cow);
    SETARG(k, 4, coff);
    SETARG(k, 5, m->mask);
    SETARG(k, 6, m->stats);
    g[0] = (size_t)(out_w + 15) / 16 * 16;
    g[1] = (size_t)(out_h + 15) / 16 * 16;
    g[2] = n;
    run(m, k, 3, g, l, "mask_post");
    ocl_read(m->ocl, m->stats, 0, stats, 6 * n * sizeof(cl_int));
}

void sam_read_mask(struct sam_model *m, int k, int out_h, int out_w, int x0,
                   int y0, int x1, int y1, unsigned char *dst)
{
    size_t origin[3] = {(size_t)x0, (size_t)y0, (size_t)k};
    size_t region[3] = {(size_t)(x1 - x0 + 1), (size_t)(y1 - y0 + 1), 1};

    ocl_check(clEnqueueReadBufferRect(m->ocl->queue, m->mask, CL_TRUE, origin,
                                      (size_t[]){0, 0, 0}, region, out_w,
                                      (size_t)out_w * out_h, region[0], 0, dst,
                                      0, NULL, NULL),
              "clEnqueueReadBufferRect");
}

void sam_free(struct sam_model *m)
{
    int p;

    for (p = 0; p < nprof; p++) {
        if (prof[p].flops > 0.0 && prof[p].wall_ms > 0.0)
            G_message("kernel %-24s %8.1f ms (wall %8.1f ms) %7ld calls "
                      "%8.1f GFLOP/s",
                      prof[p].what, prof[p].ms, prof[p].wall_ms, prof[p].calls,
                      prof[p].flops / (prof[p].wall_ms * 1e6));
        else
            G_message("kernel %-24s %8.1f ms (wall %8.1f ms) %7ld calls",
                      prof[p].what, prof[p].ms, prof[p].wall_ms,
                      prof[p].calls);
    }

    cl_mem *bufs[] = {
        &m->pe_w,   &m->pe_b,   &m->pos,  &m->s0w,     &m->s0b,     &m->s1w,
        &m->s1b,    &m->no_mem, &m->nfw,  &m->nfb,     &m->up0w,    &m->up0b,
        &m->up1w,   &m->up1b,   &m->up3w, &m->up3b,    &m->img,     &m->x,
        &m->xn,     &m->xw,     &m->qkv,  &m->qp,      &m->scores,  &m->attn,
        &m->tmp,    &m->mlpbuf, &m->emb,  &m->feat_s0, &m->feat_s1, &m->src,
        &m->pe_img, &m->srcpe,  &m->kp0,  &m->vp0,     &m->qp0,     &m->qpe,
        &m->qry,    &m->qin,    &m->sq,   &m->sk,      &m->sv,      &m->so,
        &m->sh,     &m->keys,   &m->kin,  &m->bq,      &m->bk,      &m->bv,
        &m->bo,     &m->u1,     &m->g2,   &m->u2,      &m->hyper,   &m->lowres,
        &m->mask,   &m->sel,    &m->stats};
    cl_kernel kernels[] = {m->k_gemm,      m->k_gemm_big, m->k_attn_few,
                           m->k_attn_many, m->k_ln,       m->k_softmax,
                           m->k_stem,      m->k_add,      m->k_wpart,
                           m->k_wunpart,   m->k_pool,     m->k_up2,
                           m->k_convt,     m->k_logits,   m->k_post,
                           m->k_gemm_cpu_t, m->k_gemm_cpu_n, m->k_gemm_h,
                           m->k_gemm_big_h, m->k_gemm_cpu_t_h,
                           m->k_gemm_cpu_n_h};
    size_t i;
    int j;

    for (i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++)
        ocl_release(bufs[i]);
    for (j = 0; j < SAM_STAGES; j++) {
        ocl_release(&m->fpn_w[j]);
        ocl_release(&m->fpn_b[j]);
    }
    for (j = 0; j < m->depth; j++) {
        struct hiera_block *b = &m->blocks[j];
        cl_mem *bb[] = {&b->n1w,   &b->n1b, &b->qkvw, &b->qkvb, &b->projw,
                        &b->projb, &b->n2w, &b->n2b,  &b->l1w,  &b->l1b,
                        &b->l2w,   &b->l2b, &b->dw,   &b->db};

        for (i = 0; i < sizeof(bb) / sizeof(bb[0]); i++)
            ocl_release(bb[i]);
    }
    G_free(m->blocks);
    for (j = 0; j < 2; j++) {
        struct sam_dec_layer *L = &m->layers[j];
        cl_mem *lb[] = {&L->n1w, &L->n1b, &L->n2w, &L->n2b, &L->n3w, &L->n3b,
                        &L->n4w, &L->n4b, &L->m1w, &L->m1b, &L->m2w, &L->m2b};

        release_attn(&L->self);
        release_attn(&L->t2i);
        release_attn(&L->i2t);
        for (i = 0; i < sizeof(lb) / sizeof(lb[0]); i++)
            ocl_release(lb[i]);
    }
    release_attn(&m->final_t2i);
    for (j = 0; j < 4; j++) {
        for (i = 0; i < 3; i++) {
            G_free(m->hyper_w[j][i]);
            G_free(m->hyper_b[j][i]);
        }
    }
    for (i = 0; i < 3; i++) {
        G_free(m->iou_w[i]);
        G_free(m->iou_b[i]);
    }
    for (i = 0; i < sizeof(kernels) / sizeof(kernels[0]); i++)
        if (kernels[i])
            clReleaseKernel(kernels[i]);
    G_free(m->h_queries);
    memset(m, 0, sizeof(*m));
}
