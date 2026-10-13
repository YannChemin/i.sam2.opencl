/****************************************************************************
 *
 * MODULE:       i.sam2.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Imagery group input, percentile stretch to 8-bit RGB and
 *               SAM 2 encoder preprocessing.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/imagery.h>
#include <grass/raster.h>

#include "image.h"
#include "progress.h"
#include "sam2_model.h"

#define NBINS 65536

/* SAM2Transforms: ImageNet mean and std on [0, 1] intensities. */
static const float pixel_mean[3] = {0.485f, 0.456f, 0.406f};
static const float pixel_std[3] = {0.229f, 0.224f, 0.225f};

static void resolve_group(const char *group, const char *subgroup,
                          struct Ref *ref)
{
    char name[GNAME_MAX], mapset[GMAPSET_MAX];
    const char *found;

    if (!G_name_is_fully_qualified(group, name, mapset)) {
        G_strlcpy(name, group, sizeof(name));
        mapset[0] = '\0';
    }
    found = G_find_file2("group", name, mapset);
    if (!found)
        G_fatal_error(_("Imagery group <%s> not found"), group);

    I_init_group_ref(ref);
    if (subgroup) {
        if (!I_find_subgroup2(name, subgroup, found))
            G_fatal_error(_("Subgroup <%s> not found in group <%s>"), subgroup,
                          group);
        if (!I_get_subgroup_ref2(name, subgroup, found, ref))
            G_fatal_error(_("Unable to read subgroup <%s> of group <%s>"),
                          subgroup, group);
    }
    else if (!I_get_group_ref2(name, found, ref))
        G_fatal_error(_("Unable to read imagery group <%s>"), group);

    if (ref->nfiles != 3)
        G_fatal_error(_("%s <%s> contains %d raster maps; exactly 3 (used "
                        "as R, G, B in group order) are required. Use "
                        "i.group to build a 3-band group or subgroup."),
                      subgroup ? _("Subgroup") : _("Group"),
                      subgroup ? subgroup : group, ref->nfiles);
}

/* Value at percentile pct (0..100) of a histogram over [min, max]. */
static double hist_percentile(const long *hist, long total, double min,
                              double max, double pct)
{
    double target = pct / 100.0 * (total - 1), width = (max - min) / NBINS;
    long cum = 0;
    int b;

    for (b = 0; b < NBINS; b++) {
        if (cum + hist[b] > target) {
            double frac = hist[b] ? (target - cum) / hist[b] : 0.0;

            return min + (b + frac) * width;
        }
        cum += hist[b];
    }
    return max;
}

void image_read_group(const char *group, const char *subgroup, double lo,
                      double hi, struct sam_image *img)
{
    struct Ref ref;
    int fd[3], b, row, col, nrows, ncols;
    DCELL *buf[3];
    double min[3], max[3], vlo[3], vhi[3];
    long *hist, total = 0;
    size_t n;

    resolve_group(group, subgroup, &ref);

    nrows = Rast_window_rows();
    ncols = Rast_window_cols();
    n = (size_t)nrows * ncols;
    img->rows = nrows;
    img->cols = ncols;
    img->rgb = G_calloc(n * 3, 1);
    progress_begin(PG_READ, n / 1e6);
    img->valid = G_calloc(n, 1);

    for (b = 0; b < 3; b++) {
        snprintf(img->names[b], sizeof(img->names[b]), "%s@%s",
                 ref.file[b].name, ref.file[b].mapset);
        fd[b] = Rast_open_old(ref.file[b].name, ref.file[b].mapset);
        buf[b] = Rast_allocate_d_buf();
        min[b] = INFINITY;
        max[b] = -INFINITY;
    }
    G_message(_("Input bands (R, G, B): <%s>, <%s>, <%s>"), img->names[0],
              img->names[1], img->names[2]);

    /* Pass 1: validity mask and per-band range over the region. */
    G_message(_("Reading imagery group <%s>..."), group);
    for (row = 0; row < nrows; row++) {
        progress_part(row, 3.0 * nrows);
        for (b = 0; b < 3; b++)
            Rast_get_d_row(fd[b], buf[b], row);
        for (col = 0; col < ncols; col++) {
            size_t i = (size_t)row * ncols + col;

            if (Rast_is_d_null_value(&buf[0][col]) ||
                Rast_is_d_null_value(&buf[1][col]) ||
                Rast_is_d_null_value(&buf[2][col]))
                continue;
            img->valid[i] = 1;
            total++;
            for (b = 0; b < 3; b++) {
                if (buf[b][col] < min[b])
                    min[b] = buf[b][col];
                if (buf[b][col] > max[b])
                    max[b] = buf[b][col];
            }
        }
    }
    if (total == 0)
        G_fatal_error(_("No cell has valid values in all three bands in the "
                        "current region"));

    /* Pass 2: per-band histograms for the percentile stretch. */
    hist = G_calloc(3L * NBINS, sizeof(long));
    for (row = 0; row < nrows; row++) {
        progress_part(nrows + row, 3.0 * nrows);
        for (b = 0; b < 3; b++)
            Rast_get_d_row(fd[b], buf[b], row);
        for (col = 0; col < ncols; col++) {
            if (!img->valid[(size_t)row * ncols + col])
                continue;
            for (b = 0; b < 3; b++) {
                double r = max[b] > min[b]
                               ? (buf[b][col] - min[b]) / (max[b] - min[b])
                               : 0.0;
                int k = (int)(r * NBINS);

                hist[(long)b * NBINS + (k >= NBINS ? NBINS - 1 : k)]++;
            }
        }
    }
    for (b = 0; b < 3; b++) {
        vlo[b] =
            hist_percentile(hist + (long)b * NBINS, total, min[b], max[b], lo);
        vhi[b] =
            hist_percentile(hist + (long)b * NBINS, total, min[b], max[b], hi);
        if (!(vhi[b] > vlo[b])) {
            vlo[b] = min[b];
            vhi[b] = max[b];
        }
        if (!(vhi[b] > vlo[b]))
            G_warning(_("Band <%s> is constant in the current region"),
                      img->names[b]);
        G_verbose_message(_("Band <%s>: stretch %g .. %g (range %g .. %g)"),
                          img->names[b], vlo[b], vhi[b], min[b], max[b]);
    }
    G_free(hist);

    /* Pass 3: 8-bit conversion. */
    for (row = 0; row < nrows; row++) {
        progress_part(2.0 * nrows + row, 3.0 * nrows);
        for (b = 0; b < 3; b++)
            Rast_get_d_row(fd[b], buf[b], row);
        for (col = 0; col < ncols; col++) {
            size_t i = (size_t)row * ncols + col;

            if (!img->valid[i])
                continue;
            for (b = 0; b < 3; b++) {
                double v = vhi[b] > vlo[b] ? (buf[b][col] - vlo[b]) /
                                                 (vhi[b] - vlo[b]) * 255.0
                                           : 0.0;

                img->rgb[3 * i + b] = v <= 0.0     ? 0
                                      : v >= 255.0 ? 255
                                                   : (unsigned char)(v + 0.5);
            }
        }
    }

    for (b = 0; b < 3; b++) {
        Rast_close(fd[b]);
        G_free(buf[b]);
    }
    I_free_group_ref(&ref);
    progress_end();
}

/* Antialiased bilinear resampling as torch.nn.functional.interpolate(
 * mode="bilinear", align_corners=False, antialias=True) does it, which
 * is what torchvision's Resize applies to SAM2ImagePredictor input: per
 * output pixel a triangle filter widened by the downscale factor, its
 * weights normalized over the input pixels it covers. When upscaling
 * this is plain bilinear interpolation. */

struct coeffs {
    int ksize;
    int *bounds; /* xmin, count per output pixel. */
    double *k;   /* ksize normalized weights per output pixel. */
};

static void precompute_coeffs(int in, int out, struct coeffs *c)
{
    double scale = (double)in / out;
    double support = scale >= 1.0 ? scale : 1.0;
    double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    int xx, x;

    c->ksize = (int)ceil(support) * 2 + 1;
    c->bounds = G_malloc(2 * out * sizeof(int));
    c->k = G_calloc((size_t)out * c->ksize, sizeof(double));

    for (xx = 0; xx < out; xx++) {
        double center = scale * (xx + 0.5), total = 0.0;
        double *k = c->k + (size_t)xx * c->ksize;
        int xmin = (int)(center - support + 0.5);
        int xmax = (int)(center + support + 0.5);

        if (xmin < 0)
            xmin = 0;
        if (xmax > in)
            xmax = in;
        xmax -= xmin;
        for (x = 0; x < xmax; x++) {
            double t = fabs((x + xmin - center + 0.5) * invscale);

            k[x] = t < 1.0 ? 1.0 - t : 0.0;
            total += k[x];
        }
        if (total != 0.0)
            for (x = 0; x < xmax; x++)
                k[x] /= total;
        c->bounds[2 * xx] = xmin;
        c->bounds[2 * xx + 1] = xmax;
    }
}

void image_encoder_input(const struct sam_image *img, int x0, int y0, int w,
                         int h, float *out)
{
    const unsigned char *src = img->rgb + ((size_t)y0 * img->cols + x0) * 3;
    const size_t plane = (size_t)SAM_IMG * SAM_IMG;
    struct coeffs cx, cy;
    float *tmp;
    int x, y, c, i;

    precompute_coeffs(w, SAM_IMG, &cx);
    precompute_coeffs(h, SAM_IMG, &cy);
    tmp = G_malloc((size_t)3 * h * SAM_IMG * sizeof(float));

    /* Horizontal pass on ToTensor's [0, 1] values, then vertical pass. */
#pragma omp parallel for private(x, c, i)
    for (y = 0; y < h; y++) {
        for (x = 0; x < SAM_IMG; x++) {
            const double *k = cx.k + (size_t)x * cx.ksize;
            int xmin = cx.bounds[2 * x], cnt = cx.bounds[2 * x + 1];

            for (c = 0; c < 3; c++) {
                double s = 0.0;

                for (i = 0; i < cnt; i++)
                    s += k[i] *
                         (src[((size_t)y * img->cols + xmin + i) * 3 + c] /
                          255.0f);
                tmp[((size_t)c * h + y) * SAM_IMG + x] = (float)s;
            }
        }
    }

#pragma omp parallel for private(x, c, i)
    for (y = 0; y < SAM_IMG; y++) {
        const double *k = cy.k + (size_t)y * cy.ksize;
        int ymin = cy.bounds[2 * y], cnt = cy.bounds[2 * y + 1];

        for (c = 0; c < 3; c++) {
            for (x = 0; x < SAM_IMG; x++) {
                double s = 0.0;

                for (i = 0; i < cnt; i++)
                    s += k[i] * tmp[((size_t)c * h + ymin + i) * SAM_IMG + x];
                out[c * plane + (size_t)y * SAM_IMG + x] =
                    ((float)s - pixel_mean[c]) / pixel_std[c];
            }
        }
    }

    G_free(tmp);
    G_free(cx.bounds);
    G_free(cx.k);
    G_free(cy.bounds);
    G_free(cy.k);
}

void image_free(struct sam_image *img)
{
    G_free(img->rgb);
    G_free(img->valid);
    img->rgb = img->valid = NULL;
}
