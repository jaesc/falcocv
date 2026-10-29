// FalcoCV minimal native kernel.
//
// This module intentionally has NO OpenCV dependency. It contains only the
// math-heavy operations that are measurably faster in C++:
//   - generate_pointcloud (range image -> metric point cloud)
//   - plane_fit (LSQ / RANSAC plane fitting with 3D ROI selection)
//
// Everything else lives in pure Python.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <queue>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

// ───────────────────────────── Point cloud ─────────────────────────────

static inline bool is_valid_depth(float d, bool zero_as_invalid) {
    if (std::isnan(d) || std::isinf(d)) return false;
    if (zero_as_invalid && d == 0.0f) return false;
    return true;
}

py::dict generate_pointcloud(
    py::array_t<float, py::array::c_style | py::array::forcecast> depth,
    double x_res, double y_res, double z_res, double z_offset,
    int target_points, bool zero_as_invalid,
    const std::string& /*colormap*/, bool compute_bbox)
{
    auto buf = depth.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("depth must be 2D");
    }
    const int rows = static_cast<int>(buf.shape[0]);
    const int cols = static_cast<int>(buf.shape[1]);
    const float* data = static_cast<const float*>(buf.ptr);
    const std::size_t total_pixels = static_cast<std::size_t>(rows) * cols;

    int step = 1;
    if (target_points > 0 && total_pixels > static_cast<std::size_t>(target_points)) {
        step = static_cast<int>(std::ceil(
            std::sqrt(static_cast<double>(total_pixels) / target_points)));
        if (step < 1) step = 1;
    }
    const int sampled_rows = (rows + step - 1) / step;

    std::vector<std::size_t> row_counts(static_cast<std::size_t>(sampled_rows), 0);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int ri = 0; ri < sampled_rows; ++ri) {
        const int r = ri * step;
        const float* row = data + static_cast<std::size_t>(r) * cols;
        std::size_t count = 0;
        for (int c = 0; c < cols; c += step) {
            if (is_valid_depth(row[c], zero_as_invalid)) ++count;
        }
        row_counts[static_cast<std::size_t>(ri)] = count;
    }

    std::vector<std::size_t> row_offsets(static_cast<std::size_t>(sampled_rows), 0);
    std::size_t total = 0;
    for (int ri = 0; ri < sampled_rows; ++ri) {
        row_offsets[static_cast<std::size_t>(ri)] = total;
        total += row_counts[static_cast<std::size_t>(ri)];
    }

    py::array_t<float> points(std::vector<py::ssize_t>{static_cast<py::ssize_t>(total), 3});
    py::array_t<float> intensity(std::vector<py::ssize_t>{static_cast<py::ssize_t>(total)});
    float* pts = static_cast<float*>(points.request().ptr);
    float* ints = static_cast<float*>(intensity.request().ptr);

    const float x_res_f = static_cast<float>(x_res);
    const float y_res_f = static_cast<float>(y_res);
    const float z_res_f = static_cast<float>(z_res);
    const float z_offset_f = static_cast<float>(z_offset);

    std::vector<float> row_min_x(sampled_rows, std::numeric_limits<float>::infinity());
    std::vector<float> row_min_y(sampled_rows, std::numeric_limits<float>::infinity());
    std::vector<float> row_min_z(sampled_rows, std::numeric_limits<float>::infinity());
    std::vector<float> row_max_x(sampled_rows, -std::numeric_limits<float>::infinity());
    std::vector<float> row_max_y(sampled_rows, -std::numeric_limits<float>::infinity());
    std::vector<float> row_max_z(sampled_rows, -std::numeric_limits<float>::infinity());

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int ri = 0; ri < sampled_rows; ++ri) {
        const int r = ri * step;
        const float* row = data + static_cast<std::size_t>(r) * cols;
        const float y = static_cast<float>(r) * y_res_f;
        std::size_t idx = row_offsets[static_cast<std::size_t>(ri)];
        float rmin_x = std::numeric_limits<float>::infinity();
        float rmin_y = std::numeric_limits<float>::infinity();
        float rmin_z = std::numeric_limits<float>::infinity();
        float rmax_x = -std::numeric_limits<float>::infinity();
        float rmax_y = -std::numeric_limits<float>::infinity();
        float rmax_z = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < cols; c += step) {
            const float d = row[c];
            if (!is_valid_depth(d, zero_as_invalid)) continue;
            const float px = static_cast<float>(cols - 1 - c) * x_res_f;
            const float pz = d * z_res_f + z_offset_f;
            pts[idx * 3 + 0] = px;
            pts[idx * 3 + 1] = y;
            pts[idx * 3 + 2] = pz;
            ints[idx] = d;
            if (compute_bbox) {
                rmin_x = std::min(rmin_x, px);
                rmax_x = std::max(rmax_x, px);
                rmin_y = std::min(rmin_y, y);
                rmax_y = std::max(rmax_y, y);
                rmin_z = std::min(rmin_z, pz);
                rmax_z = std::max(rmax_z, pz);
            }
            ++idx;
        }
        if (compute_bbox) {
            row_min_x[ri] = rmin_x;
            row_max_x[ri] = rmax_x;
            row_min_y[ri] = rmin_y;
            row_max_y[ri] = rmax_y;
            row_min_z[ri] = rmin_z;
            row_max_z[ri] = rmax_z;
        }
    }

    py::dict result;
    result["points"] = points;
    result["intensity"] = intensity;

    if (compute_bbox) {
        float min_x = std::numeric_limits<float>::infinity();
        float min_y = std::numeric_limits<float>::infinity();
        float min_z = std::numeric_limits<float>::infinity();
        float max_x = -std::numeric_limits<float>::infinity();
        float max_y = -std::numeric_limits<float>::infinity();
        float max_z = -std::numeric_limits<float>::infinity();
        for (int ri = 0; ri < sampled_rows; ++ri) {
            min_x = std::min(min_x, row_min_x[ri]);
            max_x = std::max(max_x, row_max_x[ri]);
            min_y = std::min(min_y, row_min_y[ri]);
            max_y = std::max(max_y, row_max_y[ri]);
            min_z = std::min(min_z, row_min_z[ri]);
            max_z = std::max(max_z, row_max_z[ri]);
        }
        if (!std::isfinite(min_x)) {
            min_x = min_y = min_z = 0.0f;
            max_x = max_y = max_z = 1.0f;
        }
        py::array_t<float> bbox_center(std::vector<py::ssize_t>{3});
        py::array_t<float> bbox_size(std::vector<py::ssize_t>{3});
        float* cptr = static_cast<float*>(bbox_center.request().ptr);
        float* sptr = static_cast<float*>(bbox_size.request().ptr);
        cptr[0] = (min_x + max_x) * 0.5f;
        cptr[1] = (min_y + max_y) * 0.5f;
        cptr[2] = (min_z + max_z) * 0.5f;
        sptr[0] = std::max(max_x - min_x, 1e-6f);
        sptr[1] = std::max(max_y - min_y, 1e-6f);
        sptr[2] = std::max(max_z - min_z, 1e-6f);
        result["bbox_center"] = bbox_center;
        result["bbox_size"] = bbox_size;
    }
    return result;
}

// ───────────────────────────── Plane fit ─────────────────────────────

static double get_double(const py::dict& d, const char* key, double def) {
    if (!d.contains(key)) return def;
    try {
        return py::cast<double>(d[key]);
    } catch (...) {
        return def;
    }
}

static int get_int(const py::dict& d, const char* key, int def) {
    if (!d.contains(key)) return def;
    try {
        return py::cast<int>(d[key]);
    } catch (...) {
        return def;
    }
}

static std::string get_string(const py::dict& d, const char* key, const char* def) {
    if (!d.contains(key)) return def;
    try {
        return py::cast<std::string>(d[key]);
    } catch (...) {
        return def;
    }
}

struct ROI3D {
    double cx, cy;
    double sx, sy;
    double angle;
    double z_min, z_max;
    bool auto_z;
};

static bool point_in_roi_xy(double x, double y, const ROI3D& r) {
    if (std::abs(r.angle) < 1e-6) {
        return (x >= r.cx - r.sx * 0.5 && x <= r.cx + r.sx * 0.5 &&
                y >= r.cy - r.sy * 0.5 && y <= r.cy + r.sy * 0.5);
    }
    const double a_rad = r.angle * 3.14159265358979323846 / 180.0;
    const double ca = std::cos(a_rad), sa = std::sin(a_rad);
    const double dx = x - r.cx;
    const double dy = y - r.cy;
    const double lx = dx * ca + dy * sa;
    const double ly = -dx * sa + dy * ca;
    return (std::abs(lx) <= r.sx * 0.5 && std::abs(ly) <= r.sy * 0.5);
}

// Jacobi eigenvalue decomposition for a symmetric 3x3 matrix.
// On return d[i] are eigenvalues and V[:, i] the corresponding eigenvectors.
static void jacobi_eigen(double A[3][3], double V[3][3], double d[3]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) V[i][j] = (i == j) ? 1.0 : 0.0;
    }

    for (int sweep = 0; sweep < 50; ++sweep) {
        const double off = std::abs(A[0][1]) + std::abs(A[0][2]) + std::abs(A[1][2]);
        if (off < 1e-15) break;

        for (int p = 0; p < 2; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                if (std::abs(A[p][q]) < 1e-18) continue;
                const double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
                const double sign = (theta >= 0.0) ? 1.0 : -1.0;
                const double t = sign / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;

                for (int k = 0; k < 3; ++k) {
                    const double akp = A[k][p];
                    const double akq = A[k][q];
                    A[k][p] = c * akp - s * akq;
                    A[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const double apk = A[p][k];
                    const double aqk = A[q][k];
                    A[p][k] = c * apk - s * aqk;
                    A[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; ++k) {
                    const double vkp = V[k][p];
                    const double vkq = V[k][q];
                    V[k][p] = c * vkp - s * vkq;
                    V[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    for (int i = 0; i < 3; ++i) d[i] = A[i][i];
}

static void svd_fit_plane(const std::vector<float>& pts, float* normal_out,
                          float* center_out, double& dist_out) {
    const std::size_t n = pts.size() / 3;
    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        cx += pts[i * 3 + 0];
        cy += pts[i * 3 + 1];
        cz += pts[i * 3 + 2];
    }
    cx /= static_cast<double>(n);
    cy /= static_cast<double>(n);
    cz /= static_cast<double>(n);
    center_out[0] = static_cast<float>(cx);
    center_out[1] = static_cast<float>(cy);
    center_out[2] = static_cast<float>(cz);

    double C00 = 0, C01 = 0, C02 = 0, C11 = 0, C12 = 0, C22 = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = pts[i * 3 + 0] - cx;
        const double dy = pts[i * 3 + 1] - cy;
        const double dz = pts[i * 3 + 2] - cz;
        C00 += dx * dx;
        C01 += dx * dy;
        C02 += dx * dz;
        C11 += dy * dy;
        C12 += dy * dz;
        C22 += dz * dz;
    }
    const double inv_n = 1.0 / static_cast<double>(n);
    C00 *= inv_n; C01 *= inv_n; C02 *= inv_n;
    C11 *= inv_n; C12 *= inv_n; C22 *= inv_n;

    double A[3][3] = {{C00, C01, C02}, {C01, C11, C12}, {C02, C12, C22}};
    double V[3][3];
    double d[3];
    jacobi_eigen(A, V, d);

    int smallest = 0;
    if (d[1] < d[smallest]) smallest = 1;
    if (d[2] < d[smallest]) smallest = 2;

    double nx = V[0][smallest];
    double ny = V[1][smallest];
    double nz = V[2][smallest];
    const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-12) {
        nx /= len; ny /= len; nz /= len;
    }
    // Stable sign: largest absolute component positive.
    if (std::abs(nx) >= std::abs(ny) && std::abs(nx) >= std::abs(nz)) {
        if (nx < 0) { nx = -nx; ny = -ny; nz = -nz; }
    } else if (std::abs(ny) >= std::abs(nz)) {
        if (ny < 0) { nx = -nx; ny = -ny; nz = -nz; }
    } else {
        if (nz < 0) { nx = -nx; ny = -ny; nz = -nz; }
    }

    normal_out[0] = static_cast<float>(nx);
    normal_out[1] = static_cast<float>(ny);
    normal_out[2] = static_cast<float>(nz);
    dist_out = nx * cx + ny * cy + nz * cz;
}

struct Pt2 {
    double x, y;
};

static py::list compute_plane_box_intersection(
    const float* normal, double dist,
    double x_min, double x_max, double y_min, double y_max,
    double z_min, double z_max, double scale_x, double scale_y, int img_w)
{
    const double edges[12][6] = {
        {x_min, y_min, z_min, x_max, y_min, z_min},
        {x_min, y_max, z_min, x_max, y_max, z_min},
        {x_min, y_min, z_max, x_max, y_min, z_max},
        {x_min, y_max, z_max, x_max, y_max, z_max},
        {x_min, y_min, z_min, x_min, y_max, z_min},
        {x_max, y_min, z_min, x_max, y_max, z_min},
        {x_min, y_min, z_max, x_min, y_max, z_max},
        {x_max, y_min, z_max, x_max, y_max, z_max},
        {x_min, y_min, z_min, x_min, y_min, z_max},
        {x_max, y_min, z_min, x_max, y_min, z_max},
        {x_min, y_max, z_min, x_min, y_max, z_max},
        {x_max, y_max, z_min, x_max, y_max, z_max},
    };

    const double nx = normal[0], ny = normal[1], nz = normal[2];
    std::vector<Pt2> pts;
    for (int i = 0; i < 12; ++i) {
        const double dx = edges[i][3] - edges[i][0];
        const double dy = edges[i][4] - edges[i][1];
        const double dz = edges[i][5] - edges[i][2];
        const double denom = nx * dx + ny * dy + nz * dz;
        if (std::abs(denom) < 1e-12) continue;
        const double t = (dist - (nx * edges[i][0] + ny * edges[i][1] +
                                   nz * edges[i][2])) / denom;
        if (t < 0.0 || t > 1.0) continue;
        const double px = edges[i][0] + t * dx;
        const double py = edges[i][1] + t * dy;
        const double col = static_cast<double>(img_w - 1) - px / scale_x;
        const double row = py / scale_y;
        pts.push_back({col, row});
    }

    if (pts.size() < 3) {
        pts = {
            {0.0, 0.0},
            {static_cast<double>(img_w - 1), 0.0},
            {static_cast<double>(img_w - 1), y_max / scale_y},
            {0.0, y_max / scale_y},
        };
    }

    double cx = 0.0, cy = 0.0;
    for (const auto& p : pts) { cx += p.x; cy += p.y; }
    cx /= static_cast<double>(pts.size());
    cy /= static_cast<double>(pts.size());
    std::sort(pts.begin(), pts.end(), [cx, cy](const Pt2& a, const Pt2& b) {
        return std::atan2(a.y - cy, a.x - cx) < std::atan2(b.y - cy, b.x - cx);
    });

    py::list polygon;
    for (const auto& p : pts) {
        py::list pt;
        pt.append(p.x);
        pt.append(p.y);
        polygon.append(pt);
    }
    return polygon;
}

py::dict plane_fit_impl(py::dict input) {
    py::dict p = input.contains("params") ? input["params"].cast<py::dict>() : py::dict();

    if (!input.contains("points")) {
        return py::dict("error"_a = "missing points");
    }
    py::array_t<float, py::array::c_style | py::array::forcecast> points_arr(input["points"]);
    auto buf = points_arr.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        return py::dict("error"_a = "points must be (N, 3) float32");
    }
    const std::size_t n = static_cast<std::size_t>(buf.shape[0]);
    if (n < 3) return py::dict("error"_a = "at least 3 points required for plane fitting");
    const float* points = static_cast<const float*>(buf.ptr);

    int img_w = 0, img_h = 0;
    double img_z_min = 0.0, img_z_max = 1.0;

    double scale_x = 1.0, scale_y = 1.0, scale_z = 1.0, offset_z = 0.0;

    if (input.contains("image")) {
        try {
            py::array_t<float, py::array::c_style | py::array::forcecast> img_arr(input["image"]);
            auto ib = img_arr.request();
            if (ib.ndim == 2) {
                img_h = static_cast<int>(ib.shape[0]);
                img_w = static_cast<int>(ib.shape[1]);
                const float* ip = static_cast<const float*>(ib.ptr);
                float raw_min = std::numeric_limits<float>::infinity();
                float raw_max = -std::numeric_limits<float>::infinity();
                for (std::size_t i = 0; i < static_cast<std::size_t>(img_h) * img_w; ++i) {
                    const float d = ip[i];
                    if (std::isfinite(d)) {
                        raw_min = std::min(raw_min, d);
                        raw_max = std::max(raw_max, d);
                    }
                }
                // calibration is parsed below; store raw range for now
                if (std::isfinite(raw_min) && std::isfinite(raw_max)) {
                    img_z_min = raw_min;
                    img_z_max = raw_max;
                }
            }
        } catch (...) {}
    }
    if (img_w <= 0 || img_h <= 0) {
        img_w = std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n)))));
        img_h = std::max(1, static_cast<int>(std::ceil(static_cast<double>(n) / img_w)));
    }

    if (input.contains("calib")) {
        try {
            py::sequence calib = input["calib"].cast<py::sequence>();
            if (calib.size() >= 1) scale_x = py::cast<double>(calib[0]);
            if (calib.size() >= 2) scale_y = py::cast<double>(calib[1]);
            if (calib.size() >= 3) scale_z = py::cast<double>(calib[2]);
            if (calib.size() >= 4) offset_z = py::cast<double>(calib[3]);
        } catch (...) {}
    }
    if (!std::isfinite(scale_x) || std::abs(scale_x) < 1e-12) scale_x = 1.0;
    if (!std::isfinite(scale_y) || std::abs(scale_y) < 1e-12) scale_y = 1.0;
    if (!std::isfinite(scale_z) || std::abs(scale_z) < 1e-12) scale_z = 1.0;
    if (!std::isfinite(offset_z)) offset_z = 0.0;

    img_z_min = img_z_min * scale_z + offset_z;
    img_z_max = img_z_max * scale_z + offset_z;
    if (img_z_max - img_z_min < 1e-6) img_z_max = img_z_min + 1.0;

    const std::string method = get_string(p, "method", "ransac");
    const double dist_thresh = get_double(p, "distance_threshold", 1.0);
    const int max_iter = get_int(p, "max_iterations", 100);
    const int min_points = get_int(p, "min_points", 10);
    int max_fit_points = get_int(p, "max_fit_points", 50000);
    if (max_fit_points < 100) max_fit_points = 100;

    std::vector<ROI3D> rois;
    if (input.contains("rois")) {
        try {
            py::sequence seq = input["rois"].cast<py::sequence>();
            for (auto item : seq) {
                if (!py::isinstance<py::dict>(item)) continue;
                py::dict e = item.cast<py::dict>();
                if (!e.contains("roi")) continue;
                py::sequence roi;
                try { roi = e["roi"].cast<py::sequence>(); } catch (...) { continue; }
                if (roi.size() < 4) continue;
                const double x = py::cast<double>(roi[0]);
                const double y = py::cast<double>(roi[1]);
                const double w = py::cast<double>(roi[2]);
                const double h = py::cast<double>(roi[3]);
                const double a_deg = roi.size() > 4 ? py::cast<double>(roi[4]) : 0.0;
                ROI3D r;
                r.cx = (static_cast<double>(img_w - 1) - (x + w * 0.5)) * scale_x;
                r.cy = (y + h * 0.5) * scale_y;
                r.sx = w * scale_x;
                r.sy = h * scale_y;
                r.angle = -a_deg;
                r.z_min = get_double(e, "z_min", 0.0);
                r.z_max = get_double(e, "z_max", 1.0);
                r.auto_z = (r.z_min == 0.0 && r.z_max == 1.0);
                rois.push_back(r);
            }
        } catch (...) {}
    }
    if (rois.empty()) {
        ROI3D r;
        r.cx = r.cy = 0.0;
        r.sx = r.sy = std::numeric_limits<double>::max();
        r.angle = 0.0;
        r.z_min = -std::numeric_limits<double>::max();
        r.z_max = std::numeric_limits<double>::max();
        r.auto_z = false;
        rois.push_back(r);
    }

    for (auto& r : rois) {
        if (!r.auto_z) continue;
        double z_min = std::numeric_limits<double>::max();
        double z_max = -std::numeric_limits<double>::max();
        auto update = [&](std::int64_t i, double& mn, double& mx) {
            const double x = points[i * 3 + 0];
            const double y = points[i * 3 + 1];
            if (!point_in_roi_xy(x, y, r)) return;
            const double z = points[i * 3 + 2];
            if (!std::isfinite(z)) return;
            mn = std::min(mn, z);
            mx = std::max(mx, z);
        };
#ifdef _OPENMP
#pragma omp parallel
        {
            double local_min = std::numeric_limits<double>::max();
            double local_max = -std::numeric_limits<double>::max();
#pragma omp for schedule(static)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
                update(i, local_min, local_max);
            }
#pragma omp critical
            {
                z_min = std::min(z_min, local_min);
                z_max = std::max(z_max, local_max);
            }
        }
#else
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
            update(i, z_min, z_max);
        }
#endif
        if (z_min <= z_max && z_max > -1e290) {
            r.z_min = z_min;
            r.z_max = z_max;
        } else {
            r.z_min = 0.0;
            r.z_max = 1.0;
        }
        if (r.z_max - r.z_min < 1e-6) r.z_max = r.z_min + 1.0;
    }

    std::vector<std::size_t> selected;
    selected.reserve(std::min<std::size_t>(n, static_cast<std::size_t>(max_fit_points) * 2));
    for (std::size_t i = 0; i < n; ++i) {
        const double x = points[i * 3 + 0];
        const double y = points[i * 3 + 1];
        const double z = points[i * 3 + 2];
        if (!std::isfinite(z)) continue;
        for (const auto& r : rois) {
            if (point_in_roi_xy(x, y, r) && z >= r.z_min && z <= r.z_max) {
                selected.push_back(i);
                break;
            }
        }
    }

    const std::size_t selected_n = selected.size();
    if (selected_n < 3) {
        return py::dict("error"_a = "not enough points inside ROIs for plane fitting");
    }

    std::vector<std::size_t> fit_idx;
    fit_idx.reserve(std::min<std::size_t>(selected_n, static_cast<std::size_t>(max_fit_points)));
    if (selected_n > static_cast<std::size_t>(max_fit_points)) {
        std::size_t stride = selected_n / static_cast<std::size_t>(max_fit_points);
        if (stride < 1) stride = 1;
        for (std::size_t i = 0; i < selected_n; i += stride) fit_idx.push_back(selected[i]);
    } else {
        fit_idx = selected;
    }

    std::vector<float> fit_points;
    fit_points.reserve(fit_idx.size() * 3);
    for (std::size_t idx : fit_idx) {
        fit_points.push_back(points[idx * 3 + 0]);
        fit_points.push_back(points[idx * 3 + 1]);
        fit_points.push_back(points[idx * 3 + 2]);
    }

    float best_normal[3] = {0.0f, 0.0f, 1.0f};
    float best_center[3] = {0.0f, 0.0f, 0.0f};
    double best_dist = 0.0;
    std::vector<unsigned char> best_inliers(selected_n, 0);
    int best_inlier_count = 0;
    double best_rms = 0.0;
    const std::size_t fit_n = fit_points.size() / 3;

    if (method == "ransac") {
        std::mt19937 rng(42);
        std::uniform_int_distribution<std::size_t> dist(0, fit_n - 1);
        for (int iter = 0; iter < max_iter; ++iter) {
            const std::size_t i0 = dist(rng);
            const std::size_t i1 = dist(rng);
            const std::size_t i2 = dist(rng);
            if (i0 == i1 || i0 == i2 || i1 == i2) continue;
            const float* p0 = fit_points.data() + i0 * 3;
            const float* p1 = fit_points.data() + i1 * 3;
            const float* p2 = fit_points.data() + i2 * 3;
            const double v1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
            const double v2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
            double nx = v1[1] * v2[2] - v1[2] * v2[1];
            double ny = v1[2] * v2[0] - v1[0] * v2[2];
            double nz = v1[0] * v2[1] - v1[1] * v2[0];
            const double len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len < 1e-12) continue;
            nx /= len; ny /= len; nz /= len;
            const double d = nx * p0[0] + ny * p0[1] + nz * p0[2];
            int inlier_count = 0;
            for (std::size_t i = 0; i < fit_n; ++i) {
                const double dp = std::abs(nx * fit_points[i * 3 + 0] +
                                           ny * fit_points[i * 3 + 1] +
                                           nz * fit_points[i * 3 + 2] - d);
                if (dp < dist_thresh) ++inlier_count;
            }
            if (inlier_count > best_inlier_count && inlier_count >= min_points) {
                std::vector<float> inlier_pts;
                inlier_pts.reserve(static_cast<std::size_t>(inlier_count) * 3);
                for (std::size_t i = 0; i < fit_n; ++i) {
                    const double dp = std::abs(nx * fit_points[i * 3 + 0] +
                                               ny * fit_points[i * 3 + 1] +
                                               nz * fit_points[i * 3 + 2] - d);
                    if (dp < dist_thresh) {
                        inlier_pts.push_back(fit_points[i * 3 + 0]);
                        inlier_pts.push_back(fit_points[i * 3 + 1]);
                        inlier_pts.push_back(fit_points[i * 3 + 2]);
                    }
                }
                float rn[3], rc[3];
                double rd;
                svd_fit_plane(inlier_pts, rn, rc, rd);
                best_inlier_count = inlier_count;
                best_normal[0] = rn[0]; best_normal[1] = rn[1]; best_normal[2] = rn[2];
                best_center[0] = rc[0]; best_center[1] = rc[1]; best_center[2] = rc[2];
                best_dist = rd;
            }
        }
        if (best_inlier_count == 0) {
            svd_fit_plane(fit_points, best_normal, best_center, best_dist);
            best_inlier_count = static_cast<int>(fit_n);
        }
        double sum_sq = 0.0;
        for (std::size_t i = 0; i < selected_n; ++i) {
            const float* pt = points + selected[i] * 3;
            const double dp = std::abs(best_normal[0] * pt[0] + best_normal[1] * pt[1] +
                                       best_normal[2] * pt[2] - best_dist);
            const bool inl = dp < dist_thresh;
            best_inliers[i] = inl ? 1 : 0;
            if (inl) sum_sq += dp * dp;
        }
        best_rms = std::sqrt(sum_sq / std::max(1, best_inlier_count));
    } else {
        svd_fit_plane(fit_points, best_normal, best_center, best_dist);
        best_inlier_count = static_cast<int>(fit_n);
        double sum_sq = 0.0;
        for (std::size_t i = 0; i < selected_n; ++i) {
            const float* pt = points + selected[i] * 3;
            const double dp = std::abs(best_normal[0] * pt[0] + best_normal[1] * pt[1] +
                                       best_normal[2] * pt[2] - best_dist);
            best_inliers[i] = 1;
            sum_sq += dp * dp;
        }
        best_rms = std::sqrt(sum_sq / selected_n);
    }

    py::array_t<float> normal_arr(std::vector<py::ssize_t>{3});
    py::array_t<float> center_arr(std::vector<py::ssize_t>{3});
    std::memcpy(normal_arr.request().ptr, best_normal, 3 * sizeof(float));
    std::memcpy(center_arr.request().ptr, best_center, 3 * sizeof(float));

    py::array_t<std::uint8_t> inlier_mask(std::vector<py::ssize_t>{static_cast<py::ssize_t>(selected_n)});
    std::memcpy(inlier_mask.request().ptr, best_inliers.data(), selected_n * sizeof(std::uint8_t));

    py::array_t<std::int64_t> selected_arr(std::vector<py::ssize_t>{static_cast<py::ssize_t>(selected_n)});
    std::int64_t* sel = static_cast<std::int64_t*>(selected_arr.request().ptr);
    for (std::size_t i = 0; i < selected_n; ++i) sel[i] = static_cast<std::int64_t>(selected[i]);

    const double box_x_min = 0.0;
    const double box_x_max = static_cast<double>(img_w - 1) * scale_x;
    const double box_y_min = 0.0;
    const double box_y_max = static_cast<double>(img_h - 1) * scale_y;
    py::list polygon = compute_plane_box_intersection(
        best_normal, best_dist, box_x_min, box_x_max, box_y_min, box_y_max,
        img_z_min, img_z_max, scale_x, scale_y, img_w);

    py::dict result;
    result["normal"] = normal_arr;
    result["center"] = center_arr;
    result["distance"] = best_dist;
    result["inlier_count"] = best_inlier_count;
    result["inlier_ratio"] = static_cast<double>(best_inlier_count) /
                             static_cast<double>(std::max<std::size_t>(1, selected_n));
    result["rms_error"] = best_rms;
    result["total_points"] = static_cast<int>(selected_n);
    result["inlier_mask"] = inlier_mask;
    result["selected_indices"] = selected_arr;
    result["polygon_2d"] = polygon;
    return result;
}


// ????????????????????????????? ShapeMatch vote ?????????????????????????????
//
// Sparse orientation-bin voting for ShapeMatch.  The Python layer extracts
// scene gradients/edges and model features; this kernel only performs the
// hypothesis voting and local-peak extraction.
py::dict shape_match_vote_impl(
    py::array_t<float, py::array::c_style | py::array::forcecast> scene_mag,
    py::array_t<float, py::array::c_style | py::array::forcecast> scene_theta,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> scene_mask,
    py::array_t<float, py::array::c_style | py::array::forcecast> model_x,
    py::array_t<float, py::array::c_style | py::array::forcecast> model_y,
    py::array_t<float, py::array::c_style | py::array::forcecast> model_theta,
    py::array_t<float, py::array::c_style | py::array::forcecast> model_mag,
    py::array_t<float, py::array::c_style | py::array::forcecast> angles,
    py::array_t<float, py::array::c_style | py::array::forcecast> scales,
    int bins,
    float min_scene_mag,
    int peaks_per_hypothesis,
    int max_candidates)
{
    auto mag_buf = scene_mag.request();
    auto theta_buf = scene_theta.request();
    auto mask_buf = scene_mask.request();
    auto mx_buf = model_x.request();
    auto my_buf = model_y.request();
    auto mt_buf = model_theta.request();
    auto mm_buf = model_mag.request();
    auto angle_buf = angles.request();
    auto scale_buf = scales.request();

    if (mag_buf.ndim != 2 || theta_buf.ndim != 2 || mask_buf.ndim != 2) {
        throw std::runtime_error("scene arrays must be 2D");
    }
    if (mx_buf.ndim != 1 || my_buf.ndim != 1 || mt_buf.ndim != 1 ||
        mm_buf.ndim != 1 || angle_buf.ndim != 1 || scale_buf.ndim != 1) {
        throw std::runtime_error("model and hypothesis arrays must be 1D");
    }

    const int height = static_cast<int>(mag_buf.shape[0]);
    const int width = static_cast<int>(mag_buf.shape[1]);
    const auto* mag = static_cast<const float*>(mag_buf.ptr);
    const auto* theta = static_cast<const float*>(theta_buf.ptr);
    const auto* mask = static_cast<const uint8_t*>(mask_buf.ptr);
    const int n_model = static_cast<int>(mx_buf.shape[0]);
    if (n_model == 0 || angle_buf.shape[0] == 0 || scale_buf.shape[0] == 0) {
        return py::dict();
    }
    const auto* mx = static_cast<const float*>(mx_buf.ptr);
    const auto* my = static_cast<const float*>(my_buf.ptr);
    const auto* mt = static_cast<const float*>(mt_buf.ptr);
    const auto* mm = static_cast<const float*>(mm_buf.ptr);
    const auto* angle_values = static_cast<const float*>(angle_buf.ptr);
    const auto* scale_values = static_cast<const float*>(scale_buf.ptr);

    const int K = std::max(1, bins);
    const float kPi = 3.14159265358979323846f;
    const float two_pi = 2.0f * kPi;
    const int n_angles = static_cast<int>(angle_buf.shape[0]);
    const int n_scales = static_cast<int>(scale_buf.shape[0]);

    // Bucket scene edge pixels by gradient orientation.
    std::vector<std::vector<std::pair<int, int>>> scene_bins(K);
    for (int y = 0; y < height; ++y) {
        const int row = y * width;
        for (int x = 0; x < width; ++x) {
            const int index = row + x;
            if (!mask[index] || mag[index] <= min_scene_mag) continue;
            int bin = static_cast<int>(std::floor((theta[index] + kPi) / two_pi * K));
            bin %= K;
            if (bin < 0) bin += K;
            scene_bins[bin].emplace_back(x, y);
        }
    }

    struct Candidate {
        float x;
        float y;
        float angle;
        float scale;
        float votes;
    };
    std::vector<Candidate> candidates;
    const int total_hypotheses = n_angles * n_scales;
    std::vector<std::vector<Candidate>> hypothesis_candidates(
        static_cast<std::size_t>(total_hypotheses));

#ifdef _OPENMP
    // Leave roughly half of the logical processors free for OpenCV's own
    // worker pool.  Oversubscribing both runtimes makes the occlusion path
    // slower even though the vote itself finishes sooner.
    const int physical_guess = std::max(1, omp_get_num_procs() / 2);
    const int worker_count = std::max(
        1, std::min({omp_get_max_threads(), physical_guess, 16}));
#else
    const int worker_count = 1;
#endif
    const std::size_t response_size =
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    std::vector<std::vector<float>> response_buffers(
        static_cast<std::size_t>(worker_count),
        std::vector<float>(response_size, 0.0f));
    std::vector<std::vector<std::uint8_t>> seen_buffers(
        static_cast<std::size_t>(worker_count),
        std::vector<std::uint8_t>(response_size, 0));

#ifdef _OPENMP
#pragma omp parallel num_threads(worker_count)
    {
        const int tid = omp_get_thread_num();
#else
    {
        const int tid = 0;
#endif
        auto& response = response_buffers[static_cast<std::size_t>(tid)];
        auto& seen = seen_buffers[static_cast<std::size_t>(tid)];
        std::vector<int> touched;
        touched.reserve(4096);
#pragma omp for schedule(static)
        for (int hypothesis = 0; hypothesis < total_hypotheses; ++hypothesis) {
            const int ai = hypothesis / n_scales;
            const int si = hypothesis % n_scales;
            const float angle = angle_values[ai];
            const float radians = angle * kPi / 180.0f;
            const float cos_a = std::cos(radians);
            const float sin_a = std::sin(radians);
            const float cos_neg = std::cos(-radians);
            const float sin_neg = std::sin(-radians);
            const float scale = scale_values[si];

            for (int i = 0; i < n_model; ++i) {
                const float rx = scale * (cos_a * mx[i] + sin_a * my[i]);
                const float ry = scale * (cos_neg * my[i] + sin_neg * mx[i]);
                int bin = static_cast<int>(
                    std::floor((mt[i] - radians + kPi) / two_pi * K));
                bin %= K;
                if (bin < 0) bin += K;
                const auto& points = scene_bins[bin];
                const float vote = mm[i];
                for (const auto& point : points) {
                    const int cx = static_cast<int>(std::lround(point.first - rx));
                    const int cy = static_cast<int>(std::lround(point.second - ry));
                    if (cx < 0 || cx >= width || cy < 0 || cy >= height) continue;
                    const std::size_t index =
                        static_cast<std::size_t>(cy) * width + cx;
                    response[index] += vote;
                    if (!seen[index]) {
                        seen[index] = 1;
                        touched.push_back(static_cast<int>(index));
                    }
                }
            }

            // Keep the strongest local maxima for this hypothesis.  Each
            // hypothesis owns its candidate vector, so the parallel region
            // never touches the same container from two threads.
            std::priority_queue<Candidate, std::vector<Candidate>,
                                std::function<bool(const Candidate&, const Candidate&)>>
                top(
                    [](const Candidate& a, const Candidate& b) {
                        return a.votes > b.votes;  // min-heap by votes
                    });
            const int keep = std::max(1, peaks_per_hypothesis);
            for (int index : touched) {
                const float value = response[static_cast<std::size_t>(index)];
                if (value <= 0.01f) continue;
                const int y = index / width;
                const int x = index - y * width;
                bool local_max = true;
                for (int dy = -2; dy <= 2 && local_max; ++dy) {
                    const int ny = y + dy;
                    if (ny < 0 || ny >= height) continue;
                    for (int dx = -2; dx <= 2; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        const int nx = x + dx;
                        if (nx < 0 || nx >= width) continue;
                        if (response[static_cast<std::size_t>(ny) * width + nx] > value) {
                            local_max = false;
                            break;
                        }
                    }
                }
                if (!local_max) continue;
                Candidate candidate{static_cast<float>(x), static_cast<float>(y), angle, scale, value};
                if (static_cast<int>(top.size()) < keep) {
                    top.push(candidate);
                } else if (value > top.top().votes) {
                    top.pop();
                    top.push(candidate);
                }
            }
            auto& local = hypothesis_candidates[static_cast<std::size_t>(hypothesis)];
            while (!top.empty()) {
                local.push_back(top.top());
                top.pop();
            }
            for (int index : touched) {
                response[static_cast<std::size_t>(index)] = 0.0f;
                seen[static_cast<std::size_t>(index)] = 0;
            }
            touched.clear();
        }
    }

    for (const auto& local : hypothesis_candidates) {
        candidates.insert(candidates.end(), local.begin(), local.end());
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.votes > b.votes;
              });
    const int limit = std::max(1, max_candidates);
    if (static_cast<int>(candidates.size()) > limit) {
        candidates.resize(limit);
    }

    const auto n = static_cast<py::ssize_t>(candidates.size());
    py::array_t<float> out_x(n), out_y(n), out_angle(n), out_scale(n), out_votes(n);
    auto x_view = out_x.mutable_unchecked<1>();
    auto y_view = out_y.mutable_unchecked<1>();
    auto a_view = out_angle.mutable_unchecked<1>();
    auto s_view = out_scale.mutable_unchecked<1>();
    auto v_view = out_votes.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < n; ++i) {
        x_view(i) = candidates[i].x;
        y_view(i) = candidates[i].y;
        a_view(i) = candidates[i].angle;
        s_view(i) = candidates[i].scale;
        v_view(i) = candidates[i].votes;
    }

    py::dict result;
    result["x"] = out_x;
    result["y"] = out_y;
    result["angle"] = out_angle;
    result["scale"] = out_scale;
    result["votes"] = out_votes;
    return result;
}

py::dict safe_shape_match_vote(
    py::array_t<float, py::array::c_style | py::array::forcecast> scene_mag,
    py::array_t<float, py::array::c_style | py::array::forcecast> scene_theta,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> scene_mask,
    py::array_t<float, py::array::c_style | py::array::forcecast> model_x,
    py::array_t<float, py::array::c_style | py::array::forcecast> model_y,
    py::array_t<float, py::array::c_style | py::array::forcecast> model_theta,
    py::array_t<float, py::array::c_style | py::array::forcecast> model_mag,
    py::array_t<float, py::array::c_style | py::array::forcecast> angles,
    py::array_t<float, py::array::c_style | py::array::forcecast> scales,
    int bins,
    float min_scene_mag,
    int peaks_per_hypothesis,
    int max_candidates)
{
    try {
        return shape_match_vote_impl(scene_mag, scene_theta, scene_mask,
                                     model_x, model_y, model_theta, model_mag,
                                     angles, scales, bins, min_scene_mag,
                                     peaks_per_hypothesis, max_candidates);
    } catch (const std::exception& e) {
        return py::dict("error"_a = e.what());
    } catch (...) {
        return py::dict("error"_a = "unknown native error");
    }
}

py::dict safe_plane_fit(py::dict input) {
    try {
        return plane_fit_impl(std::move(input));
    } catch (const std::exception& e) {
        return py::dict("error"_a = e.what());
    } catch (...) {
        return py::dict("error"_a = "unknown native error");
    }
}

}  // namespace

PYBIND11_MODULE(_falco_kernel, m) {
    m.doc() = "FalcoCV minimal native kernel (point cloud + plane fit)";
    m.def("generate_pointcloud", &generate_pointcloud,
          py::arg("depth"), py::arg("x_res") = 1.0, py::arg("y_res") = 1.0,
          py::arg("z_res") = 1.0, py::arg("z_offset") = 0.0,
          py::arg("target_points") = 1000000, py::arg("zero_as_invalid") = true,
          py::arg("colormap") = "", py::arg("compute_bbox") = false);
    m.def("plane_fit", &safe_plane_fit, py::arg("input"));
    m.def("shape_match_vote", &safe_shape_match_vote,
          py::arg("scene_mag"), py::arg("scene_theta"), py::arg("scene_mask"),
          py::arg("model_x"), py::arg("model_y"), py::arg("model_theta"), py::arg("model_mag"),
          py::arg("angles"), py::arg("scales"), py::arg("bins") = 32,
          py::arg("min_scene_mag") = 5.0f, py::arg("peaks_per_hypothesis") = 3,
          py::arg("max_candidates") = 64);
}
