#include "spline.hpp"

#include <cmath>

namespace fsd
{
namespace
{
// Centripetal alpha = 0.5 -- see FitAndSampleSpline's own comment for why
// centripetal (not uniform/chordal) is the right choice here.
constexpr double kCentripetalAlpha = 0.5;
// Fixed sub-steps per segment used to discretize the curve before
// arc-length resampling -- fine enough that the discretization error is
// negligible next to _sampleSpacing (see FitAndSampleSpline's own comment
// for why there's no closed-form arc-length alternative).
constexpr int kSubStepsPerSegment = 20;

double SegmentParamStep(const PathPoint &a, const PathPoint &b)
{
    const double dx = b.x - a.x;
    const double dy = b.y - a.y;
    const double dist = std::sqrt(dx * dx + dy * dy);
    return std::pow(dist, kCentripetalAlpha);
}

// Barry-Goldman recursive evaluation of a (possibly non-uniform) Catmull-
// Rom spline at absolute parameter _t within [_t1, _t2], given the 4
// control points and their own parameter values _t0.._t3. Works for any
// alpha (0=uniform, 0.5=centripetal, 1=chordal) via the same formula --
// this is the standard construction, not something specific to this file.
PathPoint CatmullRomEval(const PathPoint &p0, const PathPoint &p1, const PathPoint &p2, const PathPoint &p3,
                          double t0, double t1, double t2, double t3, double t)
{
    auto lerp = [](const PathPoint &a, const PathPoint &b, double ta, double tb, double t)
    {
        const double w = (t - ta) / (tb - ta);
        return PathPoint{a.x + w * (b.x - a.x), a.y + w * (b.y - a.y)};
    };
    const PathPoint a1 = lerp(p0, p1, t0, t1, t);
    const PathPoint a2 = lerp(p1, p2, t1, t2, t);
    const PathPoint a3 = lerp(p2, p3, t2, t3, t);
    const PathPoint b1 = lerp(a1, a2, t0, t2, t);
    const PathPoint b2 = lerp(a2, a3, t1, t3, t);
    return lerp(b1, b2, t1, t2, t);
}
}  // namespace

std::vector<PathPoint> FitAndSampleSpline(const std::vector<PathPoint> &points, double sampleSpacing)
{
    if (points.size() < 4)
    {
        return points;  // not enough for even one real Catmull-Rom segment
    }

    // Phantom endpoints via standard linear extrapolation, so the FIRST and
    // LAST real points still get a full 4-point basis to interpolate
    // between, same as any other interior segment.
    std::vector<PathPoint> padded;
    padded.reserve(points.size() + 2);
    padded.push_back(PathPoint{2 * points[0].x - points[1].x, 2 * points[0].y - points[1].y});
    padded.insert(padded.end(), points.begin(), points.end());
    const size_t n = points.size();
    padded.push_back(PathPoint{2 * points[n - 1].x - points[n - 2].x,
                                2 * points[n - 1].y - points[n - 2].y});

    // Dense, fixed-substep discretization of every real segment (between
    // padded[i+1] and padded[i+2], i.e. points[i] and points[i+1]).
    std::vector<PathPoint> dense;
    dense.reserve((n - 1) * static_cast<size_t>(kSubStepsPerSegment) + 1);
    dense.push_back(points[0]);
    for (size_t i = 0; i + 3 < padded.size(); ++i)
    {
        const PathPoint &p0 = padded[i];
        const PathPoint &p1 = padded[i + 1];
        const PathPoint &p2 = padded[i + 2];
        const PathPoint &p3 = padded[i + 3];
        const double t0 = 0.0;
        const double t1 = t0 + SegmentParamStep(p0, p1);
        const double t2 = t1 + SegmentParamStep(p1, p2);
        const double t3 = t2 + SegmentParamStep(p2, p3);
        for (int s = 1; s <= kSubStepsPerSegment; ++s)
        {
            const double u = static_cast<double>(s) / kSubStepsPerSegment;
            const double t = t1 + u * (t2 - t1);
            dense.push_back(CatmullRomEval(p0, p1, p2, p3, t0, t1, t2, t3, t));
        }
    }

    // Resample the dense discretization at approximately sampleSpacing
    // meters of cumulative arc length -- proportionally more samples on
    // long segments, fewer on short ones, rather than a fixed parameter
    // step that would over-sample short segments and under-sample long
    // ones.
    std::vector<PathPoint> sampled;
    sampled.push_back(dense.front());
    double accumulated = 0.0;
    for (size_t i = 1; i < dense.size(); ++i)
    {
        const double dx = dense[i].x - dense[i - 1].x;
        const double dy = dense[i].y - dense[i - 1].y;
        accumulated += std::sqrt(dx * dx + dy * dy);
        if (accumulated >= sampleSpacing)
        {
            sampled.push_back(dense[i]);
            accumulated = 0.0;
        }
    }
    if (sampled.back().x != dense.back().x || sampled.back().y != dense.back().y)
    {
        sampled.push_back(dense.back());
    }
    return sampled;
}
}  // namespace fsd
