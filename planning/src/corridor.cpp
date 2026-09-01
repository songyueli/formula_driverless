#include "corridor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "path_utils.hpp"

namespace fsd
{
namespace
{
// How far along the tangent a chain candidate can sit and still count as
// "beside" this sample, not "ahead of/behind it" -- roughly a couple of
// this track's own measured cone spacings (~2m, see kDuplicatePruneRadius's
// comment in ekf.cpp), generous enough to always find a same-side
// candidate at normal sample spacing without pulling in one so far along
// the chain it isn't really telling you the width HERE.
constexpr double kLongitudinalWindow = 3.0;  // meters
// Chain-index window searched around the running cursor, not a global
// search -- see ComputeCorridor's own comment for why.
constexpr int kChainSearchWindow = 8;
// Moving-average half-width (in samples) for smoothing the raw per-sample
// half-width -- see ComputeCorridor's own comment for why it's piecewise
// and needs this.
constexpr int kSmoothingWindow = 3;

// Builds an ordered chain (world-frame PathPoints, one color) from a
// windowed cone set, starting the traversal from _cursorStart -- same
// technique as centerline_extractor.cpp uses for the midpoint chain
// itself, just applied per-boundary-color here.
std::vector<PathPoint> BuildChain(const std::vector<WorldCone> &_cones, const PathPoint &_cursorStart)
{
    std::vector<PathPoint> points;
    points.reserve(_cones.size());
    for (const auto &c : _cones)
    {
        points.push_back(PathPoint{c.x, c.y});
    }
    return OrderWaypointsByTraversal(std::move(points), _cursorStart);
}

// Lateral (perpendicular, left-positive) distance from _sample to the
// nearest chain point "beside" it (within kLongitudinalWindow along
// _tangent), searching only a local index window around *_cursor and
// advancing *_cursor to whatever index was actually used -- this is what
// keeps consecutive samples from jumping across a hairpin fold-back to a
// same-color landmark on the chain's OTHER, not-yet-reached leg: the
// window can only ever advance a few indices per sample, the same
// discipline OrderWaypointsByTraversal's own hop cap enforces for the
// midpoint chain. Returns -1.0 (sentinel) if the chain has no candidate at
// all this cycle (empty chain, or window ran off either end) -- callers
// treat that as "no boundary data", not zero width.
double NearestChainLateralDistance(const std::vector<PathPoint> &_chain, size_t &_cursor,
                                    const PathPoint &_sample, double _tangentX, double _tangentY)
{
    if (_chain.empty())
    {
        return -1.0;
    }
    const int lo = std::max(0, static_cast<int>(_cursor) - kChainSearchWindow);
    const int hi = std::min(static_cast<int>(_chain.size()) - 1, static_cast<int>(_cursor) + kChainSearchWindow);

    // Prefer the closest-along-tangent candidate within the longitudinal
    // window -- "directly beside" the sample, not diagonally ahead of it.
    double bestLateral = -1.0;
    double bestLongAbs = std::numeric_limits<double>::max();
    int bestIdx = -1;
    for (int i = lo; i <= hi; ++i)
    {
        const double dx = _chain[static_cast<size_t>(i)].x - _sample.x;
        const double dy = _chain[static_cast<size_t>(i)].y - _sample.y;
        const double longitudinal = dx * _tangentX + dy * _tangentY;
        if (std::abs(longitudinal) < kLongitudinalWindow && std::abs(longitudinal) < bestLongAbs)
        {
            bestLongAbs = std::abs(longitudinal);
            bestLateral = std::abs(-dx * _tangentY + dy * _tangentX);
            bestIdx = i;
        }
    }
    if (bestIdx < 0)
    {
        // Nothing in the window passed the longitudinal filter this cycle
        // (sparse chain) -- fall back to the single closest-by-full-
        // distance candidate rather than reporting "no data" when there IS
        // a chain nearby, just none of it well-aligned.
        double bestDistSq = std::numeric_limits<double>::max();
        for (int i = lo; i <= hi; ++i)
        {
            const double dx = _chain[static_cast<size_t>(i)].x - _sample.x;
            const double dy = _chain[static_cast<size_t>(i)].y - _sample.y;
            const double distSq = dx * dx + dy * dy;
            if (distSq < bestDistSq)
            {
                bestDistSq = distSq;
                bestLateral = std::abs(-dx * _tangentY + dy * _tangentX);
                bestIdx = i;
            }
        }
    }
    if (bestIdx >= 0)
    {
        _cursor = static_cast<size_t>(bestIdx);
    }
    return bestLateral;
}
}  // namespace

std::vector<CorridorSample> ComputeCorridor(const std::vector<PathPoint> &splineSamples,
                                             const std::vector<WorldCone> &blue,
                                             const std::vector<WorldCone> &yellow,
                                             double safetyMargin, double minHalfWidth,
                                             double maxHalfWidth)
{
    std::vector<CorridorSample> result;
    if (splineSamples.empty())
    {
        return result;
    }
    result.reserve(splineSamples.size());

    const std::vector<PathPoint> blueChain = BuildChain(blue, splineSamples.front());
    const std::vector<PathPoint> yellowChain = BuildChain(yellow, splineSamples.front());
    size_t blueCursor = 0;
    size_t yellowCursor = 0;

    for (size_t i = 0; i < splineSamples.size(); ++i)
    {
        // Local tangent via central difference (forward/backward at the
        // endpoints) -- the spline is already dense (see spline.hpp's own
        // sample spacing), so this is a good direction estimate without
        // needing the spline's own analytic derivative.
        const PathPoint &prev = splineSamples[i == 0 ? i : i - 1];
        const PathPoint &next = splineSamples[i + 1 < splineSamples.size() ? i + 1 : i];
        double tx = next.x - prev.x;
        double ty = next.y - prev.y;
        const double tlen = std::sqrt(tx * tx + ty * ty);
        if (tlen > 1e-9)
        {
            tx /= tlen;
            ty /= tlen;
        }
        else
        {
            tx = 1.0;
            ty = 0.0;
        }

        const double blueDist = NearestChainLateralDistance(blueChain, blueCursor, splineSamples[i], tx, ty);
        const double yellowDist = NearestChainLateralDistance(yellowChain, yellowCursor, splineSamples[i], tx, ty);

        double halfWidth;
        if (blueDist < 0.0 && yellowDist < 0.0)
        {
            halfWidth = minHalfWidth;  // no boundary data at all this sample -- stay conservative
        }
        else if (blueDist < 0.0)
        {
            halfWidth = yellowDist - safetyMargin;
        }
        else if (yellowDist < 0.0)
        {
            halfWidth = blueDist - safetyMargin;
        }
        else
        {
            halfWidth = std::min(blueDist, yellowDist) - safetyMargin;
        }
        halfWidth = std::clamp(halfWidth, minHalfWidth, maxHalfWidth);

        result.push_back(CorridorSample{splineSamples[i], tx, ty, halfWidth});
    }

    // Smooth the raw per-sample half-width -- see this function's own
    // declaration in corridor.hpp for why it's piecewise by construction.
    std::vector<double> smoothed(result.size());
    for (size_t i = 0; i < result.size(); ++i)
    {
        const size_t lo = (i >= static_cast<size_t>(kSmoothingWindow)) ? i - static_cast<size_t>(kSmoothingWindow) : 0;
        const size_t hi = std::min(result.size() - 1, i + static_cast<size_t>(kSmoothingWindow));
        double sum = 0.0;
        for (size_t j = lo; j <= hi; ++j)
        {
            sum += result[j].halfWidth;
        }
        smoothed[i] = sum / static_cast<double>(hi - lo + 1);
    }
    for (size_t i = 0; i < result.size(); ++i)
    {
        result[i].halfWidth = smoothed[i];
    }

    return result;
}
}  // namespace fsd
