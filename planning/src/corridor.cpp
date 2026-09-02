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
//
// BUG FIX (2026-09-01): this call was passing no initial heading at all,
// unlike centerline_extractor.cpp's own two callers (see
// OrderWaypointsByTraversal's own comment for why an initial heading
// matters) -- confirmed live as a real, not theoretical, gap: at a real
// (non-hairpin) ~38-degree bend on this track, this track's own cones are
// a uniform, measured 3.00m apart (same-index blue/yellow pairing, direct
// from trackdrive.sdf), but the LIVE computed corridor there narrowed to
// as little as 0.91-1.05m across several consecutive samples -- a
// fold-back-style chain-ordering error in this function, not a genuinely
// narrow track. _haveInitialHeading/_initialHeadingX/_initialHeadingY seed
// the SAME directional-continuity check centerline_extractor.cpp's callers
// already use, just sourced from the local spline tangent (this function's
// own caller has no vehicle pose to draw on, only the spline samples
// already in hand) instead of the vehicle's yaw.
std::vector<PathPoint> BuildChain(const std::vector<WorldCone> &_cones, const PathPoint &_cursorStart,
                                   bool _haveInitialHeading, double _initialHeadingX, double _initialHeadingY)
{
    std::vector<PathPoint> points;
    points.reserve(_cones.size());
    for (const auto &c : _cones)
    {
        points.push_back(PathPoint{c.x, c.y});
    }
    return OrderWaypointsByTraversal(std::move(points), _cursorStart, kMaxPairDistance, _haveInitialHeading,
                                      _initialHeadingX, _initialHeadingY);
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
                                    const PathPoint &_sample, double _tangentX, double _tangentY,
                                    bool _closed)
{
    if (_chain.empty())
    {
        return -1.0;
    }
    const int chainSize = static_cast<int>(_chain.size());
    // Closed: wrap the search window modulo chain size, so the index right
    // after the last one is treated as beside the one right before the
    // first -- see ComputeCorridor's own _closed comment. Open (default):
    // unchanged, clamp at both ends exactly as before.
    std::vector<int> indices;
    indices.reserve(static_cast<size_t>(2 * kChainSearchWindow + 1));
    if (_closed)
    {
        for (int off = -kChainSearchWindow; off <= kChainSearchWindow; ++off)
        {
            indices.push_back(((static_cast<int>(_cursor) + off) % chainSize + chainSize) % chainSize);
        }
    }
    else
    {
        const int lo = std::max(0, static_cast<int>(_cursor) - kChainSearchWindow);
        const int hi = std::min(chainSize - 1, static_cast<int>(_cursor) + kChainSearchWindow);
        for (int i = lo; i <= hi; ++i)
        {
            indices.push_back(i);
        }
    }

    // Prefer the closest-along-tangent candidate within the longitudinal
    // window -- "directly beside" the sample, not diagonally ahead of it.
    double bestLateral = -1.0;
    double bestLongAbs = std::numeric_limits<double>::max();
    int bestIdx = -1;
    for (int i : indices)
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
        for (int i : indices)
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
                                             const std::vector<WorldCone> &orange,
                                             double safetyMargin, double minHalfWidth,
                                             double maxHalfWidth, bool closed)
{
    std::vector<CorridorSample> result;
    if (splineSamples.empty())
    {
        return result;
    }
    result.reserve(splineSamples.size());

    // Initial heading for BuildChain's own fold-back protection (see its
    // own comment) -- the local tangent between the first two spline
    // samples, the best directional estimate available here (no vehicle
    // pose to draw on directly). Falls back to no initial heading only if
    // there's genuinely just one sample to work with (degenerate input).
    bool haveInitialHeading = false;
    double initialHeadingX = 0.0, initialHeadingY = 0.0;
    if (splineSamples.size() >= 2)
    {
        const double dx = splineSamples[1].x - splineSamples[0].x;
        const double dy = splineSamples[1].y - splineSamples[0].y;
        const double len = std::sqrt(dx * dx + dy * dy);
        if (len > 1e-6)
        {
            haveInitialHeading = true;
            initialHeadingX = dx / len;
            initialHeadingY = dy / len;
        }
    }
    const std::vector<PathPoint> blueChain =
        BuildChain(blue, splineSamples.front(), haveInitialHeading, initialHeadingX, initialHeadingY);
    const std::vector<PathPoint> yellowChain =
        BuildChain(yellow, splineSamples.front(), haveInitialHeading, initialHeadingX, initialHeadingY);
    size_t blueCursor = 0;
    size_t yellowCursor = 0;
    const int sampleCount = static_cast<int>(splineSamples.size());

    for (size_t i = 0; i < splineSamples.size(); ++i)
    {
        // Local tangent via central difference. Closed: wraps modulo
        // sampleCount, so index 0's "prev" is the last sample and the last
        // sample's "next" is index 0 -- see this function's own _closed
        // comment for why there's no real start/end to clamp at. Open
        // (default): unchanged, clamp at both ends via forward/backward
        // difference exactly as before.
        const PathPoint &prev = closed ? splineSamples[static_cast<size_t>(
                                              (static_cast<int>(i) - 1 + sampleCount) % sampleCount)]
                                        : splineSamples[i == 0 ? i : i - 1];
        const PathPoint &next = closed
            ? splineSamples[static_cast<size_t>((static_cast<int>(i) + 1) % sampleCount)]
            : splineSamples[i + 1 < splineSamples.size() ? i + 1 : i];
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

        const double blueDist =
            NearestChainLateralDistance(blueChain, blueCursor, splineSamples[i], tx, ty, closed);
        const double yellowDist =
            NearestChainLateralDistance(yellowChain, yellowCursor, splineSamples[i], tx, ty, closed);

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

        // Tighten (never widen) against nearby orange gate cones -- see
        // this function's declaration in corridor.hpp for why blue/yellow
        // alone can leave the corridor wider than the actual gate
        // passage. Plain per-cone scan, no chain/cursor: orange is sparse
        // enough (trackdrive.sdf has exactly 2) that this is cheap and
        // there's no fold-back ambiguity to guard against.
        for (const WorldCone &o : orange)
        {
            const double odx = o.x - splineSamples[i].x;
            const double ody = o.y - splineSamples[i].y;
            const double oLongitudinal = odx * tx + ody * ty;
            if (std::abs(oLongitudinal) < kLongitudinalWindow)
            {
                const double oLateral = std::abs(-odx * ty + ody * tx);
                halfWidth = std::min(halfWidth, oLateral - safetyMargin);
            }
        }

        halfWidth = std::clamp(halfWidth, minHalfWidth, maxHalfWidth);

        result.push_back(CorridorSample{splineSamples[i], tx, ty, halfWidth});
    }

    // Smooth the raw per-sample half-width -- see this function's own
    // declaration in corridor.hpp for why it's piecewise by construction.
    // Closed: wraps modulo result.size(), same reasoning as the tangent
    // and chain-search wraparound above -- otherwise the smoothing window
    // would clamp at index 0/size-1 even though those aren't real
    // boundaries on a closed loop, leaving a visible seam in the smoothed
    // half-width right at wherever the sample array happens to start.
    std::vector<double> smoothed(result.size());
    const int resultCount = static_cast<int>(result.size());
    for (size_t i = 0; i < result.size(); ++i)
    {
        double sum = 0.0;
        int count = 0;
        if (closed)
        {
            for (int off = -kSmoothingWindow; off <= kSmoothingWindow; ++off)
            {
                const int j = ((static_cast<int>(i) + off) % resultCount + resultCount) % resultCount;
                sum += result[static_cast<size_t>(j)].halfWidth;
                ++count;
            }
        }
        else
        {
            const size_t lo = (i >= static_cast<size_t>(kSmoothingWindow)) ? i - static_cast<size_t>(kSmoothingWindow) : 0;
            const size_t hi = std::min(result.size() - 1, i + static_cast<size_t>(kSmoothingWindow));
            for (size_t j = lo; j <= hi; ++j)
            {
                sum += result[j].halfWidth;
                ++count;
            }
        }
        smoothed[i] = sum / static_cast<double>(count);
    }
    for (size_t i = 0; i < result.size(); ++i)
    {
        result[i].halfWidth = smoothed[i];
    }

    return result;
}
}  // namespace fsd
