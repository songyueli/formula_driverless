#include "path_generator.hpp"

#include <algorithm>

namespace fsd
{
namespace
{
constexpr double kMaxPairDistance = 8.0;  // meters -- see algorithm note below

// Half the real track width, used by SingleSideOffsetPath below when only
// ONE boundary color is visible this frame. Measured directly from this
// track's own cone layout (same-index blue/yellow pairs in the sim's
// scene data), not just the ">=3m Formula Student rules" MINIMUM the
// pairing algorithm below cites: this specific track's width is a
// perfectly consistent 3.00m across all 122 sampled pairs.
constexpr double kAssumedHalfTrackWidth = 1.5;  // meters
}  // namespace

namespace
{
// Single-boundary fallback: offset every cone on the ONE visible side
// toward the track's center by kAssumedHalfTrackWidth, along the body
// frame's lateral (+Y = left; see control.cpp's pure-pursuit curvature
// convention, where a positive-Y target steers left) axis. Needed because
// this whole pipeline is deliberately reactive/memoryless (see
// planning.cpp's class comment) -- a SINGLE camera frame that only catches
// one boundary color (a sharp turn, brief occlusion, one missed detection)
// used to make NearestPairMidpointPath below return ZERO waypoints, and
// control.cpp deliberately full-stops on an empty path. With zero
// velocity, the car's own viewing angle never changes on the NEXT frame
// either, so this was a permanent deadlock from a single bad frame, not a
// one-frame hiccup -- confirmed directly as the actual cause of the car
// getting stuck in every real driving test this session.
//
// A fixed lateral offset (rather than each cone's own local boundary
// tangent) is a deliberate simplification: over the short interval a
// single-color frame actually spans, the car's own forward axis already
// approximates the track's local direction closely enough for pure
// pursuit's own lookahead/smoothing to absorb the difference, and a real
// tangent needs at least 2 same-side cones, which isn't guaranteed when
// only one side is visible at all.
std::vector<PathPoint> SingleSideOffsetPath(const std::vector<ClassifiedCone> &_side, double _sign)
{
    std::vector<PathPoint> waypoints;
    waypoints.reserve(_side.size());
    for (const auto &c : _side)
    {
        waypoints.push_back(PathPoint{c.x, c.y + _sign * kAssumedHalfTrackWidth});
    }
    std::sort(waypoints.begin(), waypoints.end(),
              [](const PathPoint &a, const PathPoint &b) { return a.x < b.x; });
    return waypoints;
}
}  // namespace

std::vector<PathPoint> NearestPairMidpointPath(const TrackBoundaries &boundaries)
{
    // Single-boundary fallback -- see SingleSideOffsetPath's comment above.
    // Only kicks in when one side is COMPLETELY empty; whenever both sides
    // have at least one cone, the pairing algorithm below stays in use --
    // a real midpoint between two independently-measured boundaries is
    // strictly more accurate than an assumed fixed offset.
    if (boundaries.left.empty() && !boundaries.right.empty())
    {
        // Right (yellow) cones only -> offset toward center = leftward = +Y.
        return SingleSideOffsetPath(boundaries.right, +1.0);
    }
    if (boundaries.right.empty() && !boundaries.left.empty())
    {
        // Left (blue) cones only -> offset toward center = rightward = -Y.
        return SingleSideOffsetPath(boundaries.left, -1.0);
    }

    // For each left-boundary cone, pair with its nearest right-boundary
    // cone and take the midpoint as a candidate waypoint, skipping pairs
    // farther apart than kMaxPairDistance -- real track width is >=3m
    // (Formula Student rules), so a much larger nearest-neighbor gap means
    // there's no real boundary cone visible on the other side, not a wide
    // track.
    std::vector<PathPoint> waypoints;
    waypoints.reserve(boundaries.left.size());
    for (const auto &l : boundaries.left)
    {
        const ClassifiedCone *nearest = nullptr;
        double bestDistSq = kMaxPairDistance * kMaxPairDistance;
        for (const auto &r : boundaries.right)
        {
            const double dx = r.x - l.x;
            const double dy = r.y - l.y;
            const double distSq = dx * dx + dy * dy;
            if (distSq < bestDistSq)
            {
                bestDistSq = distSq;
                nearest = &r;
            }
        }
        if (nearest)
        {
            waypoints.push_back(PathPoint{(l.x + nearest->x) / 2.0, (l.y + nearest->y) / 2.0});
        }
    }

    // Sort by body-frame x (forward distance) so control gets an ordered,
    // nearest-first path.
    std::sort(waypoints.begin(), waypoints.end(),
              [](const PathPoint &a, const PathPoint &b) { return a.x < b.x; });

    // TODO: spline-smooth the path (currently raw midpoints, which can
    // zigzag with cone-spacing irregularities) before handing it to
    // control.
    return waypoints;
}
}  // namespace fsd
