#include "path_generator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

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

// Neither the pairing midpoint above nor the single-side fixed offset
// actually check whether the waypoint they produce stays clear of any
// REAL detected cone -- confirmed directly as a real, live-reproduced
// collision cause: a car repeatedly drove into (and, backing out, kept
// re-approaching) a cone in a tight, asymmetric-spacing section of track,
// with /planned_path and /cone_detections both looking individually
// "reasonable" the whole time. kAssumedHalfTrackWidth's own fixed-offset
// assumption is exactly the kind of thing that can silently aim a waypoint
// too close to (or past) a real cone whenever the actual local geometry
// doesn't match that assumption -- a curve, an irregular gap, or simply a
// narrower-than-3m section -- and the pairing algorithm has no equivalent
// safeguard either (it only checks that a LEFT and RIGHT cone aren't
// implausibly far apart, never that the resulting MIDPOINT is far enough
// from either one, or from any OTHER nearby cone).
//
// kMinCornerClearance is a physical, not tuned, number: half the car's own
// overall width (0.60m -- the front wheel's outer edge sits at y=0.60m in
// fsd_car/model.sdf, wider than the 0.84m chassis box's own 0.42m half-
// width) plus a cone's base radius (0.115m, simulation/models/cone_blue/
// model.sdf) plus a modest tracking-error margin, rounded to 0.85m.
constexpr double kMinCornerClearance = 0.85;  // meters

// If the given waypoint is closer than kMinCornerClearance to ANY cone in
// allCones (not just whichever pair/side produced it), pushes it directly
// away from the single closest violating cone until it's exactly
// kMinCornerClearance away. A cone sitting exactly on the waypoint (dist
// ~0) has no well-defined push direction and is left alone rather than
// divided-by-zero -- degenerate and shouldn't happen given the two
// callers' own upstream logic, but cheaper to guard than to prove
// unreachable.
PathPoint EnforceClearance(PathPoint wp, const std::vector<ClassifiedCone> &allCones)
{
    const ClassifiedCone *nearest = nullptr;
    double bestDistSq = std::numeric_limits<double>::max();
    for (const auto &c : allCones)
    {
        const double dx = c.x - wp.x;
        const double dy = c.y - wp.y;
        const double distSq = dx * dx + dy * dy;
        if (distSq < bestDistSq)
        {
            bestDistSq = distSq;
            nearest = &c;
        }
    }
    if (!nearest)
    {
        return wp;
    }
    const double dist = std::sqrt(bestDistSq);
    if (dist >= kMinCornerClearance || dist < 1e-6)
    {
        return wp;
    }
    const double dx = wp.x - nearest->x;
    const double dy = wp.y - nearest->y;
    const double scale = kMinCornerClearance / dist;
    return PathPoint{nearest->x + dx * scale, nearest->y + dy * scale};
}
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
        // Clearance-checked against _side itself -- the OTHER boundary is
        // empty by definition in this branch (that's why the fallback
        // triggered), so _side is the complete set of cones a waypoint
        // here could possibly be too close to. See EnforceClearance's own
        // comment for why this check exists at all: kAssumedHalfTrackWidth
        // is a fixed assumption that can silently aim straight at a real
        // cone whenever the actual local geometry doesn't match it.
        waypoints.push_back(EnforceClearance(PathPoint{c.x, c.y + _sign * kAssumedHalfTrackWidth}, _side));
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
    //
    // allCones is checked against for EVERY waypoint below (not just the
    // pair that produced it) -- a midpoint can end up too close to a
    // THIRD cone (from either boundary) that had nothing to do with the
    // pair defining it, especially wherever cone spacing is irregular; see
    // EnforceClearance's own comment for why this matters.
    std::vector<ClassifiedCone> allCones;
    allCones.reserve(boundaries.left.size() + boundaries.right.size());
    allCones.insert(allCones.end(), boundaries.left.begin(), boundaries.left.end());
    allCones.insert(allCones.end(), boundaries.right.begin(), boundaries.right.end());

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
            const PathPoint midpoint{(l.x + nearest->x) / 2.0, (l.y + nearest->y) / 2.0};
            waypoints.push_back(EnforceClearance(midpoint, allCones));
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
