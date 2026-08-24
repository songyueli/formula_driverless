#include "path_generator.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

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

// Neither generator below has any obstacle-awareness at all -- confirmed
// directly (2026-08-23) as the root cause of a real, reproducible stuck-
// forever failure: a live full-lap test's own /cmd_ackermann showed
// ordinary, unremarkable pure-pursuit output (not the empty-path creep/
// sweep) while the car's TRUE world position sat frozen, and the nearest
// ground-truth cone was under 1m from the car -- physically wedged
// against it. Root cause traced to THIS function: nearest-pair midpoint
// pairing has no way to tell "the two nearest boundary cones straddle the
// true local track direction" apart from "cross-paired across a sharp
// corner, where the inside boundary's tighter cone spacing beats the
// outside boundary's wider one and pairs a near cone with a far one" --
// the midpoint of a bad cross-pair can land close enough to a THIRD,
// uninvolved cone (not even the pair's own two) to put the car in contact
// with it. Since reverse recovery is disallowed under FS rules (see
// control/src/pure_pursuit_controller.cpp's own SAFETY OVERRIDE comment),
// a stuck-from-contact car has no way back -- so the right fix is
// upstream, in the path itself never coming this close to a cone in the
// first place, not a recovery maneuver after the fact.
//
// kMinCarClearance uses the car's HALF-LENGTH (1.8m chassis length per
// simulation/models/fsd_car/model.sdf's collision box, so 0.9m half), not
// half-width (0.42m) -- confirmed directly as the right dimension to use,
// not assumed: an initial 0.7m value (half-width + cone radius + margin)
// still produced a live stuck-forever case, traced via the car's own
// ground-truth pose to a near-dead-ahead approach (car yawed almost
// exactly toward the blocking cone) where the front bumper -- 0.9m ahead
// of the car's own origin, not 0.42m to the side -- came out to ~0.19m
// from the cone's center, essentially touching. A pure-pursuit-following
// car generally approaches a given waypoint roughly nose-on (that's what
// steering TOWARD a target point means), so the car's LONGER dimension is
// the one that actually matters for clearance, not the shorter one.
// 0.9m (half-length) + 0.1425m (largest real cone's own base radius,
// large_orange -- simulation/models/cone_orange/model.sdf) + margin,
// rounded up: still comfortably under kAssumedHalfTrackWidth (1.5m) so it
// can't by itself force a path out of a legally-narrow (>=3m) track.
constexpr double kMinCarClearance = 1.2;  // meters

// Pushes any waypoint that ends up too close to ANY known cone (not just
// whichever pair produced it -- see this namespace's own comment above for
// why a bad cross-pair can put a waypoint close to an uninvolved THIRD
// cone) directly away from that cone until it clears kMinCarClearance.
// Applied uniformly as a final pass after path generation, regardless of
// which of the 3 return paths below produced the waypoints, rather than
// duplicated into each -- obstacle clearance is a property every path this
// module could ever produce needs, not something specific to one
// algorithm.
std::vector<PathPoint> EnforceMinClearance(std::vector<PathPoint> waypoints,
                                            const std::vector<ClassifiedCone> &allCones)
{
    for (auto &wp : waypoints)
    {
        for (const auto &cone : allCones)
        {
            const double dx = wp.x - cone.x;
            const double dy = wp.y - cone.y;
            const double distSq = dx * dx + dy * dy;
            // distSq > ~0 guards the same near-zero-distance division-by-
            // zero case as lidar_projector.cpp's own horizRange guard --
            // a waypoint landing exactly ON a cone's own center isn't a
            // real case this pipeline produces, but the push direction
            // would be undefined if it somehow did.
            if (distSq < kMinCarClearance * kMinCarClearance && distSq > 1e-9)
            {
                const double dist = std::sqrt(distSq);
                const double push = kMinCarClearance - dist;
                wp.x += (dx / dist) * push;
                wp.y += (dy / dist) * push;
            }
        }
    }
    return waypoints;
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
        waypoints.push_back(PathPoint{c.x, c.y + _sign * kAssumedHalfTrackWidth});
    }
    std::sort(waypoints.begin(), waypoints.end(),
              [](const PathPoint &a, const PathPoint &b) { return a.x < b.x; });
    return waypoints;
}
}  // namespace

std::vector<PathPoint> NearestPairMidpointPath(const TrackBoundaries &boundaries)
{
    // Every cone this cycle actually knows about, regardless of which side
    // -- see EnforceMinClearance's own comment for why clearance has to be
    // checked against ALL of these, not just whichever pair/side produced
    // a given waypoint.
    std::vector<ClassifiedCone> allCones = boundaries.left;
    allCones.insert(allCones.end(), boundaries.right.begin(), boundaries.right.end());

    // Single-boundary fallback -- see SingleSideOffsetPath's comment above.
    // Only kicks in when one side is COMPLETELY empty; whenever both sides
    // have at least one cone, the pairing algorithm below stays in use --
    // a real midpoint between two independently-measured boundaries is
    // strictly more accurate than an assumed fixed offset.
    if (boundaries.left.empty() && !boundaries.right.empty())
    {
        // Right (yellow) cones only -> offset toward center = leftward = +Y.
        return EnforceMinClearance(SingleSideOffsetPath(boundaries.right, +1.0), allCones);
    }
    if (boundaries.right.empty() && !boundaries.left.empty())
    {
        // Left (blue) cones only -> offset toward center = rightward = -Y.
        return EnforceMinClearance(SingleSideOffsetPath(boundaries.left, -1.0), allCones);
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
    return EnforceMinClearance(std::move(waypoints), allCones);
}
}  // namespace fsd
