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
//
// Raised from the original 1.2m (2026-08-30) after live ground-truth
// tracing through this track's one sustained near-limit corner (realized
// turn radius holds close to kMinTurnRadius, path_generator.cpp's own
// constant, over a ~9m arc, not just an instant) found the car repeatedly
// coming to rest only 0.955-1.004m from a boundary cone -- consistently
// SHORT of the 1.2m target by about 0.2-0.25m, and consistently against a
// DIFFERENT cone each attempt (cone_blue_074, then _075, then
// cone_yellow_076), not the same one -- the signature of pure pursuit's
// own tracking error eating into the nominal margin under a sustained
// tight turn, not a one-off bad waypoint. This isn't fixable by tightening
// the waypoint math further (EnforceMinClearance/EnforceMinTurnRadius
// already guarantee the INTENDED waypoints and their connecting segments
// clear kMinCarClearance -- the gap is between intended and REALIZED
// trajectory). 1.35m directly closes most of that observed shortfall
// while still leaving 0.3m of centerline slack on this track's own
// measured 3.00m width (2*1.35 = 2.7m) for the path to lean off-center
// during a turn without both boundaries' pushes fighting each other at
// once -- 1.5m (kAssumedHalfTrackWidth, exactly half the track) would
// leave none.
constexpr double kMinCarClearance = 1.35;  // meters

// Pushes any waypoint that ends up too close to ANY known cone (not just
// whichever pair produced it -- see this namespace's own comment above for
// why a bad cross-pair can put a waypoint close to an uninvolved THIRD
// cone) directly away from that cone until it clears kMinCarClearance.
// Applied uniformly as a final pass after path generation, regardless of
// which of the 3 return paths below produced the waypoints, rather than
// duplicated into each -- obstacle clearance is a property every path this
// module could ever produce needs, not something specific to one
// algorithm.
//
// Checks the SEGMENT between each pair of consecutive waypoints against
// every cone, not just the waypoints themselves -- confirmed directly as a
// real, live gap, not a theoretical one: a per-waypoint-only version of
// this check let the car come to rest only 1.004m from a real cone
// (cone_blue_074 on the sim's own trackdrive world, at a corner whose
// required radius sits close to the vehicle's kMinTurnRadius floor, so
// there's little slack left over) even though BOTH of that segment's own
// endpoint waypoints individually cleared kMinCarClearance -- the ARC pure
// pursuit actually traces between them cut inside the cone that the
// waypoint check alone had no way to see. A straight-line SEGMENT (rather
// than modeling the true circular arc) is a deliberate, cheap
// approximation: with EnforceMinTurnRadius already bounding every
// waypoint's own reachability to >=kMinTurnRadius, and real consecutive
// cone-derived waypoints only ~2m apart (this track's own real cone
// spacing, see kDuplicatePruneRadius's comment in ekf.cpp), the sagitta
// between a true kMinTurnRadius-or-larger arc and its own chord over that
// short a span is only a few centimeters (L^2/8R at L=2, R=4.5 =~ 0.11m)
// -- well inside this check's own margin, so the straight-line
// approximation doesn't need to be exact to close the real gap it's
// closing.
//
// Every violating cone's push -- from either the pointwise check below or
// this segment check -- is computed against the relevant waypoint's
// ORIGINAL position and summed into a per-waypoint accumulator, then
// applied once at the end -- NOT applied in-place per cone/segment as each
// one is found. Mutating a waypoint in place while iterating was tried
// first (for the pointwise check alone) and confirmed directly as the
// cause of a live "car isn't going through the track" regression: in a
// corner, where several boundary cones legitimately sit within
// kMinCarClearance of a single midpoint waypoint, each push shifted that
// waypoint using the ALREADY-shifted position from the previous cone, so
// the cones' iteration order (not which push actually mattered) decided
// where the waypoint ended up -- cascading drift that could shove a
// corner waypoint well off the real track centerline. Summing against
// each waypoint's fixed original position instead makes the result
// order-independent: multiple simultaneous violations (whether from
// distinct cones, or from both the pointwise and segment checks touching
// the same waypoint) blend into one net direction away from all of them,
// rather than chaining through intermediate positions none of which were
// ever the intended target. A segment violation's push is split between
// its two endpoints, weighted by how close the segment's own nearest
// point to the cone sits to each end (see ClosestPointOnSegment's `t`) --
// the endpoint nearer the violation gets more of the correction than the
// far one, rather than splitting every segment violation 50/50 regardless
// of where along it the cone actually intrudes.
struct SegmentClosestPoint
{
    double t;  // [0,1] along the segment, clamped -- 0 = point A, 1 = point B
    double x, y;
};

SegmentClosestPoint ClosestPointOnSegment(double ax, double ay, double bx, double by,
                                           double px, double py)
{
    const double abx = bx - ax;
    const double aby = by - ay;
    const double abLenSq = abx * abx + aby * aby;
    double t = 0.0;
    // abLenSq ~0 means A and B are (numerically) the same point -- t stays
    // 0, collapsing this to "distance from A", the only sensible answer
    // when there's no real segment to project onto.
    if (abLenSq > 1e-9)
    {
        t = std::clamp(((px - ax) * abx + (py - ay) * aby) / abLenSq, 0.0, 1.0);
    }
    return SegmentClosestPoint{t, ax + t * abx, ay + t * aby};
}

std::vector<PathPoint> EnforceMinClearance(std::vector<PathPoint> waypoints,
                                            const std::vector<ClassifiedCone> &allCones)
{
    const std::vector<PathPoint> original = waypoints;
    std::vector<double> pushX(waypoints.size(), 0.0);
    std::vector<double> pushY(waypoints.size(), 0.0);

    for (size_t i = 0; i < original.size(); ++i)
    {
        for (const auto &cone : allCones)
        {
            const double dx = original[i].x - cone.x;
            const double dy = original[i].y - cone.y;
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
                pushX[i] += (dx / dist) * push;
                pushY[i] += (dy / dist) * push;
            }
        }
    }

    // Assumes consecutive entries in `original` are consecutive along the
    // actual path -- true for every caller of this function today (all 3
    // return paths in this file sort their waypoints by body-frame x
    // before calling this), same assumption pure pursuit's own lookahead
    // walk already relies on.
    for (size_t i = 0; i + 1 < original.size(); ++i)
    {
        const double ax = original[i].x, ay = original[i].y;
        const double bx = original[i + 1].x, by = original[i + 1].y;
        for (const auto &cone : allCones)
        {
            const SegmentClosestPoint closest = ClosestPointOnSegment(ax, ay, bx, by, cone.x, cone.y);
            const double dx = closest.x - cone.x;
            const double dy = closest.y - cone.y;
            const double distSq = dx * dx + dy * dy;
            if (distSq < kMinCarClearance * kMinCarClearance && distSq > 1e-9)
            {
                const double dist = std::sqrt(distSq);
                const double push = kMinCarClearance - dist;
                const double pushDirX = (dx / dist) * push;
                const double pushDirY = (dy / dist) * push;
                pushX[i] += pushDirX * (1.0 - closest.t);
                pushY[i] += pushDirY * (1.0 - closest.t);
                pushX[i + 1] += pushDirX * closest.t;
                pushY[i + 1] += pushDirY * closest.t;
            }
        }
    }

    for (size_t i = 0; i < waypoints.size(); ++i)
    {
        waypoints[i].x = original[i].x + pushX[i];
        waypoints[i].y = original[i].y + pushY[i];
    }
    return waypoints;
}

// The path this module generates is a raw geometric centerline with zero
// awareness of what curvature the vehicle can actually achieve. Confirmed
// directly as the cause of a live, reproducible failure: the vehicle's own
// real minimum turning radius is 4.5m (simulation/models/fsd_car/model.sdf's
// AckermannSteering plugin: steering_limit=0.332 rad, wheel_base=1.55m,
// chosen specifically for a 4.5m min-turn-radius per that file's own
// comment), while the final hairpin's midpoint centerline curves tighter
// than that. Pure pursuit (control/src/pure_pursuit_controller.cpp)
// computes curvature = 2y/(x^2+y^2) straight from these waypoints with no
// clamp of its own, so on that corner it commands a curvature Gazebo's
// AckermannSteering plugin can't produce; the plugin silently saturates the
// steering internally, and the car understeers wide of the intended line --
// exactly the "have to make a wider turn, or it isn't possible at all"
// symptom this fixes.
//
// The fix pulls each waypoint's lateral offset in (toward the car's own
// current forward axis) just enough that the arc from the car's current
// position (origin, heading +X -- the same geometry pure pursuit's own
// curvature formula assumes) through that point never requires tighter
// than kMinTurnRadius. For a fixed x, the boundary of "achievable" is the
// circle of radius R tangent to the origin along the x-axis:
// x^2 + (y-R)^2 = R^2, i.e. |y| <= R - sqrt(R^2 - x^2) for |x| <= R (and no
// constraint at all once |x| >= R -- that same circle's own max achievable
// curvature there, 1/x, is already under 1/R by construction, so nothing
// needs clamping).
//
// This is a LOCAL, per-cycle correction, not a real racing line: it only
// guarantees pure pursuit is never asked for an infeasible arc to a given
// waypoint, not that the resulting path is an optimal wide-entry/
// clip-apex/wide-exit line. But since this pipeline regenerates the path
// fresh every frame (see planning.cpp's class comment), each cycle easing
// off exactly as much as physically necessary is what actually produces
// that wide-in/tight-out shape across consecutive frames, without this
// function needing any notion of "this is a hairpin" at all. No margin
// beyond the raw physical minimum is applied here -- this is a first,
// scoped attempt at the confirmed failure; if live testing still shows
// occasional infeasible commands (e.g. from lookahead/discretization
// effects this simple per-waypoint check doesn't model), that's a reason
// to add one, empirically justified, not to guess one in up front.
constexpr double kMinTurnRadius = 4.5;  // meters -- see comment above

std::vector<PathPoint> EnforceMinTurnRadius(std::vector<PathPoint> waypoints)
{
    for (auto &wp : waypoints)
    {
        const double absX = std::abs(wp.x);
        if (absX >= kMinTurnRadius)
        {
            continue;
        }
        const double maxAbsY = kMinTurnRadius - std::sqrt(kMinTurnRadius * kMinTurnRadius - absX * absX);
        if (std::abs(wp.y) > maxAbsY)
        {
            wp.y = std::copysign(maxAbsY, wp.y);
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
        return EnforceMinClearance(EnforceMinTurnRadius(SingleSideOffsetPath(boundaries.right, +1.0)), allCones);
    }
    if (boundaries.right.empty() && !boundaries.left.empty())
    {
        // Left (blue) cones only -> offset toward center = rightward = -Y.
        return EnforceMinClearance(EnforceMinTurnRadius(SingleSideOffsetPath(boundaries.left, -1.0)), allCones);
    }

    // For each left-boundary cone, pair with its nearest right-boundary
    // cone and take the midpoint as a candidate waypoint, skipping pairs
    // farther apart than kMaxPairDistance -- real track width is >=3m
    // (Formula Student rules), so a much larger nearest-neighbor gap means
    // there's no real boundary cone visible on the other side, not a wide
    // track.
    //
    // MUTUAL nearest-neighbor required, not just one-directional -- see
    // this namespace's own top comment for the confirmed failure this
    // fixes: at a sharp corner, the inside boundary's tighter cone spacing
    // means a left cone's OWN nearest right cone can easily be one that
    // some OTHER left cone is a strictly better match for (i.e. that right
    // cone's own nearest LEFT cone is a different one) -- a genuine
    // cross-pair, not measurement noise. Live ground-truth tracing through
    // this exact track's recurring stuck corner confirmed the raw
    // one-directional match was accepting exactly this kind of asymmetric
    // pair under partial visibility (only a subset of each boundary in
    // view at once), even though the FULL-track cone layout pairs cleanly
    // (every blue cone's true correspondence is its own same-numbered
    // yellow cone, all at a consistent 3.00m). Requiring the match to be
    // symmetric -- l's nearest right cone is r, AND r's nearest left cone
    // is l -- costs an extra O(leftSize) reverse scan per accepted
    // candidate (bounded by how many cones are actually visible in one
    // frame, typically single digits to low tens) and means some left
    // cones legitimately produce NO waypoint this cycle rather than a
    // wrong one -- the same "no match is better than a bad match"
    // philosophy already used throughout this pipeline (e.g.
    // lidar_projector.cpp's Localize(), CorrectOrAddLandmark's Mahalanobis
    // gate).
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
        if (!nearest)
        {
            continue;
        }

        const ClassifiedCone *reciprocal = nullptr;
        double reciprocalBestDistSq = kMaxPairDistance * kMaxPairDistance;
        for (const auto &l2 : boundaries.left)
        {
            const double dx = l2.x - nearest->x;
            const double dy = l2.y - nearest->y;
            const double distSq = dx * dx + dy * dy;
            if (distSq < reciprocalBestDistSq)
            {
                reciprocalBestDistSq = distSq;
                reciprocal = &l2;
            }
        }
        if (reciprocal != &l)
        {
            continue;  // not a mutual match -- likely a cross-pair, skip rather than accept
        }

        waypoints.push_back(PathPoint{(l.x + nearest->x) / 2.0, (l.y + nearest->y) / 2.0});
    }

    // Sort by body-frame x (forward distance) so control gets an ordered,
    // nearest-first path.
    std::sort(waypoints.begin(), waypoints.end(),
              [](const PathPoint &a, const PathPoint &b) { return a.x < b.x; });

    // TODO: spline-smooth the path (currently raw midpoints, which can
    // zigzag with cone-spacing irregularities) before handing it to
    // control.
    return EnforceMinClearance(EnforceMinTurnRadius(std::move(waypoints)), allCones);
}
}  // namespace fsd
