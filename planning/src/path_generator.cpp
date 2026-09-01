#include "path_generator.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "path_utils.hpp"

namespace fsd
{
namespace
{
// MinimizeCurvature: how many neighbor-averaging passes to run, and how
// much of each pass's pull to actually apply. Picked for a smooth, visibly
// wider hairpin shape without needing many iterations to converge --
// see MinimizeCurvature's own comment for the algorithm. Not yet tuned
// against a second independent live measurement the way some other
// constants in this file are -- a first, reasoned attempt at a genuinely
// new algorithm; if live testing shows the racing line isn't wide enough
// (raise kCurvatureSmoothingIterations or kCurvatureSmoothingRate) or is
// cutting corners too aggressively into the clearance margin (lower them),
// that's a reason to retune with fresh measurements, not to have guessed
// harder up front.
constexpr int kCurvatureSmoothingIterations = 15;
constexpr double kCurvatureSmoothingRate = 0.35;
// Caps how far a SINGLE iteration can move any one waypoint -- confirmed
// directly as necessary, not just defensive: with only a handful of
// waypoints visible (a sparse-detection cycle), neighbor gaps can be large
// (up to kMaxPairDistance apart), and an uncapped pull swung a real live
// path into an unrealistic near-diagonal line off a legitimate track
// direction, stalling the car. Bounding the per-iteration step forces the
// same eventual reshaping to happen gradually over more iterations instead
// of in one potentially-huge jump, which is what actually keeps this
// numerically stable regardless of how sparse or wide the current cycle's
// waypoints happen to be.
constexpr double kCurvatureSmoothingMaxStep = 0.5;  // meters

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
// first place, not a recovery maneuver after the fact. See path_utils.hpp
// for EnforceMinClearance/EnforceMinTurnRadius, now shared with the
// landmark-based racing-line pipeline too.

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

// Least-curvature ("racing line") smoothing pass: iteratively pulls each
// INTERIOR waypoint toward the midpoint of its two neighbors, then projects
// back off any cone that pull moved it too close to (reusing
// EnforceMinClearance -- see its own comment). Pulling toward the neighbor
// midpoint is a standard discrete curvature-reduction step: a point exactly
// on the line between its neighbors has zero contribution to local
// curvature there, so repeated pulls straighten the path wherever it has
// room to, most visibly at the sharpest kinks first.
//
// This replaces hugging the raw geometric centerline (every previous
// waypoint sat equidistant between its paired cones, which forces the path
// itself down to the track's own tightest radius at a hairpin -- confirmed
// directly as the reason the car needed near-max steering angle sustained
// through the whole corner, leaving it with essentially zero tracking
// margin and prone to running off the actual track) with something closer
// to a real racing line: wide on entry and exit, cutting nearer the inside
// at the apex, because that's the shape that ACTUALLY minimizes curvature
// between two points on either side of a corner when the only constraint
// is staying clear of both boundaries -- not something hand-crafted per
// corner, it falls out of the same neighbor-averaging rule everywhere.
//
// First/last waypoints are left as anchors (never pulled): the first is
// effectively the vehicle's own current position (near the body-frame
// origin), and moving it would disconnect the path from where the car
// actually is; the last is wherever visibility currently ends, and has no
// "next" cone pairing beyond it to justify pulling it inward or outward.
//
// EnforceMinClearance runs INSIDE the iteration loop, not just once at the
// end -- a pull that would cut through a cone needs to be corrected before
// the NEXT iteration's neighbor-averaging uses that (invalid) position to
// compute its own neighbors' pulls, or one bad pull can propagate into
// otherwise-fine nearby waypoints over successive passes.
std::vector<PathPoint> MinimizeCurvature(std::vector<PathPoint> _waypoints,
                                          const std::vector<ClassifiedCone> &_allCones)
{
    if (_waypoints.size() < 3)
    {
        return _waypoints;  // nothing with an "interior" to pull
    }
    for (int iter = 0; iter < kCurvatureSmoothingIterations; ++iter)
    {
        std::vector<PathPoint> next = _waypoints;
        for (size_t i = 1; i + 1 < _waypoints.size(); ++i)
        {
            const double neighborMidX = (_waypoints[i - 1].x + _waypoints[i + 1].x) / 2.0;
            const double neighborMidY = (_waypoints[i - 1].y + _waypoints[i + 1].y) / 2.0;
            double pullX = kCurvatureSmoothingRate * (neighborMidX - _waypoints[i].x);
            double pullY = kCurvatureSmoothingRate * (neighborMidY - _waypoints[i].y);
            const double pullDist = std::sqrt(pullX * pullX + pullY * pullY);
            if (pullDist > kCurvatureSmoothingMaxStep)
            {
                const double scale = kCurvatureSmoothingMaxStep / pullDist;
                pullX *= scale;
                pullY *= scale;
            }
            next[i].x = _waypoints[i].x + pullX;
            next[i].y = _waypoints[i].y + pullY;
        }
        _waypoints = EnforceMinClearance(std::move(next), _allCones);
    }
    return _waypoints;
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
        // EnforceMinTurnRadius LAST -- see this file's own NearestPairMidpointPath
        // return statement for why order matters here.
        return EnforceMinTurnRadius(EnforceMinClearance(SingleSideOffsetPath(boundaries.right, +1.0), allCones));
    }
    if (boundaries.right.empty() && !boundaries.left.empty())
    {
        // Left (blue) cones only -> offset toward center = rightward = -Y.
        return EnforceMinTurnRadius(EnforceMinClearance(SingleSideOffsetPath(boundaries.left, -1.0), allCones));
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

    // Order into a travel-order chain (not a plain x-sort -- see
    // OrderWaypointsByTraversal's own comment for why that breaks at a
    // hairpin) so control gets an ordered, nearest-first path. Cursor is
    // the body-frame origin -- this pipeline's own vehicle position.
    waypoints = OrderWaypointsByTraversal(std::move(waypoints), PathPoint{0, 0});

    // Least-curvature ("racing line") smoothing -- see MinimizeCurvature's
    // own comment for why this replaces raw centerline-hugging, not just
    // jaggedness cleanup. EnforceMinTurnRadius/EnforceMinClearance still
    // run afterward as a hard safety-net clamp in case the smoothing pass
    // hasn't fully converged (e.g. very few visible waypoints this cycle),
    // not because MinimizeCurvature is expected to leave real work for them.
    //
    // EnforceMinTurnRadius runs LAST, not EnforceMinClearance -- confirmed
    // directly (2026-08-31) as a real, live bug when it was the other way
    // around: EnforceMinClearance's cone-avoidance push has no awareness of
    // the turn-radius constraint, so it can shove a waypoint that
    // EnforceMinTurnRadius had just correctly pulled in back OUTSIDE the
    // vehicle's achievable curvature -- pure pursuit then commands a
    // curvature nearly 2x the physical maximum, which the AckermannSteering
    // plugin can only respond to by saturating at its own hard steering
    // limit, producing exactly the "stuck/oscillating, steering not
    // updating to what we need" symptom this fixes. A physically
    // UNACHIEVABLE curvature is worse than a slightly-tighter-than-nominal
    // clearance margin, so turn-radius has to be the final, authoritative
    // clamp.
    waypoints = MinimizeCurvature(std::move(waypoints), allCones);
    return EnforceMinTurnRadius(EnforceMinClearance(std::move(waypoints), allCones));
}
}  // namespace fsd
