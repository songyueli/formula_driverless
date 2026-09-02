#include "path_utils.hpp"

#include <algorithm>
#include <cmath>

namespace fsd
{
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

// Pushes any waypoint that ends up too close to ANY known cone directly
// away from that cone until it clears kMinCarClearance. Checks the SEGMENT
// between each pair of consecutive waypoints against every cone too, not
// just the waypoints themselves -- confirmed directly as a real, live gap:
// a per-waypoint-only version let the car come to rest only 1.004m from a
// real cone even though both of that segment's own endpoints individually
// cleared kMinCarClearance -- the ARC pure pursuit actually traces between
// them cut inside the cone the waypoint check alone couldn't see. A
// straight-line SEGMENT (rather than the true circular arc) is a
// deliberate, cheap approximation: with EnforceMinTurnRadius already
// bounding every waypoint's own reachability to >=kMinTurnRadius, and real
// consecutive cone-derived waypoints only ~2m apart on this track, the
// sagitta between a true kMinTurnRadius-or-larger arc and its own chord
// over that short a span is only a few centimeters (L^2/8R at L=2, R=4.5
// =~ 0.11m) -- well inside this check's own margin.
//
// Every violating cone's push is computed against the relevant waypoint's
// ORIGINAL position and summed into a per-waypoint accumulator, applied
// once at the end -- NOT in-place per cone/segment as each is found.
// Mutating in place while iterating was tried first and confirmed directly
// as the cause of a live "car isn't going through the track" regression:
// in a corner, several boundary cones legitimately sit within
// kMinCarClearance of a single waypoint, and each push shifted that
// waypoint using the ALREADY-shifted position from the previous cone, so
// the cones' iteration order (not which push actually mattered) decided
// the result. Summing against each waypoint's fixed original position
// makes the result order-independent. A segment violation's push is split
// between its two endpoints, weighted by how close the segment's own
// nearest point to the cone sits to each end (ClosestPointOnSegment's `t`).
//
// kMinCarClearance uses the car's HALF-LENGTH (1.8m chassis length per
// simulation/models/fsd_car/model.sdf's collision box, so 0.9m half), not
// half-width (0.42m) -- confirmed directly as the right dimension: a
// pure-pursuit-following car generally approaches a given waypoint roughly
// nose-on, so the car's LONGER dimension is the one that actually matters.
// 0.9m (half-length) + 0.1425m (largest real cone's own base radius) +
// margin, raised to 1.35m (from an original 1.2m) after live ground-truth
// tracing through this track's one sustained near-limit corner found the
// car repeatedly coming to rest only 0.955-1.004m from a boundary cone --
// consistently SHORT of the 1.2m target, the signature of pure pursuit's
// own tracking error eating into the nominal margin under a sustained
// tight turn, not a one-off bad waypoint.
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
    // actual path -- true for every caller today (all callers order their
    // waypoints via OrderWaypointsByTraversal before calling this), same
    // assumption pure pursuit's own lookahead walk already relies on.
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

// Clamps each waypoint's lateral offset so the arc from the origin
// (heading +X) through it never requires tighter than kMinTurnRadius --
// confirmed directly as necessary: the vehicle's real minimum turning
// radius is 4.5m (simulation/models/fsd_car/model.sdf's AckermannSteering
// plugin), and without this clamp a raw geometric centerline can curve
// tighter than that at a hairpin, so pure pursuit (curvature = 2y/(x^2+y^2),
// no clamp of its own) commands a curvature the plugin can't produce -- it
// silently saturates internally and the car understeers wide of the
// intended line. For a fixed x, the boundary of "achievable" is the circle
// of radius R tangent to the origin along the x-axis: x^2+(y-R)^2=R^2, i.e.
// |y| <= R - sqrt(R^2-x^2) for |x|<=R (no constraint once |x|>=R -- that
// circle's own max achievable curvature there, 1/x, is already under 1/R).
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

// Orders an unordered set of waypoints into travel order via greedy
// nearest-neighbor walking from _cursor, repeatedly appending whichever
// remaining point is nearest to the current chain end. Replaces a plain
// x-sort, which silently assumed the path never curves back on itself
// within the visible window -- true for gentle curves, but false at a
// hairpin: far-side waypoints can have a SMALLER x than near-side ones once
// the track folds back past perpendicular, so an x-sort interleaves the two
// sides into a zigzag instead of a curve. Confirmed directly as the cause
// of a real zigzag seen live at this track's hairpin.
//
// kMaxPairDistance bounds each hop: without a cap, the chain could jump
// across a hairpin's own gap to a spatially-close point that's actually on
// the opposite, not-yet-reached leg -- a wrong hop would reintroduce the
// exact zigzag this exists to remove. Points the walk can't reach within
// that cap are left out rather than forced in with a bad hop, matching this
// pipeline's "no match is better than a bad match" philosophy.
//
// BUG FIX (2026-09-01): a distance cap alone is NOT sufficient at a sharp
// hairpin -- confirmed live (a genuine wedge/stuck event, plus a corridor
// whose left/right boundaries crossed over each other right at this
// track's own hairpin). The point just before the apex and the point just
// after it landed only 1.96m apart in raw Euclidean space (the hairpin
// folds back on itself), comfortably inside kMaxPairDistance=8.0m, so the
// greedy walk picked the wrong (behind-the-cursor) one -- a near-180-degree
// reversal that a plain nearest-neighbor rule has no way to notice.
// Measured directly against this exact hairpin's own true centerline: real
// hop-to-hop turn angle never exceeds ~24 degrees even at the sharpest
// point, while the confirmed bad hop represented ~118 degrees -- an
// enormous, safe margin between "a real turn" and "a fold-back", so
// requiring each hop to stay within kMaxHeadingReversalDeg=90 of the
// heading established by the PREVIOUS hop rejects the fold-back outright
// without ever being close to rejecting a genuine hairpin's own curvature.
// Only checked once a heading actually exists (from the second accepted
// point onward) -- the very first hop, from _cursor (not itself a waypoint
// on the chain), has no established direction to compare against and stays
// pure nearest-neighbor, same as before.
constexpr double kMaxHeadingReversalDeg = 90.0;

std::vector<PathPoint> OrderWaypointsByTraversal(std::vector<PathPoint> _waypoints, PathPoint _cursor,
                                                  double _maxHopDistance, bool _haveInitialHeading,
                                                  double _initialHeadingX, double _initialHeadingY)
{
    const double minHeadingDot = std::cos(kMaxHeadingReversalDeg * M_PI / 180.0);

    std::vector<PathPoint> ordered;
    ordered.reserve(_waypoints.size());
    std::vector<bool> used(_waypoints.size(), false);

    double cursorX = _cursor.x;
    double cursorY = _cursor.y;
    bool haveHeading = _haveInitialHeading;
    double headingX = _initialHeadingX, headingY = _initialHeadingY;
    for (size_t step = 0; step < _waypoints.size(); ++step)
    {
        int bestIdx = -1;
        double bestDistSq = _maxHopDistance * _maxHopDistance;
        for (size_t i = 0; i < _waypoints.size(); ++i)
        {
            if (used[i])
            {
                continue;
            }
            const double dx = _waypoints[i].x - cursorX;
            const double dy = _waypoints[i].y - cursorY;
            const double distSq = dx * dx + dy * dy;
            if (distSq >= bestDistSq)
            {
                continue;
            }
            if (haveHeading && distSq > 1e-9)
            {
                const double invLen = 1.0 / std::sqrt(distSq);
                const double dot = dx * invLen * headingX + dy * invLen * headingY;
                if (dot < minHeadingDot)
                {
                    continue;  // implausible reversal relative to the established
                               // heading -- almost certainly a fold-back hop, not
                               // the real next point; skip rather than take it.
                }
            }
            bestDistSq = distSq;
            bestIdx = static_cast<int>(i);
        }
        if (bestIdx < 0)
        {
            break;  // nothing left within reach of the chain -- stop rather than force a bad hop
        }
        const double newX = _waypoints[static_cast<size_t>(bestIdx)].x;
        const double newY = _waypoints[static_cast<size_t>(bestIdx)].y;
        const double hopDx = newX - cursorX;
        const double hopDy = newY - cursorY;
        const double hopLen = std::sqrt(hopDx * hopDx + hopDy * hopDy);
        if (hopLen > 1e-9)
        {
            headingX = hopDx / hopLen;
            headingY = hopDy / hopLen;
            haveHeading = true;
        }
        used[static_cast<size_t>(bestIdx)] = true;
        cursorX = newX;
        cursorY = newY;
        ordered.push_back(_waypoints[static_cast<size_t>(bestIdx)]);
    }
    return ordered;
}
}  // namespace fsd
