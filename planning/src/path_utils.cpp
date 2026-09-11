#include "path_utils.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

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
// BUG FIX (2026-09-02): this used to unconditionally keep every clamped
// point, even when the clamp left it back inside kMinCarClearance of a
// cone -- confirmed live, twice, as the actual mechanism behind two
// separate car-wedged-against-a-cone events. A drop-if-still-unsafe check
// was added in response (see git history), then progressively narrowed
// (2026-09-03, 2026-09-05) after each narrowing itself caused a new,
// confirmed-live failure: dropping far-field points it was never meant to
// touch, and emptying the whole array in a genuinely tight, low-data
// window, both traced directly to this same check.
//
// REMOVED entirely (2026-09-09, user report: "the planned path is cutoff,
// causing the car to stop moving forward... we should be drawing a
// planned path that has the same reach as the debug racing line"). Live
// evidence: a captured /planned_path vs /planning/debug_racing_line pair
// showed 5 points dropped by this exact check, leaving a ~3.4m gap right
// at pure pursuit's own lookahead target -- the car lost its
// fine-grained near-term target exactly where the corner was tightest,
// then wedged against a cone shortly after. The drop was never a
// clearance improvement (EnforceMinClearance, run by the reactive
// pipeline's own callers, and the corridor's own safety margin for the
// landmark-based one, already provide that upstream) -- it only ever
// removed points, trading continuity for a clearance re-check this
// function has no unique ability to act on anyway (it can only skip a
// point, not actually move it somewhere safer once the turn-radius clamp
// has already fixed its lateral position). Losing reach is now the
// confirmed worse failure mode of the two. Just clamp; never drop.
std::vector<PathPoint> EnforceMinTurnRadius(std::vector<PathPoint> waypoints)
{
    for (auto &wp : waypoints)
    {
        const double absX = std::abs(wp.x);
        if (absX < kMinTurnRadius)
        {
            const double maxAbsY = kMinTurnRadius - std::sqrt(kMinTurnRadius * kMinTurnRadius - absX * absX);
            if (std::abs(wp.y) > maxAbsY)
            {
                wp.y = std::copysign(maxAbsY, wp.y);
            }
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
//
// NOW A PARAMETER, not a fixed constant (2026-09-05, user report: repeated
// "stuck on a cone" at the hairpin traced to /planned_path getting
// truncated -- confirmed root cause: TwoPointMidpointExtractor
// (centerline_extractor.cpp, the OPEN pipeline's own extractor) calls this
// function on RAW, cone-spaced midpoints with NO chain/cursor structure of
// its own, unlike this file's other callers. The 90 degree default right
// above was measured and validated against the CLOSED-loop pipeline's
// densely spline-resampled points (0.5m spacing), where real hop-to-hop
// turn angle never exceeds ~24 degrees even at the sharpest corner --
// TwoPointMidpointExtractor's own midpoints are spaced by raw CONE
// distance instead (several meters apart, whatever this track's actual
// cone spacing is), so the SAME physical hairpin can require a single hop
// between two entirely legitimate, correctly-paired midpoints to turn
// through well over 90 degrees, purely from coarser sampling -- not a
// fold-back mistake. The 90 degree threshold, safely conservative for the
// dense case, was silently terminating the greedy walk right at the
// hairpin apex for the sparse case, dropping every real midpoint after it
// and truncating the published path exactly where the car needed it most.
// Defaults to kMaxHeadingReversalDeg (90.0) so every EXISTING call site
// (this file's own other three) keeps its exact validated behavior;
// TwoPointMidpointExtractor's own call now passes a larger, explicit value
// -- see its own call site for the specific number and the accepted
// tradeoff (this loosens protection against a hairpin fold-back hop for
// that one caller, a different code path from the one the 118-degree
// confirmed-bad-hop history above was ever measured against). 90.0 itself
// now lives only as this function's own default argument value (see
// path_utils.hpp) since a default argument must be visible where the
// function is declared, not buried in this .cpp file's own anonymous
// namespace.

std::vector<PathPoint> OrderWaypointsByTraversal(std::vector<PathPoint> _waypoints, PathPoint _cursor,
                                                  double _maxHopDistance, bool _haveInitialHeading,
                                                  double _initialHeadingX, double _initialHeadingY,
                                                  double _maxHeadingReversalDeg, double _longHopDistance,
                                                  double _longHopMaxHeadingReversalDeg)
{
    const double minHeadingDot = std::cos(_maxHeadingReversalDeg * M_PI / 180.0);
    // See this function's own header comment (path_utils.hpp) for why a
    // long hop needs a TIGHTER heading tolerance than a short one, not the
    // same or looser one.
    const double longHopMinHeadingDot = std::cos(_longHopMaxHeadingReversalDeg * M_PI / 180.0);
    const double longHopDistanceSq = _longHopDistance * _longHopDistance;

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
            // Skip the heading check on the very FIRST hop even when a seeded
            // heading exists -- confirmed live (2026-09-02) as a real,
            // reproducible regression from the seeding fix itself: the seed
            // is the VEHICLE's raw nose direction, not a track direction, and
            // the car isn't always sitting exactly on the racing line facing
            // squarely at the next waypoint -- it can be positioned such
            // that the nearest genuine, correct next point is laterally
            // placed (bearing >90 degrees off the nose) rather than dead
            // ahead. Confirmed directly against a live capture: the correct
            // mutual-nearest-neighbor pair only ~1.8m from the car had
            // dot=-0.63 against the vehicle's own heading (a ~128 degree
            // bearing), so EVERY candidate was rejected on hop 0, the chain
            // stayed empty, and /planned_path went empty with the car
            // stopped -- despite plentiful, well-distributed real landmark
            // data nearby (not a data-starvation case). The seeded heading's
            // actual job is only to give HOP 1's decision a stable reference
            // instead of trusting hop 0's own possibly-noisy direction (see
            // this function's own header comment for that original
            // regression) -- it was never meant to gate hop 0 itself, which
            // the original design intentionally left as pure nearest-
            // neighbor (see the header comment's own "very first hop... has
            // no established direction" line, which this restores as
            // literally true again for the SEEDED case, not just the
            // no-seed case).
            const bool checkHeading = haveHeading && step > 0;
            if (checkHeading && distSq > 1e-9)
            {
                const double invLen = 1.0 / std::sqrt(distSq);
                const double dot = dx * invLen * headingX + dy * invLen * headingY;
                // A hop past _longHopDistance must clear the TIGHTER
                // long-hop threshold, not the normal (possibly much more
                // generous) one -- see this function's own header comment
                // for why: a genuine same-leg gap-bridge stays close to the
                // established heading even over a longer span, while a
                // wrong-leg hairpin-fold hop typically needs a much bigger
                // swing to reach. Defaults make this identical to the old
                // single-threshold check for every existing caller.
                const double effectiveMinDot = (distSq > longHopDistanceSq) ? longHopMinHeadingDot : minHeadingDot;
                if (dot < effectiveMinDot)
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
        // Don't let hop 0's own direction overwrite a seeded heading: hop 0
        // was just accepted UNFILTERED (see checkHeading above), so its
        // direction can be arbitrary -- adopting it here would clobber the
        // seed before hop 1 ever gets to use it, silently reintroducing the
        // exact "noisy first hop poisons every later hop" collapse this
        // seeding fix exists to prevent (see this function's header
        // comment). Keep the seed alive through hop 1's own check; from hop
        // 1 onward every accepted hop already passed a heading check, so
        // adopting its own direction here is trustworthy again.
        const bool keepSeedForNextCheck = _haveInitialHeading && step == 0;
        if (hopLen > 1e-9 && !keepSeedForNextCheck)
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

namespace
{
// How many hops into a chain to look when judging its own net travel
// direction -- see BuildValidatedChainPair's own header comment (in
// path_utils.hpp) for the full reasoning. A handful of hops smooths out
// any single-hop jitter without diluting the signal with points from far
// enough away that local curvature alone could explain a direction change.
constexpr int kChainDivergenceCheckHops = 4;
// Cross-color net-direction dot-product floor -- 0.0 is a full 90 degrees,
// already generous (two genuinely parallel boundaries should track much
// closer than that even through a real bend); a same-track pair should
// essentially never legitimately fall below this, so treating anything
// under it as "diverged, one chain probably took a wrong first hop" is a
// safe, low-false-positive trigger.
constexpr double kChainDivergenceMinDot = 0.0;

// (dx, dy) of a chain's own net travel direction over its first
// kChainDivergenceCheckHops points, normalized -- {0,0} (a sentinel, not a
// real direction) if the chain is too short to judge.
std::pair<double, double> ChainNetDirection(const std::vector<PathPoint> &_chain)
{
    const size_t n = std::min(_chain.size(), static_cast<size_t>(kChainDivergenceCheckHops + 1));
    if (n < 2)
    {
        return {0.0, 0.0};
    }
    const double dx = _chain[n - 1].x - _chain[0].x;
    const double dy = _chain[n - 1].y - _chain[0].y;
    const double len = std::hypot(dx, dy);
    if (len < 1e-6)
    {
        return {0.0, 0.0};
    }
    return {dx / len, dy / len};
}
}  // namespace

ChainPair BuildValidatedChainPair(const std::vector<PathPoint> &_blue, const std::vector<PathPoint> &_yellow,
                                   PathPoint _cursor, double _maxHopDistance, bool _haveInitialHeading,
                                   double _initialHeadingX, double _initialHeadingY, double _maxHeadingReversalDeg,
                                   double _longHopDistance, double _longHopMaxHeadingReversalDeg)
{
    // _excludePoint: when set, that single point is dropped from the
    // candidate list before ordering -- used for the one-shot retry below,
    // to force the greedy walk's own hop 0 to consider a different start
    // once the original choice is confirmed suspicious.
    auto buildChain = [&](const std::vector<PathPoint> &_cones, const PathPoint *_excludePoint)
    {
        if (_excludePoint == nullptr)
        {
            return OrderWaypointsByTraversal(_cones, _cursor, _maxHopDistance, _haveInitialHeading,
                                              _initialHeadingX, _initialHeadingY, _maxHeadingReversalDeg,
                                              _longHopDistance, _longHopMaxHeadingReversalDeg);
        }
        std::vector<PathPoint> filtered;
        filtered.reserve(_cones.size());
        for (const auto &p : _cones)
        {
            if (std::hypot(p.x - _excludePoint->x, p.y - _excludePoint->y) > 1e-6)
            {
                filtered.push_back(p);
            }
        }
        return OrderWaypointsByTraversal(std::move(filtered), _cursor, _maxHopDistance, _haveInitialHeading,
                                          _initialHeadingX, _initialHeadingY, _maxHeadingReversalDeg,
                                          _longHopDistance, _longHopMaxHeadingReversalDeg);
    };

    std::vector<PathPoint> blueChain = buildChain(_blue, nullptr);
    std::vector<PathPoint> yellowChain = buildChain(_yellow, nullptr);

    if (_haveInitialHeading)
    {
        const auto [bdx, bdy] = ChainNetDirection(blueChain);
        const auto [ydx, ydy] = ChainNetDirection(yellowChain);
        const bool haveBothDirections = (bdx != 0.0 || bdy != 0.0) && (ydx != 0.0 || ydy != 0.0);
        if (haveBothDirections && (bdx * ydx + bdy * ydy) < kChainDivergenceMinDot)
        {
            // Diverged -- rebuild whichever chain agrees LESS with the
            // vehicle's own known heading, excluding its own first point.
            const double blueVehicleDot = bdx * _initialHeadingX + bdy * _initialHeadingY;
            const double yellowVehicleDot = ydx * _initialHeadingX + ydy * _initialHeadingY;
            if (blueVehicleDot < yellowVehicleDot && !blueChain.empty())
            {
                blueChain = buildChain(_blue, &blueChain.front());
            }
            else if (!yellowChain.empty())
            {
                yellowChain = buildChain(_yellow, &yellowChain.front());
            }
        }
    }

    return ChainPair{std::move(blueChain), std::move(yellowChain)};
}
}  // namespace fsd
