#pragma once

#include <vector>

#include "path_generator.hpp"
#include "track_boundaries.hpp"

// Shared path-shaping helpers used by BOTH the original per-cycle reactive
// pipeline (path_generator.cpp, body-frame) and the landmark-map-based
// racing-line pipeline (racing_line_optimizer.cpp and friends, world-frame
// internally, converted to body frame once right before publish -- see
// common/frame_transform.hpp). Promoted out of path_generator.cpp's own
// anonymous namespace specifically so the new pipeline can reuse them
// without duplicating already-live-tuned logic; behavior for the existing
// pipeline is unchanged (it calls OrderWaypointsByTraversal with the same
// {0,0} cursor it always implicitly used).
namespace fsd
{
// Real track width is >=3m (Formula Student rules) and a measured, perfectly
// consistent 3.00m on this specific track (see kAssumedHalfTrackWidth's own
// comment in path_generator.cpp) -- shared bound for "how far apart can two
// things legitimately be and still plausibly be directly related" (cone
// pairing, waypoint-traversal hops), not a per-user-of-this-header magic
// number re-derived twice.
constexpr double kMaxPairDistance = 8.0;  // meters

// See EnforceMinClearance/EnforceMinTurnRadius's own comments in
// path_utils.cpp for the full, live-tested justification of these two
// values (car half-length + cone radius + margin; vehicle's real min
// turning radius from model.sdf's AckermannSteering plugin).
constexpr double kMinCarClearance = 1.35;   // meters
constexpr double kMinTurnRadius = 4.5;      // meters

struct SegmentClosestPoint
{
    double t;  // [0,1] along the segment, clamped -- 0 = point A, 1 = point B
    double x, y;
};

SegmentClosestPoint ClosestPointOnSegment(double ax, double ay, double bx, double by,
                                           double px, double py);

// Pushes any waypoint (and the straight-line segment between each
// consecutive pair) that ends up within kMinCarClearance of any cone in
// _allCones directly away from it. See path_utils.cpp for the full
// reasoning (order-independent accumulate-then-apply, segment check, etc.)
std::vector<PathPoint> EnforceMinClearance(std::vector<PathPoint> waypoints,
                                            const std::vector<ClassifiedCone> &allCones);

// Clamps each waypoint's lateral offset so the arc from the origin (heading
// +X) through it never requires tighter than kMinTurnRadius -- see
// path_utils.cpp for the full derivation. Never drops a waypoint (see that
// file's own history for why an earlier drop-if-still-too-close-to-a-cone
// behavior was removed 2026-09-09): losing reach/continuity in the
// published path was confirmed worse than a rare, already-clearance-
// margined point sitting slightly tighter than ideal, and every caller here
// already has its own upstream clearance guarantee (EnforceMinClearance for
// the reactive pipeline, the corridor's own safety margin for the
// landmark-based one) that this function would otherwise be duplicating,
// imperfectly, with no caller-visible benefit.
std::vector<PathPoint> EnforceMinTurnRadius(std::vector<PathPoint> waypoints);

// Orders an unordered set of waypoints into travel order via greedy
// nearest-neighbor walking, starting from _cursor (the original reactive
// pipeline always started from the body-frame origin -- pass PathPoint{0,0}
// for that same behavior; the landmark-based pipeline starts from the
// vehicle's own world position instead). See path_utils.cpp for why a plain
// x-sort breaks at a hairpin.
//
// _maxHopDistance defaults to kMaxPairDistance -- every existing caller's
// behavior is unchanged. A caller ordering a FULL-TRACK single-color chain
// (not this project's original small-window use case) can pass a larger
// value: confirmed live (2026-08-31) as a real, not theoretical, need --
// this track's own blue boundary has a genuine cone-to-cone gap just over
// 8.0m somewhere (measured directly from live /estimated_landmarks data),
// which the default cap broke the chain at, leaving 38 of 99 blue
// landmarks completely unreached and truncating the closed-loop midpoint
// chain to barely more than half the track. Raise with real caution, not
// generously: this is still the same hairpin-fold-back safety margin the
// default value protects (see this function's own header comment) -- a
// same-color chain crossing back near itself at a tight hairpin apex could
// plausibly sit within a few meters, so an oversized cap risks the exact
// wrong-side hop this parameter exists to prevent. Use the smallest value
// confirmed (via direct replay against real landmark data) to actually
// close the chain in question, not an arbitrary generous bump.
//
// _haveInitialHeading/_initialHeadingX/_initialHeadingY (2026-09-01): seeds
// the directional-continuity check (path_utils.cpp's own comment on the
// heading-reversal rejection) with a known-good starting direction instead
// of deriving one from scratch off the first accepted hop. Confirmed live
// as necessary, not optional: leaving the first hop's OWN direction as the
// sole reference is fine when that hop is clean, but a single noisy first
// hop (ordinary landmark jitter, or a mutual-pair midpoint that's laterally
// offset from the true track direction) then wrongly rejects every
// following LEGITIMATE point that doesn't happen to align with that noisy
// reference -- observed live as the ordered-chain length intermittently
// collapsing from a healthy handful of points down to just one, cycle to
// cycle, with no change in the underlying cone geometry. Defaults to false
// (no initial heading, matching the original behavior before the heading
// check existed) -- callers with a genuinely known starting direction
// (the vehicle's own EKF-filtered yaw, or body-frame "forward" i.e. +X for
// the reactive pipeline's {0,0}-origin callers) should pass it; callers
// without one (e.g. corridor.cpp's BuildChain, cursor'd from a spline
// sample rather than the vehicle itself) are unaffected.
//
// _maxHeadingReversalDeg (2026-09-05): how far a hop's own direction may
// differ from the established heading before it's rejected as a likely
// fold-back rather than a real turn -- see path_utils.cpp's own comment at
// its declaration/history for the full reasoning. Defaults to 90.0, the
// value validated against DENSELY spline-resampled points (0.5m spacing,
// real hop-to-hop turn angle never exceeds ~24 degrees even at the
// sharpest corner on this track). A caller feeding this function RAW,
// coarsely-spaced points instead (cone spacing of several meters, not a
// resampled spline) needs a larger value -- a genuine hairpin apex can
// concentrate well over 90 degrees of turn into a single hop there purely
// from coarser sampling, not a fold-back mistake, and the default would
// silently truncate the chain right at that apex otherwise (confirmed
// live: centerline_extractor.cpp's TwoPointMidpointExtractor, the open/
// reactive pipeline's own extractor, is exactly such a caller -- see its
// own explicit override).
// _longHopDistance/_longHopMaxHeadingReversalDeg (2026-09-08, user
// principle: the published path should never be cut short -- but the two
// earlier attempts at that for TwoPointMidpointExtractor's own chain build
// showed the real tension isn't "short cap vs. long cap", it's that a
// SINGLE heading tolerance can't safely serve both jobs at once. A short,
// generously-toleranced hop (_maxHeadingReversalDeg, e.g. 150 degrees) is
// needed to get through a real hairpin apex's own sharp, coarse-cone-
// spacing turn -- confirmed necessary. But raising _maxHopDistance far
// enough to bridge a genuine "perception missed one cone" gap, while
// KEEPING that same generous tolerance, let a live run's greedy walk hop
// to a cone on the fold-back's OTHER leg instead -- confirmed live as a
// rollover (the published midpoint sequence jumped between the hairpin's
// two legs, not a continuous trace). The two failure modes share a hop-
// distance axis but need OPPOSITE tolerance behavior: short hops need to
// tolerate a large heading change (the apex itself), long hops need to
// DEMAND a small one (a genuine same-leg gap-bridge is still following the
// same gentle curve, just with a point missing -- it's the wrong-leg
// candidates that typically require the big heading swing). Any accepted
// hop farther than _longHopDistance must additionally pass
// _longHopMaxHeadingReversalDeg instead of the normal, looser
// _maxHeadingReversalDeg. Both default to values that make this a no-op
// for every existing caller (_longHopDistance defaults far beyond any real
// _maxHopDistance ever passed, so the tighter rule never engages unless a
// caller opts in explicitly).
std::vector<PathPoint> OrderWaypointsByTraversal(std::vector<PathPoint> _waypoints, PathPoint _cursor,
                                                  double _maxHopDistance = kMaxPairDistance,
                                                  bool _haveInitialHeading = false,
                                                  double _initialHeadingX = 0.0,
                                                  double _initialHeadingY = 0.0,
                                                  double _maxHeadingReversalDeg = 90.0,
                                                  double _longHopDistance = 1.0e9,
                                                  double _longHopMaxHeadingReversalDeg = 90.0);

struct ChainPair
{
    std::vector<PathPoint> blue;
    std::vector<PathPoint> yellow;
};

// Builds a blue/yellow chain PAIR (one OrderWaypointsByTraversal call per
// color, same parameters for both) and validates the result: a real
// track's two boundaries always run roughly parallel to each other, so if
// the two chains' own net travel directions (over their first several
// hops) point more than 90 degrees apart, one of them almost certainly
// took a wrong FIRST hop (2026-09-08, user report: "the planned path...
// too wide" right after the hairpin -- confirmed live via direct replay
// against a real capture: the yellow chain's hop 0 grabbed a nearby cone
// belonging to the hairpin's own just-exited inner/return leg -- still
// within kBehindMargin's own widened window and NOT rejected by any
// heading check, since hop 0 is deliberately unchecked (see
// OrderWaypointsByTraversal's own header comment for why THAT exemption
// exists) -- and every later hop then walked deeper along that wrong leg,
// each one self-consistent with the previous wrong hop. The result:
// blueChain correctly continued along the true next section while
// yellowChain instead traced back down into the hairpin, so the corridor
// built from them measured width against two chains that weren't even
// following the same stretch of track.
//
// Detected by comparing the two chains' own net directions to each
// other (should be near-parallel, not orthogonal-or-worse) rather than to
// the vehicle's heading directly -- the SAME single-hop-0 problem could
// affect either color, and a real turn can legitimately put either
// chain's own net direction well off the vehicle's instantaneous heading,
// so only a cross-color disagreement is a reliable signal something is
// actually wrong (not just "the track is turning here"). When flagged,
// rebuilds whichever chain agrees LESS with the vehicle's own heading
// (the more plausible candidate to have gone wrong) with its own
// first-hop choice excluded from the candidate pool, forcing the walk to
// consider a different start -- a single retry, not a loop: this targets
// the one confirmed failure mode (a bad hop 0), not a general search.
// Requires _haveInitialHeading (the vehicle's own known direction) to run
// the check at all -- without it there's no trustworthy tiebreaker, so
// this degrades to a plain, unvalidated chain-pair build, identical to
// calling OrderWaypointsByTraversal directly twice.
ChainPair BuildValidatedChainPair(const std::vector<PathPoint> &_blue, const std::vector<PathPoint> &_yellow,
                                   PathPoint _cursor, double _maxHopDistance, bool _haveInitialHeading,
                                   double _initialHeadingX, double _initialHeadingY, double _maxHeadingReversalDeg,
                                   double _longHopDistance = 1.0e9, double _longHopMaxHeadingReversalDeg = 90.0);
}  // namespace fsd
