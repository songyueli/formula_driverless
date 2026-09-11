#pragma once

#include <vector>

#include "landmark_map.hpp"
#include "path_generator.hpp"

// Stage 3 of the landmark-based racing-line pipeline: for each dense
// spline sample, how far laterally the racing-line optimizer (stage 4) is
// allowed to push it before it would leave the actual track. Frame-
// agnostic like the other stages.
namespace fsd
{
struct CorridorSample
{
    PathPoint point;             // spline sample position
    double tangentX, tangentY;   // unit tangent along the spline at this sample
    // Independent, asymmetric lateral bounds (left-positive convention,
    // blue=left/yellow=right -- see ComputeCorridor's own comment for why
    // this replaced a single symmetric halfWidth, 2026-09-02): how far the
    // racing-line optimizer may push this sample toward blue (leftBound)
    // and toward yellow (rightBound), independently. Each is already
    // (distance to that side's own nearest boundary cone) - safetyMargin,
    // so [-rightBound, +leftBound] is the actual clearance-safe interval,
    // not an approximation of it.
    double leftBound, rightBound;
};

// _safetyMargin subtracted from each side's own raw measured distance (same
// role as path_generator.hpp's kMinCarClearance for the reactive pipeline)
// -- BUG FIX (2026-09-02): used to be subtracted from a single
// min(blueDist, yellowDist), then the result clamped up to _minHalfWidth
// regardless of which side was actually tight. Confirmed live as the
// mechanism behind a real wedge: on this track's normal ~3.00m width, a
// centered sample's true per-side room (~1.5m) minus a correctly-sized
// safetyMargin (1.35m) is only ~0.15m -- BELOW _minHalfWidth (0.5m), so the
// floor silently overrode the real clearance requirement and let stage 4
// place the line up to 0.5m from a boundary cone anyway, well under the
// 1.35m it was supposed to guarantee. Computing each side independently
// (see CorridorSample's own comment) fixes this at the root instead of
// papering over it with EnforceMinClearance's post-hoc push: a floor that
// applies per side only kicks in when that side's chain has NO data at all
// (see the "no boundary data" branch below), never as a blanket override of
// a real, valid measurement. This also directly improves racing-line
// utilization: when the two sides aren't equidistant (the normal case
// anywhere but a razor-straight, perfectly centered stretch), stage 4 can
// now use the FULL room on the wider side up to _maxHalfWidth instead of
// being capped to whichever side is tighter, which a single symmetric bound
// always did even when there was no reason to restrict the wide side too.
// _maxHalfWidth still guards against a sparse-window region (few nearby
// landmarks) letting a bound balloon out to "no real constraint", which
// would let stage 4 cut a "racing line" through empty space the data simply
// doesn't cover rather than because the track is actually that wide there.
//
// See corridor.cpp for the full algorithm: ordered per-color boundary
// chains, walked with a monotonic per-color cursor and a small local
// index window around it (NOT a global nearest-neighbor search -- the
// same hairpin fold-back misattribution risk OrderWaypointsByTraversal
// already had to solve elsewhere: a global search could pick up a
// same-color landmark from across the fold, not the one that's actually
// beside this sample), plus a moving-average smoothing pass over each raw
// per-sample bound (piecewise by construction -- jumps whenever the
// nearest boundary landmark switches from one cone to the next as the
// sample index advances -- and an unsmoothed discontinuous bound would
// make stage 4's clamp oscillate against it).
// _closed: false (default) preserves the original windowed/open behavior
// exactly (tangent estimated via forward/backward difference clamped at
// the two ends, boundary-chain search never wraps). true is for the
// full-track closed loop (see lap_detector.hpp) -- _splineSamples is then
// expected to come from spline.hpp's FitAndSampleClosedSpline, and every
// neighbor lookup here (tangent's prev/next, and each color chain's local
// search window) wraps modulo the relevant array's own size instead of
// clamping at index 0/size-1, so the sample right after the last one is
// correctly treated as beside the sample right before the first one --
// there IS no "first"/"last" on a closed loop, only where the array
// happens to start.
//
// _orange: start/finish-gate cones (and any other orange on the track --
// see planning.cpp's ConeColorFromName comment). Deliberately excluded
// from centerline extraction upstream of this call (FSAE convention: you
// drive THROUGH the gate, not around it as a boundary), which means
// without this parameter the corridor here is defined purely by
// blue/yellow and can end up wider than the actual gate passage -- stage
// 4's optimizer would then be free to route arbitrarily close to a gate
// cone, discovering the problem only via EnforceMinClearance's much
// cruder post-hoc push on the ALREADY-optimized line. Passing it here
// instead lets the corridor itself narrow near the gate, so stage 4's own
// already-validated clamping logic keeps the racing line clear of it.
// Typically just 1-2 cones on the whole track (trackdrive.sdf has exactly
// 2), so this is a simple per-sample scan against every orange cone, not
// the chain/cursor machinery blue/yellow need -- no fold-back ambiguity
// risk with so few points. Deliberately kept as a SYMMETRIC tightening of
// both leftBound and rightBound (unlike the asymmetric blue/yellow
// treatment above) -- a gate cone close on just one side tightens the
// other side too, over-conservative but safe, and gate cones are rare
// enough that the utilization loss doesn't matter.
std::vector<CorridorSample> ComputeCorridor(const std::vector<PathPoint> &splineSamples,
                                             const std::vector<WorldCone> &blue,
                                             const std::vector<WorldCone> &yellow,
                                             const std::vector<WorldCone> &orange,
                                             double safetyMargin, double minHalfWidth,
                                             double maxHalfWidth, bool closed = false);
}  // namespace fsd
