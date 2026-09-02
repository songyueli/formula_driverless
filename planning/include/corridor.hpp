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
    double halfWidth;            // meters, lateral bound from the centerline
};

// _safetyMargin subtracted from the raw measured half-width (same role as
// path_generator.hpp's kMinCarClearance for the reactive pipeline);
// _minHalfWidth/_maxHalfWidth clamp the result -- a floor guards against a
// single jittery/mislocalized landmark collapsing the corridor to nothing,
// a ceiling guards against a sparse-window region (few nearby landmarks)
// letting the bound balloon out to "no real constraint", which would let
// stage 4 cut a "racing line" through empty space the data simply doesn't
// cover rather than because the track is actually that wide there.
//
// See corridor.cpp for the full algorithm: ordered per-color boundary
// chains, walked with a monotonic per-color cursor and a small local
// index window around it (NOT a global nearest-neighbor search -- the
// same hairpin fold-back misattribution risk OrderWaypointsByTraversal
// already had to solve elsewhere: a global search could pick up a
// same-color landmark from across the fold, not the one that's actually
// beside this sample), plus a moving-average smoothing pass over the raw
// per-sample half-width (it's piecewise by construction -- jumps whenever
// the nearest boundary landmark switches from one cone to the next as the
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
// risk with so few points. halfWidth stays a single SYMMETRIC bound (see
// CorridorSample), so a gate cone close on just one side also tightens
// the other side -- over-conservative but safe.
std::vector<CorridorSample> ComputeCorridor(const std::vector<PathPoint> &splineSamples,
                                             const std::vector<WorldCone> &blue,
                                             const std::vector<WorldCone> &yellow,
                                             const std::vector<WorldCone> &orange,
                                             double safetyMargin, double minHalfWidth,
                                             double maxHalfWidth, bool closed = false);
}  // namespace fsd
