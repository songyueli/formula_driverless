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
std::vector<CorridorSample> ComputeCorridor(const std::vector<PathPoint> &splineSamples,
                                             const std::vector<WorldCone> &blue,
                                             const std::vector<WorldCone> &yellow,
                                             double safetyMargin, double minHalfWidth,
                                             double maxHalfWidth);
}  // namespace fsd
