#pragma once

#include <vector>

#include "corridor.hpp"

// Stage 4 (final) of the landmark-based racing-line pipeline: the actual
// least-curvature racing line, constrained to stay within stage 3's
// corridor.
namespace fsd
{
// Direct generalization of path_generator.cpp's already-validated
// MinimizeCurvature: same bounded iterative neighbor-pull (pull each
// interior sample toward the midpoint of its immediate neighbors -- a
// point exactly on the line between its neighbors has zero contribution to
// local curvature there, so repeated pulls straighten the path wherever it
// has room to, most visibly at the sharpest kinks first), same per-
// iteration step cap for numerical stability (an uncapped pull is
// confirmed live, in the reactive pipeline's own history, to be able to
// swing a path into an unrealistic shape when neighbor gaps are large).
//
// The difference from MinimizeCurvature is what a pull gets projected
// against: instead of EnforceMinClearance's per-cone push, each candidate
// position's offset from its corridor sample's OWN fixed centerline point
// is decomposed into longitudinal (along the corridor tangent) and lateral
// (perpendicular) components, and the lateral component is clamped to
// [-halfWidth, +halfWidth] -- the longitudinal component is left
// unclamped, since shifting a little along-track is exactly what
// smoothing is expected to do.
//
// Why this produces a REAL racing-line shape, not just smoothing:
// straightening the chord across a bend geometrically pulls entry/exit
// points toward the corridor's outside edge and the apex point toward the
// inside edge, because that's the shape that actually minimizes curvature
// between two points on either side of a corner when the only constraint
// is staying inside a bounded corridor -- the same mechanism already
// confirmed to widen the hairpin in the reactive pipeline (see
// MinimizeCurvature's own comment), now against a continuous corridor
// bound instead of discrete cone pushes.
//
// _iterations/_rate/_maxStep start from the reactive pipeline's own tuned
// values (see path_generator.cpp's kCurvatureSmoothing* constants) but are
// passed explicitly here, not reused as constants directly -- flagged as
// likely needing different values: dense spline samples sit much closer
// together than the sparse per-cycle midpoints those were tuned against,
// so the same absolute meter-based max-step behaves differently at finer
// sample spacing. Retune from live measurement, not by guessing harder.
std::vector<PathPoint> OptimizeRacingLine(const std::vector<CorridorSample> &corridor, int iterations,
                                           double rate, double maxStep);
}  // namespace fsd
