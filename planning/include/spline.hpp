#pragma once

#include <vector>

#include "path_generator.hpp"

// Stage 2 of the landmark-based racing-line pipeline: fits a smooth curve
// through stage 1's ordered midpoints and samples it densely for stages
// 3/4. Frame-agnostic (PathPoint, no frame baked into the type -- see
// centerline_extractor.hpp's own comment) -- this module never needs to
// know or care which frame it's operating in.
namespace fsd
{
// Centripetal Catmull-Rom, NOT uniform-parametrized: stage 1's midpoint
// spacing is irregular (real cone spacing varies, and the mutual-nearest-
// neighbor pairing can skip a cone entirely rather than force a bad
// match), and uniform-parametrized Catmull-Rom is a known cusp/loop risk
// on unevenly-spaced control points -- a real risk here given how
// irregular this specific control-point source is, not a theoretical one.
// Centripetal parametrization (segment parameter step proportional to
// sqrt(distance) between consecutive points) avoids that.
//
// Sampled at approximately _sampleSpacing meters of ARC LENGTH (not a
// fixed parameter step), so short and long segments get proportionally
// many samples -- done by walking each segment at a fine fixed sub-step
// count first (no closed-form arc-length integral exists for Catmull-Rom),
// then resampling that dense discretization by cumulative distance.
//
// Needs at least 4 points to form even one interior segment (Catmull-Rom's
// 4-point-per-segment basis); fewer than that returns the input unchanged
// -- callers should treat this as "not enough data for a real spline yet"
// rather than an error.
std::vector<PathPoint> FitAndSampleSpline(const std::vector<PathPoint> &points, double sampleSpacing);
}  // namespace fsd
