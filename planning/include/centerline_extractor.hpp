#pragma once

#include <functional>
#include <vector>

#include "landmark_map.hpp"
#include "path_generator.hpp"

// Stage 1 of the landmark-based racing-line pipeline: turns a windowed set
// of blue/yellow world-frame landmarks into an ordered set of centerline
// midpoint candidates -- still in WORLD frame (the whole pipeline stays in
// world frame until the single conversion right before publish, see
// common/frame_transform.hpp). PathPoint is reused here for world-frame
// points, same as path_generator.hpp uses it for body-frame ones -- it's a
// bare coordinate pair with no frame baked into the type, same as
// ClassifiedCone communicates body-frame by comment rather than by type.
namespace fsd
{
// Swappable, mirroring track_boundaries.hpp's BoundaryExtractorFn /
// path_generator.hpp's PathGeneratorFn pattern already used by the
// reactive pipeline.
//
// FUTURE EXTENSION POINT: a Delaunay-triangulation-based extractor (blue-
// yellow-crossing triangle edge midpoints) was considered for this feature
// and deliberately deferred in favor of the simpler, already-validated
// two-point approach below -- swap out the caller's chosen
// MidpointExtractorFn to try it later without touching anything
// downstream (spline/corridor/curvature-min all just consume whatever
// ordered midpoints this stage produces).
using MidpointExtractorFn = std::function<std::vector<PathPoint>(
    const std::vector<WorldCone> &blue, const std::vector<WorldCone> &yellow,
    const Pose2D &vehiclePose)>;

// Default/current implementation: mutual-nearest-neighbor pairing between
// blue and yellow landmarks (same algorithm as path_generator.cpp's
// NearestPairMidpointPath pairing loop -- see its own comment for why
// MUTUAL matching, not one-directional, matters at a sharp corner),
// applied to a windowed WORLD-frame landmark set instead of one cycle's
// body-frame detections. Ordered via the shared OrderWaypointsByTraversal
// (path_utils.hpp), cursor = vehicle's world position.
std::vector<PathPoint> TwoPointMidpointExtractor(const std::vector<WorldCone> &blue,
                                                  const std::vector<WorldCone> &yellow,
                                                  const Pose2D &vehiclePose);
}  // namespace fsd
