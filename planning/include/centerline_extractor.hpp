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
    const std::vector<WorldCone> &orange, const Pose2D &vehiclePose)>;

// Default/current implementation: mutual-nearest-neighbor pairing between
// blue and yellow landmarks (same algorithm as path_generator.cpp's
// NearestPairMidpointPath pairing loop -- see its own comment for why
// MUTUAL matching, not one-directional, matters at a sharp corner),
// applied to a windowed WORLD-frame landmark set instead of one cycle's
// body-frame detections. Ordered via the shared OrderWaypointsByTraversal
// (path_utils.hpp), cursor = vehicle's world position.
//
// `orange` (2026-09-04, user report: "the algorithm doesn't handle orange
// cones properly" -- confirmed live: the two nearest LEGITIMATE blue-
// yellow-pair midpoints straddling the real start/finish gate sat 3.3-3.5m
// away from the gate's own center on EITHER side, a ~6.5m hole in
// centerline coverage exactly where the gate is, since blue/yellow cones
// are sparse/absent running THROUGH a real gate -- OrderWaypointsByTraversal
// then has to blindly bridge that gap with no real anchor point there, a
// plausible source of an off-track-looking "veer" unrelated to any actual
// curvature). Pairs up orange cones the same mutual-nearest-neighbor way as
// blue/yellow and adds each pair's own midpoint as a genuine waypoint
// anchoring the gate's true center, closing that gap directly instead of
// leaving it for downstream stages to paper over.
std::vector<PathPoint> TwoPointMidpointExtractor(const std::vector<WorldCone> &blue,
                                                  const std::vector<WorldCone> &yellow,
                                                  const std::vector<WorldCone> &orange,
                                                  const Pose2D &vehiclePose);

// Full-track counterpart, used once a lap completes (see lap_detector.hpp)
// when blue/yellow span the WHOLE track (~100-200+ candidates each)
// instead of a small local window. TwoPointMidpointExtractor's GLOBAL
// mutual-nearest-neighbor search doesn't scale to that: confirmed live
// (2026-08-31) that at full-track scale, far more pairs fail the strict
// reciprocal-match requirement (many more candidates competing globally
// for each match, vs. a handful in a local window), and
// OrderWaypointsByTraversal's own hop-distance cap then stops the
// resulting chain early at the first real gap -- only 11 of ~200+ raw
// cones survived, forming a short arc, not a loop.
//
// This extractor never does a global search at all: order EACH COLOR's
// cones into its own chain independently first (OrderWaypointsByTraversal,
// same proven technique corridor.cpp's BuildChain already uses
// successfully at this same full-track scale), then walk the blue chain
// in its own already-correct order and, for each point, find its paired
// yellow via a LOCAL, cursor-advancing index window into the yellow chain
// -- the same anti-fold-back discipline corridor.cpp's
// NearestChainLateralDistance already relies on (a window can only ever
// advance a few indices per step, so it can't jump across a hairpin
// fold-back to a same-color-adjacent-but-wrong-side candidate). Both
// chains' own traversal starts near the vehicle's current position, so
// their index-0 entries are already reasonably close to each other --
// a sound starting alignment for the cursor. Output is implicitly in
// travel order already (blueChain's own order), no separate
// OrderWaypointsByTraversal pass needed on the result.
// `orange` -- same gate-anchor addition as TwoPointMidpointExtractor's own
// (see its own comment), applied here too so the closed-loop spline gets
// the same real anchor point at the gate once a lap completes.
std::vector<PathPoint> ClosedLoopMidpointExtractor(const std::vector<WorldCone> &blue,
                                                    const std::vector<WorldCone> &yellow,
                                                    const std::vector<WorldCone> &orange,
                                                    const Pose2D &vehiclePose);
}  // namespace fsd
