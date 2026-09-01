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
// +X) through it never requires tighter than kMinTurnRadius. See
// path_utils.cpp for the full derivation.
std::vector<PathPoint> EnforceMinTurnRadius(std::vector<PathPoint> waypoints);

// Orders an unordered set of waypoints into travel order via greedy
// nearest-neighbor walking, starting from _cursor (the original reactive
// pipeline always started from the body-frame origin -- pass PathPoint{0,0}
// for that same behavior; the landmark-based pipeline starts from the
// vehicle's own world position instead). See path_utils.cpp for why a plain
// x-sort breaks at a hairpin.
std::vector<PathPoint> OrderWaypointsByTraversal(std::vector<PathPoint> _waypoints,
                                                  PathPoint _cursor);
}  // namespace fsd
