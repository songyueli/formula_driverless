#pragma once

#include <functional>
#include <vector>

#include "track_boundaries.hpp"

namespace fsd
{
struct PathPoint
{
    double x;
    double y;
};

// A PathGenerator turns track boundaries into an ordered path (nearest-
// ahead-first, body frame -- see planning.cpp's own doc comment for why
// body frame). This is the other swappable stage of planning (see
// track_boundaries.hpp for the first) -- see planning.cpp's
// kActivePathGenerator, the one line to change to try a different
// algorithm.
//
// FUTURE EXTENSION POINT: a real centerline generator would implement this
// same signature -- e.g. Delaunay-triangulate the boundaries' combined
// points, walk the triangulation's midline edges to get an ordered,
// evenly-spaced centerline (a strict improvement over this default's raw
// nearest-pair midpoints, which can zigzag with cone-spacing
// irregularities -- see the TODO in path_generator.cpp). Racing-line
// optimization (e.g. minimum-curvature) would be a further step built on
// top of whatever centerline a first pass like that produces -- still
// something that fits behind this same signature, since it's still "given
// boundaries, produce an ordered path," just a fancier path. Either way
// the return type stays std::vector<PathPoint>, which is what
// control.cpp's pure-pursuit consumes.
using PathGeneratorFn = std::function<std::vector<PathPoint>(const TrackBoundaries &)>;

// Default/current implementation: pair each left-boundary cone with its
// nearest right-boundary cone (skip pairs farther apart than
// kMaxPairDistance -- see .cpp), take the midpoint, sort by forward (body-
// frame x) distance. Exactly planning.cpp's original algorithm, unchanged.
std::vector<PathPoint> NearestPairMidpointPath(const TrackBoundaries &boundaries);
}  // namespace fsd
