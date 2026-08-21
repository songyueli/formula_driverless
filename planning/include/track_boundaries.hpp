#pragma once

#include <functional>
#include <string>
#include <vector>

namespace fsd
{
// One detected cone: position (car body frame, meters) plus its class name
// (matches perception's kClassNames -- "blue"/"yellow"/"orange"/
// "large_orange").
struct ClassifiedCone
{
    double x;
    double y;
    std::string colorClass;
};

// Track boundaries as two point sets. Deliberately no ordering guarantee
// here -- whether left/right come out ordered (e.g. nearest-first) is a
// property of the specific extractor used, not something callers can
// assume. The default extractor below produces them in whatever order
// perception's detections arrived in; a path generator that needs
// ordering must produce its own (see path_generator.hpp).
struct TrackBoundaries
{
    std::vector<ClassifiedCone> left;
    std::vector<ClassifiedCone> right;
};

// A BoundaryExtractor turns a raw, per-cycle set of classified cone
// detections into left/right track boundaries. This is one of the two
// swappable stages of planning (see path_generator.hpp for the other) --
// see planning.cpp's kActiveBoundaryExtractor, the one line to change to
// try a different algorithm.
//
// FUTURE EXTENSION POINT: a Delaunay-triangulation-based extractor would
// implement this same signature, but could ignore colorClass entirely --
// triangulate ALL cone positions, then classify each triangulation edge as
// "inside the track" (short, roughly perpendicular to the direction of
// travel) or "boundary" based on edge geometry rather than color, which is
// more robust to perception color misclassification than this default
// implementation. It would still need to partition cones into "left of the
// drivable corridor" / "right of the drivable corridor" to satisfy this
// same TrackBoundaries return shape, so path generators downstream don't
// need to change.
using BoundaryExtractorFn = std::function<TrackBoundaries(const std::vector<ClassifiedCone> &)>;

// Default/current implementation -- trusts perception's own color
// classification directly: left = all "blue" detections, right = all
// "yellow" detections (FSAE convention: blue marks the left boundary,
// yellow the right, viewed driving forward). orange/large_orange (start/
// finish markers) are dropped here, same as the original (pre-split)
// algorithm.
TrackBoundaries ColorSplitBoundaries(const std::vector<ClassifiedCone> &cones);
}  // namespace fsd
