#pragma once

#include <cstdint>
#include <unordered_map>

#include "path_generator.hpp"

// Persists racing-line/corridor-boundary state across planning cycles,
// spatially indexed by a coarse world-frame grid, blending each cycle's
// freshly-computed value toward whatever's already cached via an
// exponential moving average. This is what lets the racing line
// "accumulate and get tuned" as the car re-observes the same track
// section (2026-08-31 user request), instead of being thrown away and
// recomputed from raw geometry every single cycle -- which is what was
// producing visible cycle-to-cycle jitter even when the underlying track
// geometry hadn't actually changed (a landmark's position refining by a
// few centimeters, or the mutual-nearest-neighbor pairing picking a
// slightly different candidate, was enough to reshuffle the WHOLE
// downstream shape every time, since nothing carried forward from the
// previous cycle's already-good answer).
namespace fsd
{
class RacingLineCache
{
public:
    // Blends _freshValue toward whatever's cached at (_worldX, _worldY)
    // and stores the result back for next cycle -- the FIRST time a given
    // grid cell is seen, there's nothing to blend toward yet, so the
    // fresh value is stored as-is (not blended toward an arbitrary
    // default). _alpha is how much of THIS cycle's fresh value to blend
    // in (0 = ignore fresh data entirely and never converge; 1 = no
    // smoothing at all, identical to not caching).
    PathPoint BlendAndStore(double _worldX, double _worldY, const PathPoint &_freshValue, double _alpha);

    // Overwrites the cached value at (_worldX, _worldY) directly, no
    // blending. For callers that post-process a BlendAndStore() result
    // (e.g. re-clamping it back inside a corridor bound) and need that
    // correction to persist into NEXT cycle's blend too -- otherwise the
    // cache keeps holding the pre-correction value and every future cycle
    // blends toward it again, fighting the correction instead of
    // incorporating it.
    void Overwrite(double _worldX, double _worldY, const PathPoint &_value);

private:
    static int64_t GridCellKey(double _x, double _y);
    std::unordered_map<int64_t, PathPoint> m_cache;
};
}  // namespace fsd
