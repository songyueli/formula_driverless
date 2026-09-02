#include "racing_line_cache.hpp"

#include <cmath>

namespace fsd
{
namespace
{
// Matches spline.hpp's own dense-sample spacing -- fine enough that a
// point genuinely on the same piece of track lands in the same cell (or
// an immediate neighbor) cycle to cycle, coarse enough that the cache
// doesn't explode into a huge number of near-duplicate cells for one
// track.
constexpr double kGridCellSize = 0.5;  // meters
}  // namespace

int64_t RacingLineCache::GridCellKey(double _x, double _y)
{
    const int64_t gx = static_cast<int64_t>(std::floor(_x / kGridCellSize));
    const int64_t gy = static_cast<int64_t>(std::floor(_y / kGridCellSize));
    // Same pack-two-32-bit-halves-into-one-int64 pattern as ekf.cpp's own
    // RetiredGridCellKey, including WHY it goes through uint64_t first:
    // left-shifting a NEGATIVE signed value is undefined behavior pre-
    // C++20 (this file is C++17), and gx/gy are legitimately negative for
    // any world position south/west of the origin -- genuinely true on
    // this track (real driven positions go down to roughly x=-56, y=-37),
    // not a theoretical edge case. Converting to uint64_t first is
    // well-defined (standard signed->unsigned conversion, effectively mod
    // 2^64), and shifting/OR-ing unsigned values is always well-defined
    // regardless of the original sign.
    const uint64_t ugx = static_cast<uint64_t>(gx);
    const uint64_t ugy = static_cast<uint64_t>(gy);
    return static_cast<int64_t>((ugx << 32) | (ugy & 0xFFFFFFFFULL));
}

PathPoint RacingLineCache::BlendAndStore(double _worldX, double _worldY, const PathPoint &_freshValue,
                                          double _alpha)
{
    const int64_t key = GridCellKey(_worldX, _worldY);
    auto it = m_cache.find(key);
    PathPoint blended;
    if (it == m_cache.end())
    {
        blended = _freshValue;  // nothing to blend toward yet
    }
    else
    {
        blended.x = it->second.x + _alpha * (_freshValue.x - it->second.x);
        blended.y = it->second.y + _alpha * (_freshValue.y - it->second.y);
    }
    m_cache[key] = blended;
    return blended;
}

void RacingLineCache::Overwrite(double _worldX, double _worldY, const PathPoint &_value)
{
    m_cache[GridCellKey(_worldX, _worldY)] = _value;
}
}  // namespace fsd
