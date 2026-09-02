#include "lap_detector.hpp"

#include <cmath>

namespace fsd
{
namespace
{
// How close is "back at the start" -- tight enough that a hit genuinely
// means passing back through the start line, not just wandering into the
// same general area.
constexpr double kReturnRadius = 5.0;  // meters
// PRIMARY gate -- see this file's header comment for why distance-
// traveled, not just leave/return proximity, is required. First-guess
// value: this track's own landmark positions span over 100m in both x and
// y (see planning.cpp's own live-measured examples), so a real lap is
// almost certainly several hundred meters; 150m is comfortably past the
// small partial-arc distance (well under 50m) that produced the confirmed
// false-positive this gate exists to prevent, while still conservative
// enough not to require nearly a full lap's own worth of margin before
// even checking the return radius. Retune from the ACTUAL distance
// measured (via DistanceTraveled()) the first time a real lap completes
// live, same as every other first-attempt constant in this pipeline.
constexpr double kMinLapDistance = 150.0;  // meters
}  // namespace

void LapDetector::Update(double _worldX, double _worldY)
{
    if (m_lapComplete)
    {
        return;  // one-way latch -- see this file's header comment
    }
    if (!m_haveStart)
    {
        m_haveStart = true;
        m_startX = _worldX;
        m_startY = _worldY;
        m_haveLast = true;
        m_lastX = _worldX;
        m_lastY = _worldY;
        return;
    }

    if (m_haveLast)
    {
        const double stepX = _worldX - m_lastX;
        const double stepY = _worldY - m_lastY;
        m_distanceTraveled += std::sqrt(stepX * stepX + stepY * stepY);
    }
    m_lastX = _worldX;
    m_lastY = _worldY;
    m_haveLast = true;

    if (m_distanceTraveled < kMinLapDistance)
    {
        return;  // hasn't covered enough ground yet for a return hit to mean anything
    }

    const double dx = _worldX - m_startX;
    const double dy = _worldY - m_startY;
    if (dx * dx + dy * dy < kReturnRadius * kReturnRadius)
    {
        m_lapComplete = true;
    }
}
}  // namespace fsd
