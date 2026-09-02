#include "racing_line_optimizer.hpp"

#include <algorithm>
#include <cmath>

namespace fsd
{
PathPoint ClampToCorridor(const PathPoint &point, const CorridorSample &sample)
{
    const double dx = point.x - sample.point.x;
    const double dy = point.y - sample.point.y;
    const double longitudinal = dx * sample.tangentX + dy * sample.tangentY;
    const double lateral = -dx * sample.tangentY + dy * sample.tangentX;  // left-positive
    const double clampedLateral = std::clamp(lateral, -sample.halfWidth, sample.halfWidth);
    const double leftNormalX = -sample.tangentY;
    const double leftNormalY = sample.tangentX;

    PathPoint result;
    result.x = sample.point.x + longitudinal * sample.tangentX + clampedLateral * leftNormalX;
    result.y = sample.point.y + longitudinal * sample.tangentY + clampedLateral * leftNormalY;
    return result;
}

std::vector<PathPoint> OptimizeRacingLine(const std::vector<CorridorSample> &corridor, int iterations,
                                           double rate, double maxStep, bool closed)
{
    std::vector<PathPoint> current;
    current.reserve(corridor.size());
    for (const auto &c : corridor)
    {
        current.push_back(c.point);
    }
    if (current.size() < 3)
    {
        return current;  // nothing with an "interior" to pull
    }
    const int n = static_cast<int>(current.size());

    for (int iter = 0; iter < iterations; ++iter)
    {
        std::vector<PathPoint> next = current;
        // Closed: every index is "interior" (pulled), neighbors wrap
        // modulo n. Open (default): unchanged -- index 0 and n-1 stay
        // fixed anchors, only 1..n-2 is pulled. See this function's own
        // _closed comment in the header for why no anchor is needed in
        // the closed case.
        const int startIdx = closed ? 0 : 1;
        const int endIdx = closed ? n : n - 1;  // exclusive
        for (int i = startIdx; i < endIdx; ++i)
        {
            const size_t prevI = closed ? static_cast<size_t>((i - 1 + n) % n) : static_cast<size_t>(i - 1);
            const size_t nextI = closed ? static_cast<size_t>((i + 1) % n) : static_cast<size_t>(i + 1);
            const double neighborMidX = (current[prevI].x + current[nextI].x) / 2.0;
            const double neighborMidY = (current[prevI].y + current[nextI].y) / 2.0;
            double pullX = rate * (neighborMidX - current[static_cast<size_t>(i)].x);
            double pullY = rate * (neighborMidY - current[static_cast<size_t>(i)].y);
            const double pullDist = std::sqrt(pullX * pullX + pullY * pullY);
            if (pullDist > maxStep)
            {
                const double scale = maxStep / pullDist;
                pullX *= scale;
                pullY *= scale;
            }
            const double candX = current[static_cast<size_t>(i)].x + pullX;
            const double candY = current[static_cast<size_t>(i)].y + pullY;

            // Project the pulled candidate's offset from this sample's
            // FIXED corridor centerline into (longitudinal, lateral)
            // relative to the corridor's own tangent, clamp only the
            // lateral component -- see this file's header comment.
            next[static_cast<size_t>(i)] =
                ClampToCorridor(PathPoint{candX, candY}, corridor[static_cast<size_t>(i)]);
        }
        current = std::move(next);
    }
    return current;
}
}  // namespace fsd
