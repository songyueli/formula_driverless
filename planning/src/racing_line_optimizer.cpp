#include "racing_line_optimizer.hpp"

#include <algorithm>
#include <cmath>

namespace fsd
{
std::vector<PathPoint> OptimizeRacingLine(const std::vector<CorridorSample> &corridor, int iterations,
                                           double rate, double maxStep)
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

    for (int iter = 0; iter < iterations; ++iter)
    {
        std::vector<PathPoint> next = current;
        for (size_t i = 1; i + 1 < current.size(); ++i)
        {
            const double neighborMidX = (current[i - 1].x + current[i + 1].x) / 2.0;
            const double neighborMidY = (current[i - 1].y + current[i + 1].y) / 2.0;
            double pullX = rate * (neighborMidX - current[i].x);
            double pullY = rate * (neighborMidY - current[i].y);
            const double pullDist = std::sqrt(pullX * pullX + pullY * pullY);
            if (pullDist > maxStep)
            {
                const double scale = maxStep / pullDist;
                pullX *= scale;
                pullY *= scale;
            }
            const double candX = current[i].x + pullX;
            const double candY = current[i].y + pullY;

            // Project the pulled candidate's offset from this sample's
            // FIXED corridor centerline into (longitudinal, lateral)
            // relative to the corridor's own tangent, clamp only the
            // lateral component -- see this file's header comment.
            const CorridorSample &c = corridor[i];
            const double dx = candX - c.point.x;
            const double dy = candY - c.point.y;
            const double longitudinal = dx * c.tangentX + dy * c.tangentY;
            const double lateral = -dx * c.tangentY + dy * c.tangentX;  // left-positive
            const double clampedLateral = std::clamp(lateral, -c.halfWidth, c.halfWidth);
            const double leftNormalX = -c.tangentY;
            const double leftNormalY = c.tangentX;

            next[i].x = c.point.x + longitudinal * c.tangentX + clampedLateral * leftNormalX;
            next[i].y = c.point.y + longitudinal * c.tangentY + clampedLateral * leftNormalY;
        }
        current = std::move(next);
    }
    return current;
}
}  // namespace fsd
