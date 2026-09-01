#include "centerline_extractor.hpp"

#include "path_utils.hpp"

namespace fsd
{
std::vector<PathPoint> TwoPointMidpointExtractor(const std::vector<WorldCone> &blue,
                                                  const std::vector<WorldCone> &yellow,
                                                  const Pose2D &vehiclePose)
{
    // Same mutual-nearest-neighbor requirement as
    // path_generator.cpp's NearestPairMidpointPath, and for the same
    // reason: at a sharp corner, the inside boundary's tighter cone
    // spacing means a blue landmark's own nearest yellow can easily be one
    // a DIFFERENT blue landmark is a strictly better match for -- a
    // genuine cross-pair, not noise. Requiring the match to be symmetric
    // costs an extra O(blue) reverse scan per candidate but means a bad
    // pair is skipped rather than accepted.
    std::vector<PathPoint> waypoints;
    waypoints.reserve(blue.size());
    for (const auto &l : blue)
    {
        const WorldCone *nearest = nullptr;
        double bestDistSq = kMaxPairDistance * kMaxPairDistance;
        for (const auto &r : yellow)
        {
            const double dx = r.x - l.x;
            const double dy = r.y - l.y;
            const double distSq = dx * dx + dy * dy;
            if (distSq < bestDistSq)
            {
                bestDistSq = distSq;
                nearest = &r;
            }
        }
        if (!nearest)
        {
            continue;
        }

        const WorldCone *reciprocal = nullptr;
        double reciprocalBestDistSq = kMaxPairDistance * kMaxPairDistance;
        for (const auto &l2 : blue)
        {
            const double dx = l2.x - nearest->x;
            const double dy = l2.y - nearest->y;
            const double distSq = dx * dx + dy * dy;
            if (distSq < reciprocalBestDistSq)
            {
                reciprocalBestDistSq = distSq;
                reciprocal = &l2;
            }
        }
        if (reciprocal != &l)
        {
            continue;  // not a mutual match -- likely a cross-pair, skip rather than accept
        }

        waypoints.push_back(PathPoint{(l.x + nearest->x) / 2.0, (l.y + nearest->y) / 2.0});
    }

    // Order into a travel-order chain starting from the vehicle's own
    // WORLD position (not the body-frame origin -- these are world-frame
    // midpoints), same reasoning as path_generator.cpp's own ordering step
    // for why a plain x-sort breaks at a hairpin.
    return OrderWaypointsByTraversal(std::move(waypoints), PathPoint{vehiclePose.x, vehiclePose.y});
}
}  // namespace fsd
