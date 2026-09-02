#include "landmark_map.hpp"

#include <cmath>

namespace fsd
{
// How far BEHIND the vehicle (along its own current heading) a landmark
// can still sit and be included in the window -- confirmed directly as
// necessary, not just defensive: an isotropic (full-circle) window pulled
// in landmarks from track sections the car had ALREADY driven through
// (behind it), which the ordering/pairing logic downstream has no way to
// distinguish from landmarks genuinely ahead. That produced a live,
// reproducible failure: the extracted midpoint chain started with a point
// BEHIND the vehicle (nearest by raw distance, but the wrong direction),
// so the published path pointed backward before swinging around to the
// real forward direction -- none of its early waypoints ever reached pure
// pursuit's lookahead distance, and the car stalled. A small negative
// margin (rather than a hard cutoff at exactly 0) keeps landmarks just
// barely behind the vehicle's origin available for pairing continuity
// right around the car, matching the kind of small tolerance used
// elsewhere in this codebase rather than a knife-edge boundary.
constexpr double kBehindMargin = -3.0;  // meters


void LandmarkMap::UpdateLandmarks(std::vector<WorldCone> _landmarks)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_landmarks = std::move(_landmarks);
}

void LandmarkMap::UpdatePose(Pose2D _pose)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_pose = _pose;
    m_poseValid = true;
}

LandmarkMap::WindowResult LandmarkMap::QueryWindow(double _radius) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    WindowResult result;
    result.vehiclePose = m_pose;
    result.poseValid = m_poseValid;

    const double radiusSq = _radius * _radius;
    const double cosYaw = std::cos(m_pose.yaw);
    const double sinYaw = std::sin(m_pose.yaw);
    for (const auto &cone : m_landmarks)
    {
        const double dx = cone.x - m_pose.x;
        const double dy = cone.y - m_pose.y;
        if (dx * dx + dy * dy > radiusSq)
        {
            continue;
        }
        // Forward-facing filter -- see kBehindMargin's own comment.
        const double forward = dx * cosYaw + dy * sinYaw;
        if (forward < kBehindMargin)
        {
            continue;
        }
        switch (cone.color)
        {
            case ConeColor::Blue:   result.blue.push_back(cone);   break;
            case ConeColor::Yellow: result.yellow.push_back(cone); break;
            case ConeColor::Orange: result.orange.push_back(cone); break;
            default: break;  // Unknown -- excluded from all three, same as
                              // track_boundaries.hpp's ColorSplitBoundaries
                              // dropping anything that isn't blue/yellow.
        }
    }
    return result;
}

LandmarkMap::WindowResult LandmarkMap::QueryAll() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    WindowResult result;
    result.vehiclePose = m_pose;
    result.poseValid = m_poseValid;
    for (const auto &cone : m_landmarks)
    {
        switch (cone.color)
        {
            case ConeColor::Blue:   result.blue.push_back(cone);   break;
            case ConeColor::Yellow: result.yellow.push_back(cone); break;
            case ConeColor::Orange: result.orange.push_back(cone); break;
            default: break;
        }
    }
    return result;
}
}  // namespace fsd
