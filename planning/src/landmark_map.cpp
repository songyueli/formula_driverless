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
//
// WIDENED -3.0 -> -8.0 (2026-09-05, user request: "increase the memory so
// that a cone seen earlier doesn't get forgotten" -- live investigation at
// the hairpin found a REAL, already-localized cone (ground truth: 0.98m
// from the car, roughly dead ahead) producing a completely EMPTY
// /cone_detections that cycle (YOLO/lidar near-field gap, not this file's
// own problem to fix -- see the session's own separate finding), yet the
// corridor still couldn't use it despite it already being a known landmark
// in the persistent map (LandmarkMap draws from every cone ever localized,
// not just the current cycle's fresh detections -- that's this whole
// class's own documented purpose). Root cause here: this is the EXACT SAME
// heading-lag mechanism already found and fixed once tonight in
// planning.cpp's RemoveBehindCarPoints, just in a different function that
// filter wasn't touching -- at a sharp/hairpin turn, the vehicle's own
// current heading can lag the track's local curvature by 90 degrees or
// more, so a landmark only a few meters away but past the apex projects to
// a heading-relative "forward" value far more negative than its own true
// distance would suggest (at 5m and 150 degrees of heading mismatch,
// forward = 5*cos(150deg) = -4.3m, already past the old -3.0m margin; a
// sharper mismatch pushes it further still). The landmark was never
// forgotten by the MAP (it's still in m_landmarks, added once and kept
// forever short of the EKF's own pruning) -- it was being excluded from
// THIS QUERY specifically by a directional filter that conflates "behind
// the car's current instantaneous heading" with "genuinely on an already-
// passed section of track", which is exactly the same conflation
// RemoveBehindCarPoints' own fix (2026-09-05, planning.cpp) already
// identified and corrected for the published-path case. -8.0m gives real
// margin for that heading-lag effect at a hairpin (comfortably covers the
// 5m/150deg case above, and even 8m at a near-180-degree mismatch) while
// staying well inside kWindowRadius=20.0m overall, so a genuinely-distant,
// actually-already-passed straight section still can't leak in just
// because this margin grew -- the original failure this constant guards
// against needed something on the order of tens of meters behind to
// reproduce, not single digits.
constexpr double kBehindMargin = -8.0;  // meters


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
