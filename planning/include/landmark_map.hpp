#pragma once

#include <mutex>
#include <vector>

#include "common/frame_transform.hpp"
#include "common/types.hpp"

// Caches the latest /estimated_landmarks + /estimated_pose (both published
// by localization.cpp, world frame) and answers windowed queries for the
// landmark-based racing-line pipeline (racing_line_generator.cpp and
// friends). This is what lets planning draw from every cone the car has
// EVER seen and localized, not just whatever's in the camera's
// instantaneous view -- the whole point of this pipeline (see
// planning.cpp's own doc comment for the problem this solves).
//
// Deliberately NOT the same struct as path_generator.hpp's ClassifiedCone:
// that one is explicitly documented as body-frame, and reusing it here for
// world-frame data would blur exactly the frame distinction this module
// exists to get right.
namespace fsd
{
struct WorldCone
{
    double x;
    double y;
    ConeColor color;
};

class LandmarkMap
{
public:
    // Called from the /estimated_landmarks subscription callback.
    void UpdateLandmarks(std::vector<WorldCone> _landmarks);

    // Called from the /estimated_pose subscription callback.
    void UpdatePose(Pose2D _pose);

    struct WindowResult
    {
        std::vector<WorldCone> blue;
        std::vector<WorldCone> yellow;
        std::vector<WorldCone> orange;
        Pose2D vehiclePose;
        bool poseValid;
    };

    // Landmarks within _radius of the vehicle's current world position,
    // split by color -- bounds downstream cost (the full track can hold
    // hundreds of landmarks over a long drive) and avoids pulling in
    // geometrically-close-but-track-unrelated sections (parallel straights,
    // a track that loops near itself). orange landmarks are returned
    // separately, not merged into blue/yellow -- callers building the
    // centerline should exclude them (matches track_boundaries.hpp's
    // ColorSplitBoundaries, which does the same for the reactive pipeline)
    // but still use them for clearance.
    WindowResult QueryWindow(double _radius) const;

    // Every landmark ever seen, no radius/forward-facing filter -- used
    // once a lap completes (see lap_detector.hpp) to build the full-track
    // CLOSED-LOOP racing line instead of a local window. Reuses
    // WindowResult's shape (blue/yellow/orange + pose) even though nothing
    // here is actually "windowed" -- same fields callers already know how
    // to consume.
    WindowResult QueryAll() const;

private:
    mutable std::mutex m_mutex;
    std::vector<WorldCone> m_landmarks;
    Pose2D m_pose{0.0, 0.0, 0.0};
    bool m_poseValid = false;
};
}  // namespace fsd
