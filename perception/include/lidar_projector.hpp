#pragma once

#include <gz/msgs/pointcloud_packed.pb.h>
#include <opencv2/core.hpp>

#include <optional>
#include <vector>

// Matches lidar points (gz's gpu_lidar PointCloudPacked, in the sensor's
// own frame) against 2-D cone detections from the cylindrical panorama, to
// recover each cone's 3-D position in the car (chassis) frame.
//
// The lidar and every camera are mounted on the SAME chassis link with
// known, fixed offsets (see model.sdf) -- no estimation needed, same as
// everywhere else in this pipeline: lidar at (0.3, 0, 0.35), zero rotation;
// front camera (the panorama's reference frame) at (0.9, 0, 1.0), pitched
// down 0.3 rad. SetPointCloud() converts each lidar point once into BOTH
// the chassis frame (what gets returned) and the panorama's own pixel
// space (cached, so Localize() -- called once per detection, of which
// there can be dozens per frame -- is a cheap linear scan rather than
// re-deriving the projection every time).
class LidarProjector
{
public:
    struct ConePosition
    {
        float x, y, z; // car (chassis) frame, meters
        float range;   // distance from the car's own origin, meters
    };

    // panoWidth/panoHeight/fPano must exactly match the CameraStitcher this
    // panorama came from -- fPano is pixels/radian (cylindrical), the same
    // value as CameraStitcher's own internal focal length (K.fx).
    LidarProjector(int panoWidth, int panoHeight, double fPano);

    void SetPointCloud(const gz::msgs::PointCloudPacked &msg);

    // bbox in panorama pixel space. Returns the closest (by range) lidar
    // point whose projection falls inside it, or nullopt if none do -- a
    // detection can legitimately have no match (a thin/distant part of a
    // cone with no lidar return there), and the lidar (10 Hz) isn't
    // synchronized against detections the way the 3 cameras are
    // synchronized against each other (30 Hz) -- see perception.cpp.
    std::optional<ConePosition> Localize(float x1, float y1, float x2, float y2) const;

private:
    struct ProjectedPoint
    {
        float u, v;       // panorama pixel space
        ConePosition pos; // chassis frame + range
    };

    int m_panoWidth;
    int m_panoHeight;
    double m_fPano;
    std::vector<ProjectedPoint> m_points;
};
