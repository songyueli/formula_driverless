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
// everywhere else in this pipeline: lidar at (0.3, 0, 0.6) -- raised from
// the original 0.35 to give it a better sightline instead of a shallow
// grazing angle onto the ground and any cone at range, which was letting
// lidar returns from the WRONG object win Localize()'s closest-range
// comparison (see this file's own comment on that further down, and
// lidar_projector.cpp's kLidarZ) -- zero rotation; front camera (the
// panorama's reference frame) at (0.9, 0, 1.0), pitched down 0.3 rad.
// SetPointCloud() converts each lidar point once into BOTH
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

    // bbox in panorama pixel space. Returns a confidence-gated CENTROID
    // position built from every lidar point whose projection falls inside
    // the box AND whose range is within kClusterRangeBand of the closest
    // such point (isolates the cone's own near-facing surface from any
    // farther background also captured by the same 2-D box), or nullopt if
    // that cluster has fewer than kMinPointsForDetection points -- a
    // detection can legitimately have too little lidar evidence (a thin/
    // distant part of a cone with few or no returns), and the lidar
    // (10 Hz) isn't synchronized against detections the way the 3 cameras
    // are synchronized against each other (30 Hz) -- see perception.cpp.
    // See lidar_projector.cpp's own comment on this method for why a
    // single nearest-point pick (the previous version) is a confirmed,
    // real systematic bias, not just noise.
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
