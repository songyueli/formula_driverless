#include "lidar_projector.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace
{
// Known, fixed extrinsics from simulation/models/fsd_car/model.sdf -- see
// lidar_projector.hpp's class comment.
constexpr double kLidarX = 0.3, kLidarY = 0.0, kLidarZ = 0.35;
constexpr double kCamX = 0.9, kCamY = 0.0, kCamZ = 1.0;
constexpr double kCamPitchRad = 0.3;
} // namespace

LidarProjector::LidarProjector(int panoWidth, int panoHeight, double fPano)
    : m_panoWidth(panoWidth), m_panoHeight(panoHeight), m_fPano(fPano)
{
}

void LidarProjector::SetPointCloud(const gz::msgs::PointCloudPacked &msg)
{
    m_points.clear();

    // Looked up by name rather than hardcoded, even though the exact layout
    // (x=0, y=4, z=8, all FLOAT32) was confirmed empirically via `gz topic
    // -e` -- costs nothing here (a handful of fields, once per message) and
    // isn't silently wrong if a future Gazebo version ever reorders them.
    int offsetX = -1, offsetY = -1, offsetZ = -1;
    for (const auto &f : msg.field())
    {
        if (f.name() == "x") offsetX = f.offset();
        else if (f.name() == "y") offsetY = f.offset();
        else if (f.name() == "z") offsetZ = f.offset();
    }
    if (offsetX < 0 || offsetY < 0 || offsetZ < 0)
    {
        return; // unexpected layout -- produce no points rather than misread memory
    }

    const std::string &data = msg.data();
    const uint32_t stride = msg.point_step();
    const uint64_t numPoints = stride > 0 ? data.size() / stride : 0;

    const double cosPitch = std::cos(kCamPitchRad);
    const double sinPitch = std::sin(kCamPitchRad);

    m_points.reserve(numPoints);
    for (uint64_t i = 0; i < numPoints; ++i)
    {
        const char *base = data.data() + i * stride;
        float lx, ly, lz;
        std::memcpy(&lx, base + offsetX, sizeof(float));
        std::memcpy(&ly, base + offsetY, sizeof(float));
        std::memcpy(&lz, base + offsetZ, sizeof(float));

        if (!std::isfinite(lx) || !std::isfinite(ly) || !std::isfinite(lz))
        {
            continue; // no return at this ray
        }

        // Lidar -> chassis frame: lidar has zero rotation relative to the
        // chassis (model.sdf), so this is a pure translation by its mount
        // offset.
        const double cx = lx + kLidarX;
        const double cy = ly + kLidarY;
        const double cz = lz + kLidarZ;

        // Chassis -> front camera's own (pitched) reference frame -- the
        // SAME frame the panorama's own pixel math (camera_stitcher.cpp)
        // uses, and the SAME pitch-rotation formula derived (and verified,
        // via the horizon-straightness proof) for that code, reused here
        // rather than re-derived, to avoid a fresh sign error on the same
        // math: position relative to the camera's own mount point, then
        // rotate by -pitch.
        const double relX = cx - kCamX;
        const double relY = cy - kCamY;
        const double relZ = cz - kCamZ;
        const double fwd = relX * cosPitch - relZ * sinPitch;
        const double left = relY;
        const double up = relX * sinPitch + relZ * cosPitch;

        if (fwd <= 1e-3)
        {
            continue; // behind the camera-reference plane -- can't be in any detection
        }

        const double theta = std::atan2(left, fwd);
        const double horizMag = std::sqrt(fwd * fwd + left * left);
        const double h = up / horizMag;

        const float u = static_cast<float>(m_panoWidth / 2.0 - theta * m_fPano);
        const float v = static_cast<float>(m_panoHeight / 2.0 - h * m_fPano);

        if (u < 0 || u >= m_panoWidth || v < 0 || v >= m_panoHeight)
        {
            continue; // projects outside the panorama entirely
        }

        const float range = static_cast<float>(std::sqrt(cx * cx + cy * cy + cz * cz));
        m_points.push_back(ProjectedPoint{
            u, v,
            ConePosition{static_cast<float>(cx), static_cast<float>(cy), static_cast<float>(cz), range}});
    }
}

std::optional<LidarProjector::ConePosition> LidarProjector::Localize(
    float x1, float y1, float x2, float y2) const
{
    std::optional<ConePosition> best;
    for (const auto &p : m_points)
    {
        if (p.u < x1 || p.u > x2 || p.v < y1 || p.v > y2)
        {
            continue;
        }
        if (!best || p.pos.range < best->range)
        {
            best = p.pos;
        }
    }
    return best;
}
