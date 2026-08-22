#include "lidar_projector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace
{
// Known, fixed extrinsics from simulation/models/fsd_car/model.sdf -- see
// lidar_projector.hpp's class comment. kLidarZ raised from the original
// 0.35 (settled on 0.6 after 0.9 measured better accuracy but sat too high
// in practice) -- MUST stay in sync with the lidar sensor's <pose> in
// model.sdf (see that file's comment for why the mount height changed).
constexpr double kLidarX = 0.3, kLidarY = 0.0, kLidarZ = 0.6;
constexpr double kCamX = 0.9, kCamY = 0.0, kCamZ = 1.0;
constexpr double kCamPitchRad = 0.3;

// Localize() picks whichever cached point has the smallest RANGE among
// everything landing inside a detection's 2-D box -- it has no notion of
// whether that point is actually a good depth match for what's IN the box,
// just that its projection happens to overlap it. Near-ground and
// self-referential returns close to the car (verified directly against a
// real lidar scan: ~100+ such points, ALL measuring 2.94-3.0m range,
// landing inside valid panorama pixel space near its lower edge, where
// the ground close to the car is visible) will always win that comparison
// against a real cone's actual, farther surface if their projections
// happen to overlap the same box -- silently mislocalizing the detection
// to wherever the car currently is instead of the cone. That produces
// landmarks that trace the car's own path rather than real cone
// positions, confirmed directly against a live capture.
//
// 4.0m gives that measured 2.94-3.0m cluster comfortable margin, while
// staying well under every real cone detection range actually observed
// live (6m+ once the panorama-edge duplicate case is excluded separately
// -- see perception.cpp's kPanoEdgeMarginPx). An earlier version of this
// fix used 1.5m -- comfortably WRONG, since it sat entirely below the
// measured 2.94-3.0m artifact range and excluded nothing; left here as a
// reminder to verify a threshold against the actual measured data driving
// it, not just what "sounds" plausible.
constexpr float kMinValidRange = 4.0f; // meters

// Localize() has no notion of whether the closest-range point inside a
// box is actually a good depth match for what's in it (see this file's
// class comment above) -- that's a real problem in BOTH directions, not
// just the near-ground one kMinValidRange guards against. At long range a
// detection's 2-D box is only a few pixels wide, so the angular resolution
// covering it is coarse enough that unrelated background/terrain points
// far behind the actual cone can easily land inside the same tiny box and
// (being a valid, finite return) win the closest-range comparison just as
// easily as a near-ground artifact would. Confirmed directly: landmarks
// were appearing ~50m from the vehicle immediately at startup, before the
// car had moved at all, tracing back to exactly this -- a detection whose
// box happened to also contain a stray far-range point. A cone that far
// away is also inherently a bad landmark candidate on its own merits even
// when the match IS a genuine cone surface: bearing/range noise at that
// distance is large relative to close-range detections, so seeding a
// landmark from it starts that landmark's estimate from a much worse prior
// than usual. 20m is comfortably beyond every real, reliable detection
// actually observed live (2-6m for the near/working cases -- see
// kMinValidRange's comment), while safely excluding the ~50m artifact.
constexpr float kMaxValidRange = 20.0f; // meters

// A single nearest-range point (the previous version of Localize()) is a
// confirmed, real SYSTEMATIC bias, not just noise -- confirmed directly by
// comparing raw /cone_detections against ground truth from a precisely
// known, perfectly stationary car pose (via gz-sim's own set_pose service,
// eliminating pose-estimation error from the measurement): frame-to-frame
// JITTER was already small (~0.015m mean stddev across 30 consecutive
// frames of the same physical cones), but every single cone showed a
// CONSISTENT offset in the same direction (0.02-0.105m, mean 0.067m) --
// not noise, a bias. Root cause: lidar can only ever return points off a
// cone's near-facing surface (opaque object, no way to see its far side or
// its own central axis), so "closest point wins" always reads at or in
// front of the true axis by some fraction of the cone's own radius
// (0.115m at the base -- see simulation/models/cone_blue/model.sdf --
// matching the observed bias magnitude closely). A centroid over a
// CLUSTER of near-surface points doesn't eliminate this (every point in
// the cluster has the same near-surface-only limitation), but it does two
// things a single point can't: averages out the WITHIN-cluster spread
// (reducing the frame-to-frame jitter component further), and gives
// Localize() actual evidence of confidence (point COUNT) to gate on --
// see kMinPointsForDetection.
constexpr int kMinPointsForDetection = 2;

// Points within this range of the closest in-box point are treated as the
// same cluster (the cone's own near surface) -- points farther than this
// are assumed to be a different object (background/occlusion behind the
// cone, still inside the same 2-D box) and excluded from the centroid
// rather than dragging it toward some other surface entirely. Sized
// comfortably larger than a cone's own front-to-back depth extent (at most
// ~2*0.115m = 0.23m for the base radius) to tolerate ordinary lidar range
// noise (0.008m stddev per model.sdf) without fragmenting genuine
// same-cone points into separate clusters, while still meaningfully
// excluding anything that's a genuinely different, farther object.
constexpr float kClusterRangeBand = 0.4f; // meters

// Even a clean multi-point centroid (see kMinPointsForDetection above) is
// still built ENTIRELY from near-facing-surface returns -- lidar has no
// way to see a solid cone's far side or its own central axis, so the
// centroid necessarily sits somewhere between the true axis and the near
// surface, not AT the axis. Confirmed directly, not assumed: comparing raw
// /cone_detections against ground truth from a perfectly stationary,
// precisely known car pose (gz-sim's own set_pose service) measured a
// consistent ~0.058m mean bias toward the car across 5 cones, ALWAYS in
// the same direction (toward the sensor) -- matching, almost exactly, HALF
// of the cone's own 0.115m base radius (simulation/models/cone_blue/
// model.sdf) -- physically exactly what's expected from a centroid
// averaged over points spanning the cone's tapered surface from base
// (full 0.115m offset) up toward its tip (~0m offset), if hit heights are
// roughly uniformly distributed. Pushing the centroid's HORIZONTAL
// position outward along its own ray (away from the car) by this amount
// recovers an estimate of the true central axis instead of the near
// surface. z is left uncorrected -- the measured bias was in horizontal
// (range) position only; this pipeline's downstream consumers (EKF
// landmarks, ground-truth comparison) only ever use (x, y) too.
constexpr float kConeSurfaceToAxisCorrection = 0.058f; // meters
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
    // Pass 1: every in-box, in-range point, and the closest range among
    // them -- needed before clustering can decide what "close enough to
    // the nearest one" even means.
    std::vector<const ConePosition *> inBox;
    float nearestRange = kMaxValidRange;
    for (const auto &p : m_points)
    {
        if (p.u < x1 || p.u > x2 || p.v < y1 || p.v > y2)
        {
            continue;
        }
        if (p.pos.range < kMinValidRange || p.pos.range > kMaxValidRange)
        {
            continue;
        }
        inBox.push_back(&p.pos);
        nearestRange = std::min(nearestRange, p.pos.range);
    }

    // Pass 2: keep only the near cluster (within kClusterRangeBand of the
    // closest point) -- see kClusterRangeBand's comment for why this
    // excludes farther background/occlusion sharing the same 2-D box
    // rather than letting it drag the centroid off the cone entirely.
    float sumX = 0.0f, sumY = 0.0f, sumZ = 0.0f;
    int count = 0;
    for (const ConePosition *p : inBox)
    {
        if (p->range > nearestRange + kClusterRangeBand)
        {
            continue;
        }
        sumX += p->x;
        sumY += p->y;
        sumZ += p->z;
        ++count;
    }

    // Not enough independent lidar evidence to be confident this is a real,
    // well-localized cone surface rather than a single stray/noisy return
    // -- see kMinPointsForDetection's comment. Matches this class's own
    // existing philosophy (a detection can legitimately have NO match);
    // this just raises the bar from "at least one point" to "enough points
    // to trust a position from".
    if (count < kMinPointsForDetection)
    {
        return std::nullopt;
    }

    ConePosition centroid{sumX / count, sumY / count, sumZ / count, 0.0f};

    // Push outward along the horizontal ray from the car's own origin --
    // see kConeSurfaceToAxisCorrection's comment for why. horizRange is
    // the pre-correction horizontal distance; guarded against ~0 (a cone
    // directly on top of the car's origin isn't a real case this pipeline
    // ever sees -- kMinValidRange already excludes anything under 4m --
    // but division by a near-zero horizRange would be undefined otherwise).
    const float horizRange = std::sqrt(centroid.x * centroid.x + centroid.y * centroid.y);
    if (horizRange > 1e-3f)
    {
        const float scale = (horizRange + kConeSurfaceToAxisCorrection) / horizRange;
        centroid.x *= scale;
        centroid.y *= scale;
    }

    centroid.range = std::sqrt(centroid.x * centroid.x + centroid.y * centroid.y
                                + centroid.z * centroid.z);
    return centroid;
}
