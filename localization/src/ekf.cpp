#include "ekf.hpp"

#include <cmath>

namespace
{
double NormalizeAngle(double _angle)
{
    while (_angle > M_PI) _angle -= 2.0 * M_PI;
    while (_angle < -M_PI) _angle += 2.0 * M_PI;
    return _angle;
}

// Euclidean data-association gate, in meters. A tighter Mahalanobis-
// distance gate (using each landmark's own growing/shrinking uncertainty
// rather than a fixed radius) would be the natural refinement, but track
// cones are spaced >=5m apart (Formula Student rules) and this is well
// under half that, so it isn't the limiting factor for a first
// implementation.
constexpr double kLandmarkGateDist = 1.0;

// Sanity cap on total DISCOVERED landmarks (active + retired combined --
// see the class comment's "Submap / local correlation" section). The
// actual track has ~246 cones (confirmed via Gazebo's own scene service
// during earlier testing), so this is generous headroom, not a tight
// limit meant to bind in normal operation -- it exists purely as a
// backstop against unbounded growth in m_retiredLandmarks (a plain
// vector append, cheap per-item but not something that should grow
// forever) if something else keeps making every detection look like a
// brand new landmark instead of a match. This no longer bounds
// correction cost by itself -- kMaxActiveLandmarks below does that now --
// but it's still worth keeping as a backstop for this separate,
// unbounded-growth concern.
constexpr size_t kMaxLandmarks = 600;

// Bounds how many landmarks stay in the joint (correlated) state at once
// -- THIS is what actually bounds every correction's O(n^2) cost now, not
// kMaxLandmarks above. Sized to comfortably exceed how many cones are
// ever visible in one camera frame in practice (observed up to ~15-20 in
// a dense frame) with real margin, while keeping n = kVehicleStateDim +
// 2*kMaxActiveLandmarks small enough that a single correction's matrix
// math is negligible: at 80, n=166, vs. the ~800+ that was measured
// costing 250-300ms per /cone_detections message (far slower than the
// ~33ms between messages) before this fix existed.
constexpr size_t kMaxActiveLandmarks = 80;

// A brand-new landmark's initial world position is computed directly from
// the CURRENT vehicle pose estimate (see AddLandmark) -- if that estimate
// is still highly uncertain (e.g. early in a run, before GNSS/heading have
// meaningfully reduced m_P's startup value of 1e6 -- see the Ekf()
// constructor), the landmark gets permanently seeded at an essentially
// arbitrary position. Its associated covariance (Pll in AddLandmark)
// correctly reflects that uncertainty, but Landmarks() -- and everything
// downstream (/estimated_landmarks, Foxglove) -- only ever reads the POINT
// ESTIMATE, never the covariance, so a landmark added during this window
// renders as a confident, precise marker that can be tens of meters from
// the real cone -- exactly the "cones far in the distance, in addition to
// the correct ones" failure observed in practice. Worse, since the vehicle
// pose is still drifting frame-to-frame during this same window, the SAME
// physical cone keeps failing kLandmarkGateDist's tight 1.0m gate against
// its own just-added (already-wrong) estimate, so this doesn't happen
// once per cone -- it can repeat every frame for every currently-visible
// cone until the filter converges, which is what actually produces a
// landmark count multiples of the real cone count, not just a handful of
// outliers.
//
// Gating new landmark CREATION (not matching -- an existing landmark can
// still be corrected regardless of current vehicle uncertainty) on the
// vehicle's own position variance being below this threshold means a
// landmark is only ever seeded once the filter already has a reasonably
// trustworthy fix on where the car is. 4.0 (m^2, i.e. ~2m stddev) is
// comfortably tighter than the >=5m cone spacing this file already relies
// on elsewhere (kLandmarkGateDist), and loose enough that any real
// correction (GNSS, heading) converges past it well before the car would
// plausibly already be near its first cone.
constexpr double kMaxVehiclePosVarianceForNewLandmark = 4.0; // meters^2
} // namespace

namespace fsd
{

Ekf::Ekf()
{
    m_x = Eigen::VectorXd::Zero(kVehicleStateDim);
    // Large initial uncertainty -- deliberately not zero. The first
    // correction (of any kind) should be trusted almost entirely over
    // this made-up starting guess, and a large P is what makes the Kalman
    // gain do that.
    m_P = Eigen::MatrixXd::Identity(kVehicleStateDim, kVehicleStateDim) * 1.0e6;
}

void Ekf::Predict(double _dt)
{
    if (_dt <= 0.0)
    {
        return;
    }

    // Clamp rather than trust an arbitrarily large gap. gz-transport's
    // pub/sub is lossy under backpressure (drops messages instead of
    // queueing them when a subscriber falls behind), so if this process
    // gets starved of CPU for a while -- competing with something else
    // running on the machine, or just its own cost growing with a large
    // landmark map -- the NEXT message it does process can carry a large
    // gap since the last one actually handled. Applying that gap directly
    // breaks the linearization this filter relies on and can make the
    // pose estimate diverge outright, which then corrupts every
    // subsequent landmark's data-association gating (see
    // CorrectOrAddLandmark) -- a real failure observed in practice, not a
    // theoretical concern. Clamping caps how far a single bad gap can
    // throw the estimate off; it deliberately doesn't try to reconstruct
    // what "really" happened during the missed interval.
    constexpr double kMaxPredictDt = 1.0;  // seconds
    if (_dt > kMaxPredictDt)
    {
        _dt = kMaxPredictDt;
    }

    const double yaw = m_x(2);
    const double vx = m_x(3);
    const double vy = m_x(4);
    const double yawRate = m_x(5);
    const double cosYaw = std::cos(yaw);
    const double sinYaw = std::sin(yaw);

    m_x(0) += (vx * cosYaw - vy * sinYaw) * _dt;
    m_x(1) += (vx * sinYaw + vy * cosYaw) * _dt;
    m_x(2) = NormalizeAngle(yaw + yawRate * _dt);
    // vx, vy, yaw_rate, and every landmark: F is identity for all of them
    // (constant-velocity vehicle assumption; landmarks are static cones).

    const int n = static_cast<int>(m_x.size());

    // F = I + Fd, where Fd is zero except a 3x6 block (rows {x,y,yaw},
    // columns {x,y,yaw,vx,vy,yaw_rate}). Computing F*P*F^T naively as a
    // dense n x n multiply would be O(n^3) in the TOTAL state size
    // (vehicle + 2*landmark count, which can reach several hundred for a
    // full track's worth of cones) -- called from every sensor callback,
    // up to 400Hz, that's the difference between sub-millisecond and
    // multiple seconds per call. Exploiting Fd's sparsity keeps this O(n).
    Eigen::MatrixXd Fd = Eigen::MatrixXd::Zero(3, 6);
    Fd(0, 2) = (-vx * sinYaw - vy * cosYaw) * _dt;
    Fd(0, 3) = cosYaw * _dt;
    Fd(0, 4) = -sinYaw * _dt;
    Fd(1, 2) = (vx * cosYaw - vy * sinYaw) * _dt;
    Fd(1, 3) = sinYaw * _dt;
    Fd(1, 4) = cosYaw * _dt;
    Fd(2, 5) = _dt;

    // A = Fd*P, nonzero only in rows {0,1,2} (all n columns) -- computed
    // directly from P's top 6 rows rather than ever forming the full
    // n x n Fd or F.
    const Eigen::MatrixXd A = Fd * m_P.topRows(6);  // 3 x n

    Eigen::MatrixXd newP = m_P;
    if (n > 3)
    {
        // Off-corner parts: rows {0,1,2} x columns {3..n-1}, and (P being
        // symmetric) the mirrored columns {0,1,2} x rows {3..n-1}.
        newP.block(0, 3, 3, n - 3) += A.block(0, 3, 3, n - 3);
        newP.block(3, 0, n - 3, 3) += A.block(0, 3, 3, n - 3).transpose();
    }

    // Corner (rows/cols {0,1,2}): P + Fd*P + P*Fd^T + Fd*P*Fd^T, all
    // restricted to this 3x3 block. (P*Fd^T)_corner = A_corner^T (since P
    // is symmetric: (P*Fd^T)(i,j) = sum_k P(i,k)Fd(j,k) = sum_k
    // Fd(j,k)P(k,i) = (Fd*P)(j,i) = A(j,i)). Fd*P*Fd^T's corner reuses A's
    // first 6 columns against Fd again.
    const Eigen::Matrix3d Acorner = A.leftCols(3);
    const Eigen::Matrix3d FdPFdTCorner = A.leftCols(6) * Fd.transpose();
    newP.block(0, 0, 3, 3) =
        m_P.block(0, 0, 3, 3) + Acorner + Acorner.transpose() + FdPFdTCorner;

    // Process noise: only vx, vy, yaw_rate drift between updates
    // (position/yaw's uncertainty growth is entirely inherited through F
    // above, and landmarks don't move at all, so they get none).
    constexpr double kVelNoiseDensity = 0.5;      // (m/s)^2 per second
    constexpr double kYawRateNoiseDensity = 0.1;  // (rad/s)^2 per second
    newP(3, 3) += kVelNoiseDensity * _dt;
    newP(4, 4) += kVelNoiseDensity * _dt;
    newP(5, 5) += kYawRateNoiseDensity * _dt;

    // Small, unconditional process noise directly on position/yaw --
    // verified directly as a real, observed failure mode, not a
    // theoretical concern: with a full track's worth of landmarks
    // tracked, /cone_detections can drive hundreds of
    // CorrectMatchedLandmark() calls per second (many cones visible per
    // frame, at camera rate), each one using a deliberately tight
    // kLandmarkStddev (see localization.cpp) and each one shrinking P via
    // the SAME vehicle-pose columns of H every time. Position/yaw's ONLY
    // other source of uncertainty growth is inherited through F above
    // (itself driven by velocity uncertainty, which is under the exact
    // same pressure from ground-speed/IMU corrections) -- with nothing
    // added here directly, that many tightly-trusted corrections per
    // second can crush P's vehicle-pose block toward numerical zero
    // faster than F*P*F^T can rebuild it. Once that happens, the Kalman
    // gain for EVERY future correction -- including GNSS -- collapses
    // toward zero too (gain is proportional to P), so the filter stops
    // responding to new measurements at all and the estimate freezes in
    // place. Confirmed directly: /estimated_pose moved by ~3mm over
    // several seconds of active driving.
    //
    // This is an UNCONDITIONAL addition (matching kVelNoiseDensity/
    // kYawRateNoiseDensity above), not a clamp -- clamping a diagonal
    // entry directly (newP(0,0) = floor) would leave its off-diagonal
    // correlations with every other state dimension (velocity, yaw, every
    // tracked landmark) inconsistent with the new diagonal value, which
    // can break P's positive-semi-definiteness and cause worse downstream
    // numerical failures (e.g. in a later correction's S.inverse()) than
    // the freeze this is fixing. Adding a small PSD (diagonal,
    // non-negative) matrix to a PSD matrix is always safe.
    //
    // 0.01 (m^2/s, i.e. sqrt(0.01)=0.1m of stddev growth per second of
    // Predict() calls, however finely divided) is sized to roughly match
    // kLandmarkStddev's own scale (0.1m) -- enough to give even the most
    // aggressive plausible correction rate something to work against, not
    // so much that it becomes the dominant source of position uncertainty
    // under normal (non-pathological) operation.
    constexpr double kPosNoiseDensity = 0.01;   // meters^2 per second
    constexpr double kYawNoiseDensity = 0.0001; // radians^2 per second
    newP(0, 0) += kPosNoiseDensity * _dt;
    newP(1, 1) += kPosNoiseDensity * _dt;
    newP(2, 2) += kYawNoiseDensity * _dt;

    m_P = std::move(newP);
}

void Ekf::CorrectBodyVelocity(double _vx, double _vy, double _stddevVx, double _stddevVy)
{
    const int n = static_cast<int>(m_x.size());
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, n);
    H(0, 3) = 1.0;
    H(1, 4) = 1.0;

    const Eigen::Vector2d z(_vx, _vy);
    const Eigen::Vector2d h(m_x(3), m_x(4));
    const Eigen::Vector2d y = z - h;

    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    R(0, 0) = _stddevVx * _stddevVx;
    R(1, 1) = _stddevVy * _stddevVy;

    const Eigen::Matrix2d S = H * m_P * H.transpose() + R;
    const Eigen::MatrixXd K = m_P * H.transpose() * S.inverse();

    m_x += K * y;
    m_x(2) = NormalizeAngle(m_x(2));
    // P_new = P - K*(H*P), NOT (I - K*H)*P -- the latter is mathematically
    // equivalent but forms a dense n x n (I-K*H) matrix and multiplies it
    // by P, an O(n^3) operation Eigen has no way to avoid on its own. H
    // only has 1-2 nonzero rows, so H*P is O(n) rows worth of work (O(n)
    // per row = O(n) here since k<=2), and K*(H*P) is (n x k)*(k x n) =
    // O(k*n^2) -- with k a small constant, that's O(n^2) overall, not
    // O(n^3). At a few hundred state dimensions (a landmark-heavy SLAM
    // map), that difference is the entire performance budget.
    m_P = m_P - K * (H * m_P);
}

void Ekf::CorrectYawRate(double _yawRate, double _stddev)
{
    const int n = static_cast<int>(m_x.size());
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, n);
    H(0, 5) = 1.0;

    const double y = _yawRate - m_x(5);
    const double S = (H * m_P * H.transpose())(0, 0) + _stddev * _stddev;
    const Eigen::MatrixXd K = m_P * H.transpose() / S;

    m_x += K * y;
    m_x(2) = NormalizeAngle(m_x(2));
    // P_new = P - K*(H*P), NOT (I - K*H)*P -- the latter is mathematically
    // equivalent but forms a dense n x n (I-K*H) matrix and multiplies it
    // by P, an O(n^3) operation Eigen has no way to avoid on its own. H
    // only has 1-2 nonzero rows, so H*P is O(n) rows worth of work (O(n)
    // per row = O(n) here since k<=2), and K*(H*P) is (n x k)*(k x n) =
    // O(k*n^2) -- with k a small constant, that's O(n^2) overall, not
    // O(n^3). At a few hundred state dimensions (a landmark-heavy SLAM
    // map), that difference is the entire performance budget.
    m_P = m_P - K * (H * m_P);
}

void Ekf::CorrectGnssPosition(double _measuredEast, double _measuredNorth,
                               double _antennaOffsetX, double _antennaOffsetY,
                               double _stddev)
{
    const int n = static_cast<int>(m_x.size());
    const double yaw = m_x(2);
    const double cosYaw = std::cos(yaw);
    const double sinYaw = std::sin(yaw);

    const Eigen::Vector2d h(
        m_x(0) + _antennaOffsetX * cosYaw - _antennaOffsetY * sinYaw,
        m_x(1) + _antennaOffsetX * sinYaw + _antennaOffsetY * cosYaw);
    const Eigen::Vector2d z(_measuredEast, _measuredNorth);
    const Eigen::Vector2d y = z - h;

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, n);
    H(0, 0) = 1.0;
    H(0, 2) = -_antennaOffsetX * sinYaw - _antennaOffsetY * cosYaw;
    H(1, 1) = 1.0;
    H(1, 2) = _antennaOffsetX * cosYaw - _antennaOffsetY * sinYaw;

    const Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * (_stddev * _stddev);

    const Eigen::Matrix2d S = H * m_P * H.transpose() + R;
    const Eigen::MatrixXd K = m_P * H.transpose() * S.inverse();

    m_x += K * y;
    m_x(2) = NormalizeAngle(m_x(2));
    // P_new = P - K*(H*P), NOT (I - K*H)*P -- the latter is mathematically
    // equivalent but forms a dense n x n (I-K*H) matrix and multiplies it
    // by P, an O(n^3) operation Eigen has no way to avoid on its own. H
    // only has 1-2 nonzero rows, so H*P is O(n) rows worth of work (O(n)
    // per row = O(n) here since k<=2), and K*(H*P) is (n x k)*(k x n) =
    // O(k*n^2) -- with k a small constant, that's O(n^2) overall, not
    // O(n^3). At a few hundred state dimensions (a landmark-heavy SLAM
    // map), that difference is the entire performance budget.
    m_P = m_P - K * (H * m_P);
}

void Ekf::CorrectHeading(double _measuredYaw, double _stddev)
{
    const int n = static_cast<int>(m_x.size());
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, n);
    H(0, 2) = 1.0;

    const double y = NormalizeAngle(_measuredYaw - m_x(2));
    const double S = (H * m_P * H.transpose())(0, 0) + _stddev * _stddev;
    const Eigen::MatrixXd K = m_P * H.transpose() / S;

    m_x += K * y;
    m_x(2) = NormalizeAngle(m_x(2));
    // P_new = P - K*(H*P), NOT (I - K*H)*P -- the latter is mathematically
    // equivalent but forms a dense n x n (I-K*H) matrix and multiplies it
    // by P, an O(n^3) operation Eigen has no way to avoid on its own. H
    // only has 1-2 nonzero rows, so H*P is O(n) rows worth of work (O(n)
    // per row = O(n) here since k<=2), and K*(H*P) is (n x k)*(k x n) =
    // O(k*n^2) -- with k a small constant, that's O(n^2) overall, not
    // O(n^3). At a few hundred state dimensions (a landmark-heavy SLAM
    // map), that difference is the entire performance budget.
    m_P = m_P - K * (H * m_P);
}

void Ekf::CorrectOrAddLandmark(double _measuredBodyX, double _measuredBodyY,
                                ConeColor _color, double _stddev)
{
    const double yaw = m_x(2);
    const double cosYaw = std::cos(yaw);
    const double sinYaw = std::sin(yaw);
    const double worldX = m_x(0) + _measuredBodyX * cosYaw - _measuredBodyY * sinYaw;
    const double worldY = m_x(1) + _measuredBodyX * sinYaw + _measuredBodyY * cosYaw;

    // 1. Try to match against an ACTIVE (in the joint state) landmark.
    int bestIndex = -1;
    double bestDistSq = kLandmarkGateDist * kLandmarkGateDist;
    for (size_t i = 0; i < m_landmarkColors.size(); ++i)
    {
        if (m_landmarkColors[i] != _color)
        {
            continue;
        }
        const int li = kVehicleStateDim + 2 * static_cast<int>(i);
        const double dx = m_x(li) - worldX;
        const double dy = m_x(li + 1) - worldY;
        const double distSq = dx * dx + dy * dy;
        if (distSq < bestDistSq)
        {
            bestDistSq = distSq;
            bestIndex = static_cast<int>(i);
        }
    }

    if (bestIndex >= 0)
    {
        CorrectMatchedLandmark(bestIndex, _measuredBodyX, _measuredBodyY, _stddev);
        EvictStaleIfOverCapacity();
        return;
    }

    // 2. No active match -- check RETIRED landmarks (same gate). A match
    // here means the car is revisiting a landmark it previously evicted
    // from the active state (see the class comment). Reactivate it by
    // dropping it from m_retiredLandmarks and falling through to the
    // "add" path below, using THIS detection -- simpler and safer than
    // inventing a second, seeded-from-the-retired-estimate covariance
    // construction; AddLandmark's existing Jacobian-based augmentation
    // already correctly handles "new landmark from a fresh detection",
    // and reusing it here means there's only one place that math needs
    // to be right.
    for (size_t i = 0; i < m_retiredLandmarks.size(); ++i)
    {
        if (m_retiredLandmarks[i].color != _color)
        {
            continue;
        }
        const double dx = m_retiredLandmarks[i].x - worldX;
        const double dy = m_retiredLandmarks[i].y - worldY;
        if (dx * dx + dy * dy < kLandmarkGateDist * kLandmarkGateDist)
        {
            m_retiredLandmarks.erase(m_retiredLandmarks.begin() + static_cast<long>(i));
            break;
        }
    }

    // 3. Genuinely new (or just-reactivated) landmark.
    if (m_P(0, 0) > kMaxVehiclePosVarianceForNewLandmark ||
        m_P(1, 1) > kMaxVehiclePosVarianceForNewLandmark)
    {
        // Vehicle's own position estimate isn't converged enough yet to
        // trust seeding a brand-new landmark from it -- see
        // kMaxVehiclePosVarianceForNewLandmark's comment. Silently drop:
        // this detection simply gets no landmark this cycle, same as if
        // it had no lidar match at all; it'll be tried again next cycle
        // once (or if) the filter has converged further.
        return;
    }
    if (m_landmarkColors.size() + m_retiredLandmarks.size() >= kMaxLandmarks)
    {
        // At the overall discovered-landmark cap -- silently drop rather
        // than keep growing. Existing landmarks (active or retired, any
        // color) can still be matched/reactivated normally; this only
        // stops brand new ones from being created.
        return;
    }
    AddLandmark(_measuredBodyX, _measuredBodyY, _color, _stddev);
    EvictStaleIfOverCapacity();
}

void Ekf::CorrectMatchedLandmark(int _landmarkIndex, double _measuredBodyX,
                                  double _measuredBodyY, double _stddev)
{
    const int n = static_cast<int>(m_x.size());
    const int li = kVehicleStateDim + 2 * _landmarkIndex;

    const double yaw = m_x(2);
    const double cosYaw = std::cos(yaw);
    const double sinYaw = std::sin(yaw);

    // Landmark position is read from the STATE now (a joint estimate,
    // correlated with the vehicle and every other landmark), not a known
    // constant -- the key structural difference from the pre-SLAM
    // CorrectLandmark this replaces.
    const double relX = m_x(li) - m_x(0);
    const double relY = m_x(li + 1) - m_x(1);
    const double bodyXPred = relX * cosYaw + relY * sinYaw;
    const double bodyYPred = -relX * sinYaw + relY * cosYaw;

    const Eigen::Vector2d z(_measuredBodyX, _measuredBodyY);
    const Eigen::Vector2d h(bodyXPred, bodyYPred);
    const Eigen::Vector2d y = z - h;

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, n);
    // w.r.t. vehicle x, y, yaw (same form as the pre-SLAM landmark
    // correction, since h()'s dependence on the vehicle sub-state is
    // unchanged).
    H(0, 0) = -cosYaw;
    H(0, 1) = -sinYaw;
    H(0, 2) = bodyYPred;
    H(1, 0) = sinYaw;
    H(1, 1) = -cosYaw;
    H(1, 2) = -bodyXPred;
    // w.r.t. the matched landmark's own (x, y) state -- NEW vs. the
    // pre-SLAM version, since the landmark is now part of the state being
    // differentiated against, not a constant.
    H(0, li) = cosYaw;
    H(0, li + 1) = sinYaw;
    H(1, li) = -sinYaw;
    H(1, li + 1) = cosYaw;

    const Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * (_stddev * _stddev);

    const Eigen::Matrix2d S = H * m_P * H.transpose() + R;
    const Eigen::MatrixXd K = m_P * H.transpose() * S.inverse();

    m_x += K * y;
    m_x(2) = NormalizeAngle(m_x(2));
    // P_new = P - K*(H*P), NOT (I - K*H)*P -- the latter is mathematically
    // equivalent but forms a dense n x n (I-K*H) matrix and multiplies it
    // by P, an O(n^3) operation Eigen has no way to avoid on its own. H
    // only has 1-2 nonzero rows, so H*P is O(n) rows worth of work (O(n)
    // per row = O(n) here since k<=2), and K*(H*P) is (n x k)*(k x n) =
    // O(k*n^2) -- with k a small constant, that's O(n^2) overall, not
    // O(n^3). At a few hundred state dimensions (a landmark-heavy SLAM
    // map), that difference is the entire performance budget.
    m_P = m_P - K * (H * m_P);

    m_landmarkLastSeen[static_cast<size_t>(_landmarkIndex)] = ++m_tick;
}

void Ekf::AddLandmark(double _measuredBodyX, double _measuredBodyY,
                       ConeColor _color, double _stddev)
{
    const int n = static_cast<int>(m_x.size());
    const double yaw = m_x(2);
    const double cosYaw = std::cos(yaw);
    const double sinYaw = std::sin(yaw);

    const double lmX = m_x(0) + _measuredBodyX * cosYaw - _measuredBodyY * sinYaw;
    const double lmY = m_x(1) + _measuredBodyX * sinYaw + _measuredBodyY * cosYaw;

    // State-augmentation Jacobians: the new landmark's position is a
    // function g(vehicle_state, measurement) of the CURRENT state and the
    // raw detection, so its initial uncertainty -- and crucially its
    // initial CORRELATION with the rest of the state -- has to be
    // propagated through g, not just seeded with a guessed diagonal
    // covariance. That correlation is exactly what lets a later
    // re-observation of this landmark correct the vehicle's pose too (see
    // the class comment in ekf.hpp).
    Eigen::MatrixXd Gx = Eigen::MatrixXd::Zero(2, 3);  // w.r.t. vehicle x, y, yaw
    Gx(0, 0) = 1.0;
    Gx(0, 2) = -_measuredBodyX * sinYaw - _measuredBodyY * cosYaw;
    Gx(1, 1) = 1.0;
    Gx(1, 2) = _measuredBodyX * cosYaw - _measuredBodyY * sinYaw;

    Eigen::Matrix2d Gz;  // w.r.t. the raw measurement (bodyX, bodyY)
    Gz(0, 0) = cosYaw;
    Gz(0, 1) = -sinYaw;
    Gz(1, 0) = sinYaw;
    Gz(1, 1) = cosYaw;

    const Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * (_stddev * _stddev);

    // Cross-covariance with the FULL existing state (2 x n) -- only
    // depends on the vehicle sub-block (Gx is zero elsewhere), so this is
    // O(n) via P's top 3 rows rather than a full n x n multiply.
    const Eigen::MatrixXd Pcross = Gx * m_P.topRows(3);  // 2 x n
    const Eigen::Matrix2d Pll =
        Gx * m_P.topLeftCorner(3, 3) * Gx.transpose() + Gz * R * Gz.transpose();

    m_x.conservativeResize(n + 2);
    m_x(n) = lmX;
    m_x(n + 1) = lmY;

    m_P.conservativeResize(n + 2, n + 2);
    m_P.block(n, 0, 2, n) = Pcross;
    m_P.block(0, n, n, 2) = Pcross.transpose();
    m_P.block(n, n, 2, 2) = Pll;

    m_landmarkColors.push_back(_color);
    m_landmarkLastSeen.push_back(++m_tick);
}

std::vector<Ekf::LandmarkEstimate> Ekf::Landmarks() const
{
    std::vector<LandmarkEstimate> result;
    result.reserve(m_landmarkColors.size() + m_retiredLandmarks.size());
    for (size_t i = 0; i < m_landmarkColors.size(); ++i)
    {
        const int li = kVehicleStateDim + 2 * static_cast<int>(i);
        result.push_back(LandmarkEstimate{m_x(li), m_x(li + 1), m_landmarkColors[i]});
    }
    // Retired landmarks are no longer part of the joint state (see the
    // class comment), but /estimated_landmarks should still show the full
    // discovered map, not just whichever subset happens to still be
    // actively correlated with the vehicle.
    for (const auto &retired : m_retiredLandmarks)
    {
        result.push_back(retired);
    }
    return result;
}

void Ekf::RemoveActiveLandmark(size_t _index)
{
    const int li = kVehicleStateDim + 2 * static_cast<int>(_index);
    const int n = static_cast<int>(m_x.size());
    const int tail = n - li - 2; // size of the surviving "after" region

    // Rebuilt into fresh, smaller objects rather than shifted in place --
    // see this method's declaration in ekf.hpp for why an in-place block
    // shift is riskier than it looks (Eigen aliasing).
    Eigen::VectorXd newX(n - 2);
    newX.segment(0, li) = m_x.segment(0, li);
    if (tail > 0)
    {
        newX.segment(li, tail) = m_x.segment(li + 2, tail);
    }

    Eigen::MatrixXd newP(n - 2, n - 2);
    newP.block(0, 0, li, li) = m_P.block(0, 0, li, li);
    if (tail > 0)
    {
        newP.block(0, li, li, tail) = m_P.block(0, li + 2, li, tail);
        newP.block(li, 0, tail, li) = m_P.block(li + 2, 0, tail, li);
        newP.block(li, li, tail, tail) = m_P.block(li + 2, li + 2, tail, tail);
    }

    m_x = std::move(newX);
    m_P = std::move(newP);
    m_landmarkColors.erase(m_landmarkColors.begin() + static_cast<long>(_index));
    m_landmarkLastSeen.erase(m_landmarkLastSeen.begin() + static_cast<long>(_index));
}

void Ekf::EvictStaleIfOverCapacity()
{
    while (m_landmarkColors.size() > kMaxActiveLandmarks)
    {
        size_t staleIndex = 0;
        uint64_t oldest = m_landmarkLastSeen[0];
        for (size_t i = 1; i < m_landmarkLastSeen.size(); ++i)
        {
            if (m_landmarkLastSeen[i] < oldest)
            {
                oldest = m_landmarkLastSeen[i];
                staleIndex = i;
            }
        }

        const int li = kVehicleStateDim + 2 * static_cast<int>(staleIndex);
        m_retiredLandmarks.push_back(
            LandmarkEstimate{m_x(li), m_x(li + 1), m_landmarkColors[staleIndex]});
        RemoveActiveLandmark(staleIndex);
    }
}

} // namespace fsd
