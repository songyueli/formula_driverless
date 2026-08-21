#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Dense>

#include <common/types.hpp>

// EKF-SLAM: jointly estimates the vehicle's own pose/velocity AND the
// world-frame positions of every cone landmark it has observed, as one
// correlated state vector -- not vehicle-only localization against a
// separately-known map (that was the pre-SLAM version of this class; see
// git history).
//
// State = [vehicle (6): x, y, yaw, vx_body, vy_body, yaw_rate,
//          landmark_0 (2): x, y,
//          landmark_1 (2): x, y,
//          ...]
//
// Vehicle sub-state semantics are unchanged from before: position/yaw in
// the world frame, velocity in the vehicle's own body frame. Landmarks are
// always world-frame and, being cones, assumed static -- Predict() only
// evolves the vehicle sub-state; landmark rows/columns of the covariance
// are otherwise untouched by prediction (their process noise is exactly
// zero, since they don't move).
//
// The state grows dynamically as new landmarks are discovered
// (CorrectOrAddLandmark), which is why this uses Eigen::VectorXd/MatrixXd
// throughout rather than fixed-size types.
//
// What makes this SLAM rather than "localization with a map": landmark
// positions are STATE VARIABLES with their own uncertainty, correlated
// with the vehicle's pose and with each other through the full covariance
// matrix (not block-diagonal). Re-observing a landmark corrects both its
// own estimate AND, through that correlation, the vehicle's pose -- most
// visibly on loop closure (driving back past cones seen early in a lap),
// where accumulated drift can partially collapse in one update instead of
// only ever being corrected by GNSS.
//
// Submap / local correlation (bounds the state, and therefore every
// correction's cost): a full joint covariance means EVERY correction --
// even one from a single freshly-observed landmark -- touches the ENTIRE
// P matrix, not just that landmark's own rows/columns, because H*P is
// dense (P correlates every landmark with every other one and with the
// vehicle). At n state dimensions that's O(n^2) per correction; verified
// directly in practice, this made a full track's worth of landmarks
// (n ~ 800+) cost 250-300ms PER /cone_detections message -- far slower
// than the ~33ms between messages, causing localization to fall
// catastrophically behind real time. Landmarks NOT currently visible
// don't need to stay in that expensive joint state at all -- they only
// need to be revisited (reactivated) if the car drives back near them.
// So m_landmarkColors/m_x/m_P only ever hold up to kMaxActiveLandmarks
// landmarks; anything evicted (see EvictStaleIfOverCapacity) moves to
// m_retiredLandmarks -- its last known estimate, no longer correlated
// with the vehicle or with other landmarks, no longer touched by any
// correction math, but still reported by Landmarks() (so
// /estimated_landmarks keeps showing the full discovered map, active or
// not) and still eligible to be matched again on CorrectOrAddLandmark
// (reactivating it back into the active state) if the car revisits it.
namespace fsd
{

class Ekf
{
public:
    static constexpr int kVehicleStateDim = 6;

    struct LandmarkEstimate
    {
        double x;
        double y;
        ConeColor color;
    };

    Ekf();

    // Advances the state by _dt seconds using the constant body-velocity /
    // constant yaw-rate motion model for the vehicle sub-state; landmarks
    // are untouched (they don't move). Safe to call with _dt <= 0 (no-op).
    void Predict(double _dt);

    // Ground-speed sensor: measures vx_body/vy_body directly.
    void CorrectBodyVelocity(double _vx, double _vy, double _stddevVx, double _stddevVy);

    // IMU gyro Z axis: measures yaw_rate directly.
    void CorrectYawRate(double _yawRate, double _stddev);

    // A single GNSS antenna's position, already converted to local ENU
    // meters (see geodetic.hpp). _antennaOffset{X,Y} is that antenna's
    // known mount position in the vehicle's body frame.
    void CorrectGnssPosition(double _measuredEast, double _measuredNorth,
                              double _antennaOffsetX, double _antennaOffsetY,
                              double _stddev);

    // Dual-antenna GNSS-compass heading: an absolute yaw measurement.
    void CorrectHeading(double _measuredYaw, double _stddev);

    // A cone detection in the vehicle's own body frame, from
    // /cone_detections. Internally: try to match it (Mahalanobis-gated,
    // see kLandmarkGateChiSq in ekf.cpp) against an existing SAME-COLOR
    // ACTIVE landmark; if found, apply a joint vehicle+landmark correction
    // (both get updated, see the class comment above). If not, try a
    // (necessarily approximate, see the retired-matching code) gate
    // against RETIRED landmarks -- a match there means the car is
    // revisiting a landmark it previously evicted from the active state,
    // so it's reactivated (added back into the joint state fresh, from
    // this detection). Only if neither matches is it truly new -- added
    // via a proper covariance-augmentation Jacobian (not just a bare
    // append with a guessed uncertainty -- see AddLandmark for why that
    // matters). Either kind of addition may in turn evict the now-
    // stalest active landmark if this pushes the active count over
    // kMaxActiveLandmarks -- see EvictStaleIfOverCapacity.
    void CorrectOrAddLandmark(double _measuredBodyX, double _measuredBodyY,
                               ConeColor _color, double _stddev);

    double X() const { return m_x(0); }
    double Y() const { return m_x(1); }
    double Yaw() const { return m_x(2); }

    // Every discovered landmark, active or retired -- see the class
    // comment's "Submap / local correlation" section. Active landmarks'
    // positions are read live from the joint state; retired ones are
    // whatever their estimate was at the moment they were evicted (no
    // longer updated by anything unless reactivated).
    std::vector<LandmarkEstimate> Landmarks() const;

private:
    // The measurement Jacobian H, innovation y = z - h(x), and innovation
    // covariance S = H*P*H^T + R for a given EXISTING landmark against a
    // fresh detection -- shared by CorrectMatchedLandmark (which needs all
    // three to actually apply the correction) and CorrectOrAddLandmark's
    // active-landmark search loop (which only needs y/S to compute a
    // Mahalanobis distance for gating). Factored out specifically because
    // this Jacobian is fiddly enough that duplicating it in two places
    // would be a real correctness risk if they ever drifted out of sync.
    void LandmarkInnovation(int _landmarkIndex, double _measuredBodyX, double _measuredBodyY,
                             double _stddev, Eigen::Vector2d &_y, Eigen::Matrix2d &_S,
                             Eigen::MatrixXd &_H) const;
    void CorrectMatchedLandmark(int _landmarkIndex, double _measuredBodyX,
                                 double _measuredBodyY, double _stddev);
    void AddLandmark(double _measuredBodyX, double _measuredBodyY,
                      ConeColor _color, double _stddev);
    // Removes active landmark _index from m_x/m_P/m_landmarkColors/
    // m_landmarkLastSeen entirely (NOT a retirement -- caller is
    // responsible for saving its estimate into m_retiredLandmarks first
    // if that's the intent, see EvictStaleIfOverCapacity). Rebuilds
    // m_x/m_P into fresh, smaller objects rather than shifting the
    // existing ones in place -- an in-place block shift risks Eigen
    // aliasing bugs (reading and writing overlapping regions of the same
    // matrix in one assignment); this is a rare, one-off-per-eviction
    // operation, so a temporary allocation costs nothing meaningful.
    void RemoveActiveLandmark(size_t _index);
    // If the active count exceeds kMaxActiveLandmarks, repeatedly retires
    // the LEAST RECENTLY matched/added active landmark (m_landmarkLastSeen)
    // until back at capacity. Recency, not distance from the vehicle's
    // current position estimate, is deliberately used as the eviction
    // signal: it doesn't depend on the very pose estimate this whole
    // mechanism exists to help keep correctable, and since the car moves
    // through the track roughly continuously (no teleporting), "not seen
    // in a while" is already a good proxy for "physically far away now".
    void EvictStaleIfOverCapacity();

    Eigen::VectorXd m_x;
    Eigen::MatrixXd m_P;
    // m_landmarkColors[i] / m_landmarkLastSeen[i] correspond to state
    // indices kVehicleStateDim + 2*i (x) and kVehicleStateDim + 2*i + 1
    // (y) -- ACTIVE landmarks only. m_landmarkLastSeen[i] is the m_tick
    // value at the most recent match or add for that landmark.
    std::vector<ConeColor> m_landmarkColors;
    std::vector<uint64_t> m_landmarkLastSeen;
    uint64_t m_tick = 0;
    // Evicted landmarks -- no longer part of the joint state (see the
    // class comment), kept only for Landmarks() reporting and for
    // CorrectOrAddLandmark's reactivation check.
    std::vector<LandmarkEstimate> m_retiredLandmarks;
};

} // namespace fsd
