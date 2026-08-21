#include "ekf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

// TEMPORARY diagnostic instrumentation -- prints every kDiagLogEvery-th call
// to each correction/predict function in full detail, indefinitely. Meant to
// be stripped back out once the current early-divergence investigation is
// resolved; NOT meant to stay in the codebase long-term (unthrottled verbose
// logging like this is exactly the kind of stdout cost perception's own
// per-detection printing was just flagged as contributing to real
// per-cycle latency).
//
// Deliberately throttled-but-indefinite, NOT a first-N-calls cutoff -- a
// first-N-calls cutoff was tried first and was a real bug in its own right:
// GNSS/ground_speed fire at ~500-1000Hz, so even a generous few-thousand-call
// budget is exhausted within seconds of process start, going permanently
// silent well before a divergence event that only shows up later in a
// longer test (confirmed directly: a manual-drive test's actual divergence
// was invisible in the log because logging had already stopped by the time
// it happened, even though live /estimated_pose vs. ground-truth comparison
// showed the error growing in real time).
namespace
{
constexpr int kDiagLogEvery = 50;
} // namespace

namespace
{
double NormalizeAngle(double _angle)
{
    while (_angle > M_PI) _angle -= 2.0 * M_PI;
    while (_angle < -M_PI) _angle += 2.0 * M_PI;
    return _angle;
}

// Mahalanobis-distance data-association gate (chi-squared threshold, 2
// DOF), replacing an earlier fixed 1.0m Euclidean gate. The fixed-radius
// version broke down once the vehicle pose could actively drift (verified
// directly in practice): a real cone's PREDICTED body-frame position
// moves as the pose estimate drifts, so a fixed radius eventually rejects
// a legitimate re-observation, which then gets (wrongly) added as a brand
// new landmark -- exactly what kept driving the landmark count to its
// cap even after fixing the earlier, more acute bugs. Mahalanobis
// distance divides the discrepancy by the innovation covariance S =
// H*P*H^T + R -- the filter's own ACTUAL uncertainty about where this
// landmark should appear right now -- so the effective gate automatically
// widens when the filter is less sure of itself and tightens when it's
// confident, rather than using one fixed number regardless of context.
// This is standard EKF-SLAM data association (individual-compatibility
// nearest neighbor); 5.99 is the standard 95%-confidence chi-squared
// critical value for 2 degrees of freedom (closed form for chi-squared at
// k=2: CDF(x) = 1 - e^(-x/2), solving 1-e^(-x/2)=0.95 gives
// x = -2*ln(0.05) = 5.99).
constexpr double kLandmarkGateChiSq = 5.99;

// TEMPORARY: dropped tremendously (from 600) while debugging the
// landmark-duplication problem -- a lower cap means a real duplication
// bug hits it, and stops growing, within seconds instead of requiring a
// long test run to even notice, and keeps the retired-match search loop
// (O(m) per detection over retired count) cheap and easy to reason about
// by hand. 60 is 6x kMaxActiveLandmarks below, generous margin without
// being so large a real bug takes a while to surface. Restore to a real
// value once debugged.
constexpr size_t kMaxLandmarks = 60;

// TEMPORARY: dropped tremendously (from 80, per explicit instruction) while
// debugging the landmark-duplication problem -- this is normally the
// performance-driven cap (bounds every correction's O(n^2) cost via
// n = kVehicleStateDim + 2*kMaxActiveLandmarks). Raised from 10 to 25 while
// debugging landmark jitter: confirmed directly that 10 was FAR below how
// many distinct cones are visible at once, causing near-constant evict/
// reactivate churn -- every landmark kept getting bounced out of the joint
// state (discarding its accumulated correlation structure, only a plain
// diagonal variance snapshot surviving via reactivation blending) before
// it ever got the chance to converge through several consecutive, properly
// -correlated Kalman updates the way a landmark's variance is supposed to
// monotonically shrink under this file's math (landmarks are static, so
// Predict() adds zero process noise to their dimensions -- every real
// observation should only ever tighten Pll, never re-widen it). Still well
// below the real value of 80, so debug output stays tractable. Restore to
// a real value once jitter is fully debugged.
constexpr size_t kMaxActiveLandmarks = 25;

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
// physical cone keeps failing the data-association gate (see
// kLandmarkGateChiSq above) against its own just-added (already-wrong)
// estimate, so this doesn't happen once per cone -- it can repeat every
// frame for every currently-visible cone until the filter converges,
// which is what actually produces a landmark count multiples of the real
// cone count, not just a handful of outliers.
//
// Gating new landmark CREATION (not matching -- an existing landmark can
// still be corrected regardless of current vehicle uncertainty) on the
// vehicle's own position variance being below this threshold means a
// landmark is only ever seeded once the filter already has a reasonably
// trustworthy fix on where the car is. 4.0 (m^2, i.e. ~2m stddev) is
// comfortably tighter than the >=5m cone spacing (Formula Student rules)
// this file already relies on elsewhere, and loose enough that any real
// correction (GNSS, heading) converges past it well before the car would
// plausibly already be near its first cone.
constexpr double kMaxVehiclePosVarianceForNewLandmark = 4.0; // meters^2

// Minimum variance (per axis) assumed when gating a retired-landmark
// re-match, regardless of how tight the landmark's own retained Pll
// diagonal or the vehicle's current position variance are. This gate's
// two failure directions are NOT symmetric: a false reject means a
// legitimately-revisited cone gets re-added as a brand new landmark --
// confirmed directly as the dominant driver of runaway landmark growth
// (600-landmark cap hit within ~12s of normal driving) -- and repeats
// EVERY time that same cone is seen again, since the freshly-added
// duplicate immediately becomes just as susceptible to the same failure
// once IT retires in turn. A false accept just reactivates a nearby but
// wrong landmark, which is comparatively harmless: real cones are >=5m
// apart (Formula Student rules, same margin kMaxVehiclePosVarianceForNewLandmark
// relies on), so this floor -- corresponding to a ~1m per-axis stddev,
// chi-squared gate of kLandmarkGateChiSq across 2 DOF -- is nowhere near
// large enough to confuse two distinct real cones.
constexpr double kMinRetiredMatchVariance = 1.0; // meters^2

// StabilizeCovariance's touched-indices scope for a vehicle-only
// correction (BodyVel, YawRate, GNSS, Heading, Predict) -- see
// StabilizeCovariance's declaration in ekf.hpp for why this is scoped at
// all rather than a full n x n scan.
const std::vector<int> kVehicleDims = {0, 1, 2, 3, 4, 5};

// Shared by PruneStaleRetiredDuplicates/PruneStaleActiveDuplicates -- see
// their declarations in ekf.hpp. BUG FIX: this was 2.0m, based on this
// file's other comments repeatedly citing ">=5m real cone spacing (FSAE
// rules)" -- that assumption is WRONG for this specific simulated track,
// confirmed directly by querying Gazebo's own /world/trackdrive/scene/info
// service for real cone_blue_* positions: consecutive cones measured
// ~1.97m apart, not >=5m. At 2.0m, this was actively MERGING two
// genuinely distinct, legitimately-closely-spaced real cones into one --
// confirmed directly: a detection would correctly fail the (real,
// statistical) Mahalanobis gate against its actual nearest neighbor
// (Mahalanobis 20-1200, correctly signaling "this is a DIFFERENT cone"),
// get created as a new landmark via AddLandmark (correct), and then
// immediately get discarded by this prune check because it was within
// 2.0m of that same, genuinely-different neighbor -- an infinite create-
// then-immediately-prune cycle that looked identical to "landmark
// disappearing" from outside.
//
// Then dropped to 0.75, which went too far the OTHER way: cross-checking
// live /estimated_landmarks against Gazebo's real cone_* positions
// directly (matching each estimate to its nearest same-color real cone)
// found 5 confirmed straggler duplicate pairs sitting 0.76-1.35m apart --
// a well-converged landmark plus a slightly-off duplicate that 0.75 was
// too tight to catch. 1.5m sits roughly halfway between the worst
// confirmed straggler (1.35m) and the confirmed real minimum spacing
// (1.97m), with real margin on both sides: comfortably catches every
// duplicate distance actually observed so far, while staying well clear
// of merging two genuinely distinct adjacent cones.
constexpr double kDuplicatePruneRadius = 1.5; // meters
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

void Ekf::StabilizeCovariance(const std::vector<int> &_touchedIndices)
{
    // Symmetrize: average P with its own transpose, but ONLY the touched
    // rows against the rest of the matrix (O(k*n), not a full O(n^2) pass
    // -- this used to be a full m_P = 0.5*(m_P+m_P.transpose()), which was
    // confirmed as the dominant remaining cost even AFTER scoping the
    // off-diagonal clamp below: it's the same asymptotic order as the
    // O(n^2) covariance update itself, called from every single
    // correction (including the high-rate ones -- GNSS/ground-speed/IMU at
    // ~500-1000Hz -- not just landmark corrections), and it allocates a
    // FULL fresh n x n transpose temporary every time. The correction this
    // follows (P -= K*HP) genuinely does perturb every entry of P, not
    // just the touched rows/columns -- so in principle a scoped symmetrize
    // could miss asymmetry elsewhere -- but the MAGNITUDE of that
    // perturbation in an untouched entry P(i,j) is bounded by how
    // correlated i,j are with the touched dims through K/HP, which decays
    // for less-related state, and any asymmetry large enough to actually
    // matter (i.e. get exploited by a LATER correction's K=P*H^T/S) can
    // only appear in an entry that correction's OWN touched indices cover
    // -- which its OWN call to this function fixes. Same argument as the
    // clamp below, just applied to symmetry instead of the Cauchy-Schwarz
    // bound.
    for (int i : _touchedIndices)
    {
        for (int j = 0; j < m_P.cols(); ++j)
        {
            if (j == i)
            {
                continue;
            }
            const double avg = 0.5 * (m_P(i, j) + m_P(j, i));
            m_P(i, j) = avg;
            m_P(j, i) = avg;
        }
    }

    // Floor the diagonal: a negative "variance" is not just imprecise,
    // it's meaningless (see this method's declaration in ekf.hpp for the
    // concrete P(1,1)<0 case that motivated this). A tiny positive floor
    // (not zero) keeps every future correction's Kalman gain
    // well-defined -- a hard zero could still produce a degenerate S in
    // some later correction. O(n), not O(n^2) -- cheap regardless of
    // scope, so this stays a full scan too.
    constexpr double kMinVariance = 1.0e-9;
    for (int i = 0; i < m_P.rows(); ++i)
    {
        if (m_P(i, i) < kMinVariance)
        {
            m_P(i, i) = kMinVariance;
        }
    }

    // Clamp off-diagonal entries among _touchedIndices to the range a
    // VALID covariance matrix permits: |P(i,j)| <= sqrt(P(i,i)*P(j,j))
    // (Cauchy-Schwarz -- a correlation coefficient can't exceed 1 in
    // magnitude). See this method's declaration in ekf.hpp for why this
    // is scoped to _touchedIndices (a handful of dims: the 6 vehicle ones,
    // plus a landmark's own 2 when this call came from a landmark
    // correction) rather than a full n x n scan -- a full scan here was
    // a confirmed, severe (7-17ms/message) performance regression once
    // active landmark count grew toward kMaxActiveLandmarks, and every
    // exploitable corruption pathway in this file (a corrupted entry
    // actually amplified into a state jump by some correction's K=P*H^T/S)
    // provably falls within some correction's own _touchedIndices, since
    // no H matrix in this file ever spans two DIFFERENT landmarks at once.
    const size_t k = _touchedIndices.size();
    for (size_t a = 0; a < k; ++a)
    {
        const int i = _touchedIndices[a];
        for (size_t b = a + 1; b < k; ++b)
        {
            const int j = _touchedIndices[b];
            const double bound = std::sqrt(m_P(i, i) * m_P(j, j));
            if (m_P(i, j) > bound)
            {
                m_P(i, j) = bound;
                m_P(j, i) = bound;
            }
            else if (m_P(i, j) < -bound)
            {
                m_P(i, j) = -bound;
                m_P(j, i) = -bound;
            }
        }
    }
}

Eigen::MatrixXd Ekf::ApplyCorrection(const std::vector<int> &_touchedIndices,
                                      const Eigen::MatrixXd &_Hsub,
                                      const Eigen::VectorXd &_y,
                                      const Eigen::MatrixXd &_R)
{
    const int n = static_cast<int>(m_x.size());
    const int k = static_cast<int>(_touchedIndices.size());

    // P's touched rows (k x n) -- P is symmetric, so these are also P's
    // touched COLUMNS, which is what H*P actually needs (H is zero outside
    // these columns). See this method's declaration in ekf.hpp for why
    // this replaces a naive dense H*P/P*H^T that Eigen would otherwise
    // compute in full despite H being mostly zero.
    Eigen::MatrixXd Prows(k, n);
    for (int r = 0; r < k; ++r)
    {
        Prows.row(r) = m_P.row(_touchedIndices[r]);
    }

    const Eigen::MatrixXd HP = _Hsub * Prows; // measDim x n -- O(measDim*k*n)

    Eigen::MatrixXd HPtouched(_Hsub.rows(), k); // HP's touched columns
    for (int c = 0; c < k; ++c)
    {
        HPtouched.col(c) = HP.col(_touchedIndices[c]);
    }
    const Eigen::MatrixXd S = HPtouched * _Hsub.transpose() + _R; // measDim x measDim

    // K = P*H^T*S^-1 = (H*P)^T*S^-1 (P symmetric) -- reuses HP instead of
    // computing P*H^T as a SEPARATE dense multiply the way the original
    // per-function code did.
    const Eigen::MatrixXd K = HP.transpose() * S.inverse(); // n x measDim -- O(n*measDim^2)

    m_x += K * _y;
    m_x(2) = NormalizeAngle(m_x(2));
    m_P = m_P - K * HP; // n x n -- O(n^2), the one part that's genuinely unavoidable
    StabilizeCovariance(_touchedIndices);

    return K;
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

    const double dx = (vx * cosYaw - vy * sinYaw) * _dt;
    const double dy = (vx * sinYaw + vy * cosYaw) * _dt;
    {
        static int calls = 0;
        if (calls % kDiagLogEvery == 0)
        {
            std::fprintf(stderr,
                "[DIAG Predict] #%d dt=%.6f yaw=%.4f vx=%.4f vy=%.4f yawRate=%.4f -> dx=%.6f dy=%.6f | x=%.3f y=%.3f\n",
                calls, _dt, yaw, vx, vy, yawRate, dx, dy, m_x(0), m_x(1));
        }
        ++calls;
    }

    m_x(0) += dx;
    m_x(1) += dy;
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
    StabilizeCovariance(kVehicleDims);
}

void Ekf::CorrectBodyVelocity(double _vx, double _vy, double _stddevVx, double _stddevVy)
{
    const Eigen::Vector2d z(_vx, _vy);
    const Eigen::Vector2d h(m_x(3), m_x(4));
    const Eigen::Vector2d y = z - h;

    Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
    R(0, 0) = _stddevVx * _stddevVx;
    R(1, 1) = _stddevVy * _stddevVy;

    const Eigen::Matrix2d Hsub = Eigen::Matrix2d::Identity(); // touches {3,4}
    const Eigen::MatrixXd K = ApplyCorrection({3, 4}, Hsub, y, R);

    static int diagCalls = 0;
    if (diagCalls % kDiagLogEvery == 0)
    {
        std::fprintf(stderr,
            "[DIAG BodyVel] #%d z=(%.4f,%.4f) h=(%.4f,%.4f) y=(%.4f,%.4f) K30=%.4f K41=%.4f\n",
            diagCalls, z(0), z(1), h(0), h(1), y(0), y(1), K(3, 0), K(4, 1));
    }
    ++diagCalls;
}

void Ekf::CorrectYawRate(double _yawRate, double _stddev)
{
    const double h = m_x(5);
    const double y = _yawRate - h;

    static int diagCalls = 0;
    const bool diag = diagCalls % kDiagLogEvery == 0;
    double beforeX = 0, beforeY = 0;
    if (diag)
    {
        beforeX = m_x(0);
        beforeY = m_x(1);
    }

    Eigen::MatrixXd Hsub(1, 1);
    Hsub(0, 0) = 1.0; // touches {5}
    Eigen::MatrixXd R(1, 1);
    R(0, 0) = _stddev * _stddev;
    Eigen::VectorXd yv(1);
    yv(0) = y;
    const Eigen::MatrixXd K = ApplyCorrection({5}, Hsub, yv, R);

    if (diag)
    {
        std::fprintf(stderr,
            "[DIAG YawRate] #%d z=%.4f h=%.4f y=%.4f K1=%.4f | before x=%.3f y=%.3f -> after x=%.3f y=%.3f\n",
            diagCalls, _yawRate, h, y, K(1, 0), beforeX, beforeY, m_x(0), m_x(1));
    }
    ++diagCalls;
}

void Ekf::CorrectGnssPosition(double _measuredEast, double _measuredNorth,
                               double _antennaOffsetX, double _antennaOffsetY,
                               double _stddev)
{
    const double yaw = m_x(2);
    const double cosYaw = std::cos(yaw);
    const double sinYaw = std::sin(yaw);

    const Eigen::Vector2d h(
        m_x(0) + _antennaOffsetX * cosYaw - _antennaOffsetY * sinYaw,
        m_x(1) + _antennaOffsetX * sinYaw + _antennaOffsetY * cosYaw);
    const Eigen::Vector2d z(_measuredEast, _measuredNorth);
    const Eigen::Vector2d y = z - h;

    Eigen::MatrixXd Hsub(2, 3); // touches {0, 1, 2}
    Hsub(0, 0) = 1.0;
    Hsub(0, 1) = 0.0;
    Hsub(0, 2) = -_antennaOffsetX * sinYaw - _antennaOffsetY * cosYaw;
    Hsub(1, 0) = 0.0;
    Hsub(1, 1) = 1.0;
    Hsub(1, 2) = _antennaOffsetX * cosYaw - _antennaOffsetY * sinYaw;

    const Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * (_stddev * _stddev);

    static int diagCalls = 0;
    const bool diag = diagCalls % kDiagLogEvery == 0;
    double beforeX = 0, beforeY = 0, beforeP00 = 0, beforeP11 = 0;
    if (diag)
    {
        beforeX = m_x(0);
        beforeY = m_x(1);
        beforeP00 = m_P(0, 0);
        beforeP11 = m_P(1, 1);
    }

    const Eigen::MatrixXd K = ApplyCorrection({0, 1, 2}, Hsub, y, R);

    if (diag)
    {
        std::fprintf(stderr,
            "[DIAG GNSS] #%d z=(%.3f,%.3f) h=(%.3f,%.3f) y=(%.3f,%.3f) K00=%.4f K11=%.4f | "
            "before x=%.3f y=%.3f P00=%.3e P11=%.3e -> after x=%.3f y=%.3f P00=%.3e P11=%.3e\n",
            diagCalls, z(0), z(1), h(0), h(1), y(0), y(1), K(0, 0), K(1, 1),
            beforeX, beforeY, beforeP00, beforeP11, m_x(0), m_x(1), m_P(0, 0), m_P(1, 1));
    }
    ++diagCalls;
}

void Ekf::CorrectHeading(double _measuredYaw, double _stddev)
{
    const double h = m_x(2);
    const double y = NormalizeAngle(_measuredYaw - h);

    static int diagCalls = 0;
    const bool diag = diagCalls % kDiagLogEvery == 0;
    double beforeX = 0, beforeY = 0;
    if (diag)
    {
        beforeX = m_x(0);
        beforeY = m_x(1);
    }

    Eigen::MatrixXd Hsub(1, 1);
    Hsub(0, 0) = 1.0; // touches {2}
    Eigen::MatrixXd R(1, 1);
    R(0, 0) = _stddev * _stddev;
    Eigen::VectorXd yv(1);
    yv(0) = y;
    const Eigen::MatrixXd K = ApplyCorrection({2}, Hsub, yv, R);

    if (diag)
    {
        std::fprintf(stderr,
            "[DIAG Heading] #%d z=%.4f h=%.4f y=%.4f K1=%.4f | before x=%.3f y=%.3f -> after x=%.3f y=%.3f\n",
            diagCalls, _measuredYaw, h, y, K(1, 0), beforeX, beforeY, m_x(0), m_x(1));
    }
    ++diagCalls;
}

void Ekf::CorrectOrAddLandmark(double _measuredBodyX, double _measuredBodyY,
                                ConeColor _color, double _stddev)
{
    const double yaw = m_x(2);
    const double cosYaw = std::cos(yaw);
    const double sinYaw = std::sin(yaw);
    const double worldX = m_x(0) + _measuredBodyX * cosYaw - _measuredBodyY * sinYaw;
    const double worldY = m_x(1) + _measuredBodyX * sinYaw + _measuredBodyY * cosYaw;

    // 1. Try to match against an ACTIVE (in the joint state) landmark, by
    // Mahalanobis distance -- see kLandmarkGateChiSq's comment for why a
    // fixed Euclidean radius isn't enough once the pose can drift. Note
    // this costs O(n) PER CANDIDATE now (LandmarkInnovation forms a 2xn H
    // and multiplies it through P), not the O(1) a plain coordinate
    // difference was -- acceptable because kMaxActiveLandmarks already
    // bounds both n and the candidate count to something small.
    int bestIndex = -1;
    double bestMahalanobisSq = kLandmarkGateChiSq;
    // TEMPORARY (landmark-duplication debugging): track the closest
    // same-color active candidate's Mahalanobis value REGARDLESS of
    // whether it passed the gate, and its Euclidean distance, so the
    // AddLandmark diagnostic below can show WHY a real match was missed
    // (nothing nearby at all, vs. something nearby that failed the gate
    // and by how much) instead of just "no match found".
    double closestActiveDistSq = -1.0;
    double closestActiveMahalanobisSq = -1.0;
    for (size_t i = 0; i < m_landmarkColors.size(); ++i)
    {
        if (m_landmarkColors[i] != _color)
        {
            continue;
        }
        // Cheap Euclidean pre-filter before paying for LandmarkInnovation's
        // O(n) Prows extraction + matrix work: kMaxActiveLandmarks candidates
        // get tried per detection, and most of them -- anything the car
        // isn't currently near -- can never plausibly pass the real
        // Mahalanobis gate, so there's no reason to build S for them at
        // all. kCoarseGateRadius is deliberately generous (much larger
        // than the real gate's effective radius under normal, converged
        // uncertainty) so this can only ever SKIP a candidate the real
        // gate would also have rejected -- it changes what gets computed,
        // never what gets matched.
        const int li = kVehicleStateDim + 2 * static_cast<int>(i);
        const double dxCoarse = m_x(li) - worldX;
        const double dyCoarse = m_x(li + 1) - worldY;
        const double distSq = dxCoarse * dxCoarse + dyCoarse * dyCoarse;
        if (closestActiveDistSq < 0.0 || distSq < closestActiveDistSq)
        {
            closestActiveDistSq = distSq;
        }
        constexpr double kCoarseGateRadius = 15.0; // meters -- generous margin above the
                                                    // real gate's typical effective radius,
                                                    // since a converged filter's P is small
                                                    // but P legitimately grows during real
                                                    // drift/uncertainty, and this must never
                                                    // reject a candidate the real (uncertainty
                                                    // -aware) Mahalanobis gate would accept
        if (distSq > kCoarseGateRadius * kCoarseGateRadius)
        {
            continue;
        }
        Eigen::Vector2d y;
        Eigen::Matrix2d S;
        Eigen::MatrixXd Hcols; // unused here besides the out-param -- only y/S are needed for gating
        LandmarkInnovation(static_cast<int>(i), _measuredBodyX, _measuredBodyY, _stddev, y, S, Hcols);
        const double mahalanobisSq = y.transpose() * S.inverse() * y;
        if (closestActiveMahalanobisSq < 0.0 || mahalanobisSq < closestActiveMahalanobisSq)
        {
            closestActiveMahalanobisSq = mahalanobisSq;
        }
        if (mahalanobisSq < bestMahalanobisSq)
        {
            bestMahalanobisSq = mahalanobisSq;
            bestIndex = static_cast<int>(i);
        }
    }

    if (bestIndex >= 0)
    {
        static int matchDiagCalls = 0;
        if (matchDiagCalls % kDiagLogEvery == 0)
        {
            std::fprintf(stderr,
                "[DIAG LandmarkMatch] #%d color=%d world=(%.2f,%.2f) matchedIndex=%d mahalanobisSq=%.3f active=%zu retired=%zu\n",
                matchDiagCalls, static_cast<int>(_color), worldX, worldY, bestIndex,
                bestMahalanobisSq, m_landmarkColors.size(), m_retiredLandmarks.size());
        }
        ++matchDiagCalls;
        CorrectMatchedLandmark(bestIndex, _measuredBodyX, _measuredBodyY, _stddev);
        EvictStaleIfOverCapacity();
        // An active match means the retired list was never even consulted
        // this call -- exactly the situation that lets a stale retired
        // duplicate of THIS landmark linger forever (see
        // PruneStaleRetiredDuplicates's declaration in ekf.hpp).
        PruneStaleRetiredDuplicates();
        PruneStaleActiveDuplicates();
        return;
    }

    // 2. No active match -- check RETIRED landmarks. A match here means
    // the car is revisiting a landmark it previously evicted from the
    // active state (see the class comment). Reactivate it by dropping it
    // from m_retiredLandmarks and falling through to the "add" path
    // below, using THIS detection -- simpler and safer than inventing a
    // second, seeded-from-the-retired-estimate covariance construction;
    // AddLandmark's existing Jacobian-based augmentation already
    // correctly handles "new landmark from a fresh detection", and
    // reusing it here means there's only one place that math needs to be
    // right.
    //
    // A retired landmark has no LIVE tracked covariance to build a proper
    // Mahalanobis distance from (see the class comment -- that's the whole
    // point of retiring it), but it DOES carry its own Pll diagonal
    // snapshot from the moment it was retired (LandmarkEstimate::varX/varY
    // -- see EvictStaleIfOverCapacity). That, not the vehicle's own
    // CURRENT position variance, is the dominant source of "how far off
    // could this legitimately be": the vehicle's own P(0,0)/P(1,1)
    // converges to ~1e-5 within the first second or two of any run
    // (confirmed directly via diagnostic instrumentation) regardless of
    // how well-constrained any given landmark was, which made this gate's
    // effective tolerance a few TENTHS of a meter -- tighter than
    // realistic re-detection noise -- and silently forced nearly every
    // re-observation of a retired landmark to fail the gate and get
    // re-added as brand new, which is what was actually driving the
    // landmark count to its cap on every real driving lap (confirmed: it
    // hit the 600 cap within ~12 seconds of a normal drive). The vehicle's
    // current variance is still added in (a diagonal approximation: no
    // cross-correlation term, since none is available post-retirement) --
    // the vehicle's own pose can still have drifted since the landmark was
    // retired -- along with the measurement variance as a floor.
    // TEMPORARY (landmark-duplication debugging): same closest-candidate
    // tracking as the active search above, for the AddLandmark diagnostic
    // below.
    //
    // BUG FIX: this used to erase and reactivate the FIRST retired
    // candidate that passed the gate, in list order -- not the BEST
    // (closest) one, unlike the active search above (which correctly
    // tracks bestMahalanobisSq). With enough retired landmarks
    // accumulated, a detection could match some coincidentally-nearby-
    // enough retired entry that happened to come first in the list while
    // the TRUE match (the actual same physical cone's retired copy,
    // sitting right next to the detection) got left behind unclaimed --
    // confirmed directly: a detection at (117.24,38.60) reactivated a
    // retired candidate 2.01m away (barely under the gate) while the
    // genuine match, evicted moments earlier from the exact same spot
    // (117.20,38.53), was still sitting in the list. The abandoned true
    // match then never gets reclaimed (nothing else has any reason to
    // prefer it over whatever ELSE also happens to pass the gate first),
    // so it just sits there as a near-duplicate of wherever the detection
    // actually ended up. Now tracks the best (lowest Mahalanobis) match
    // across ALL retired candidates, same as the active search, and only
    // erases/reactivates that one.
    bool reactivated = false;
    LandmarkEstimate reactivatedPrior{}; // saved before erase() invalidates it -- passed to AddLandmark below
    int bestRetiredIndex = -1;
    double bestRetiredMahalanobisSq = kLandmarkGateChiSq;
    double closestRetiredDistSq = -1.0;
    double closestRetiredMahalanobisSq = -1.0;
    for (size_t i = 0; i < m_retiredLandmarks.size(); ++i)
    {
        if (m_retiredLandmarks[i].color != _color)
        {
            continue;
        }
        const double dx = m_retiredLandmarks[i].x - worldX;
        const double dy = m_retiredLandmarks[i].y - worldY;
        const double distSq = dx * dx + dy * dy;
        if (closestRetiredDistSq < 0.0 || distSq < closestRetiredDistSq)
        {
            closestRetiredDistSq = distSq;
        }
        const double varX = std::max(kMinRetiredMatchVariance,
            m_retiredLandmarks[i].varX + m_P(0, 0) + _stddev * _stddev);
        const double varY = std::max(kMinRetiredMatchVariance,
            m_retiredLandmarks[i].varY + m_P(1, 1) + _stddev * _stddev);
        const double approxMahalanobisSq = (dx * dx) / varX + (dy * dy) / varY;
        if (closestRetiredMahalanobisSq < 0.0 || approxMahalanobisSq < closestRetiredMahalanobisSq)
        {
            closestRetiredMahalanobisSq = approxMahalanobisSq;
        }
        if (approxMahalanobisSq < bestRetiredMahalanobisSq)
        {
            bestRetiredMahalanobisSq = approxMahalanobisSq;
            bestRetiredIndex = static_cast<int>(i);
        }
    }
    if (bestRetiredIndex >= 0)
    {
        reactivatedPrior = m_retiredLandmarks[bestRetiredIndex];
        m_retiredLandmarks.erase(m_retiredLandmarks.begin() + bestRetiredIndex);
        reactivated = true;
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

    // TEMPORARY (landmark-duplication debugging): every AddLandmark call
    // is either a genuinely new cone or a symptom of a matching-gate
    // failure -- this print shows which, and if it's the latter, how
    // close the nearest real candidate actually was and why it didn't
    // pass (distance alone doesn't say whether the failure was the
    // active gate or the retired one, so both are reported).
    std::fprintf(stderr,
        "[DIAG AddLandmark] color=%d world=(%.2f,%.2f) reactivated=%d "
        "closestActiveDist=%.2f closestActiveMahalanobisSq=%.2f "
        "closestRetiredDist=%.2f closestRetiredMahalanobisSq=%.2f "
        "active=%zu retired=%zu\n",
        static_cast<int>(_color), worldX, worldY, reactivated ? 1 : 0,
        closestActiveDistSq < 0.0 ? -1.0 : std::sqrt(closestActiveDistSq), closestActiveMahalanobisSq,
        closestRetiredDistSq < 0.0 ? -1.0 : std::sqrt(closestRetiredDistSq), closestRetiredMahalanobisSq,
        m_landmarkColors.size(), m_retiredLandmarks.size());

    AddLandmark(_measuredBodyX, _measuredBodyY, _color, _stddev,
                reactivated ? &reactivatedPrior : nullptr);
    EvictStaleIfOverCapacity();
    PruneStaleRetiredDuplicates();
    PruneStaleActiveDuplicates();
}

void Ekf::LandmarkInnovation(int _landmarkIndex, double _measuredBodyX, double _measuredBodyY,
                              double _stddev, Eigen::Vector2d &_y, Eigen::Matrix2d &_S,
                              Eigen::MatrixXd &_Hcols) const
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
    _y = z - h;

    // H's 5 nonzero columns {0,1,2,li,li+1} (vehicle x/y/yaw + this
    // landmark's own x/y), computed DIRECTLY into a small 2x5 matrix --
    // never as a full 2 x n H that's mostly zero. That full-H-then-extract
    // step used to exist here (build Zero(2,n), fill 10 entries, then copy
    // the same 5 nonzero columns right back out into Hcols) purely to feed
    // ApplyCorrection/CorrectMatchedLandmark's K computation -- once that
    // was rewritten to take the sparse Hcols directly (see ApplyCorrection
    // in ekf.hpp for why), the full H had no remaining use anywhere in this
    // file, so building it at all was pure waste.
    _Hcols = Eigen::MatrixXd(2, 5);
    _Hcols(0, 0) = -cosYaw;
    _Hcols(0, 1) = -sinYaw;
    _Hcols(0, 2) = bodyYPred;
    _Hcols(1, 0) = sinYaw;
    _Hcols(1, 1) = -cosYaw;
    _Hcols(1, 2) = -bodyXPred;
    // w.r.t. the matched landmark's own (x, y) state -- NEW vs. the
    // pre-SLAM version, since the landmark is now part of the state being
    // differentiated against, not a constant.
    _Hcols(0, 3) = cosYaw;
    _Hcols(0, 4) = sinYaw;
    _Hcols(1, 3) = -sinYaw;
    _Hcols(1, 4) = cosYaw;

    const Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * (_stddev * _stddev);

    // S = H*P*H^T + R, WITHOUT a naive dense 2xn * nxn multiply -- H is
    // nonzero only in 5 columns (vehicle x/y/yaw + this landmark's own
    // x/y), so H*P only depends on the matching 5 ROWS of P (same
    // sparse-aware technique already used throughout this file, e.g.
    // Predict()'s A = Fd*P.topRows(6), or ApplyCorrection's own Prows
    // extraction). This matters MORE here than in a single accepted
    // correction: LandmarkInnovation is called once PER CANDIDATE during
    // data-association search (see CorrectOrAddLandmark), so a naive
    // O(n^2) S computation costs O(n^2) PER CANDIDATE evaluated, not just
    // once -- confirmed directly as a real regression during testing
    // (localization briefly cost 400-500ms per /cone_detections message
    // again, worse than before the submap fix) before this optimization
    // was added.
    Eigen::MatrixXd Prows(5, n); // P's matching 5 rows
    Prows.row(0) = m_P.row(0);
    Prows.row(1) = m_P.row(1);
    Prows.row(2) = m_P.row(2);
    Prows.row(3) = m_P.row(li);
    Prows.row(4) = m_P.row(li + 1);

    const Eigen::MatrixXd HP = _Hcols * Prows; // 2 x n -- O(n), not O(n^2)

    Eigen::MatrixXd HPcols(2, 5); // HP's matching 5 columns
    HPcols.col(0) = HP.col(0);
    HPcols.col(1) = HP.col(1);
    HPcols.col(2) = HP.col(2);
    HPcols.col(3) = HP.col(li);
    HPcols.col(4) = HP.col(li + 1);

    _S = HPcols * _Hcols.transpose() + R; // 2x5 * 5x2 -- O(1)
}

void Ekf::CorrectMatchedLandmark(int _landmarkIndex, double _measuredBodyX,
                                  double _measuredBodyY, double _stddev)
{
    Eigen::Vector2d y;
    Eigen::Matrix2d S; // unused here -- ApplyCorrection recomputes S itself
                       // from R below; S is only needed by the search loop
                       // in CorrectOrAddLandmark, which shares this same
                       // LandmarkInnovation call signature.
    Eigen::MatrixXd Hcols;
    LandmarkInnovation(_landmarkIndex, _measuredBodyX, _measuredBodyY, _stddev, y, S, Hcols);

    const Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * (_stddev * _stddev);
    const int li = kVehicleStateDim + 2 * _landmarkIndex;
    ApplyCorrection({0, 1, 2, li, li + 1}, Hcols, y, R);

    m_landmarkLastSeen[static_cast<size_t>(_landmarkIndex)] = ++m_tick;
}

void Ekf::AddLandmark(double _measuredBodyX, double _measuredBodyY,
                       ConeColor _color, double _stddev,
                       const LandmarkEstimate *_priorEstimate)
{
    const int n = static_cast<int>(m_x.size());
    const double yaw = m_x(2);
    const double cosYaw = std::cos(yaw);
    const double sinYaw = std::sin(yaw);

    double lmX = m_x(0) + _measuredBodyX * cosYaw - _measuredBodyY * sinYaw;
    double lmY = m_x(1) + _measuredBodyX * sinYaw + _measuredBodyY * cosYaw;

    // Reactivation: blend the fresh detection's implied position with the
    // retired landmark's own retained estimate, inverse-variance weighted
    // -- standard fusion of two independent Gaussian estimates of the same
    // quantity. See this method's declaration in ekf.hpp for why this
    // matters (jitter from reseeding purely off one fresh, noisy detection
    // every reactivation). _stddev here is the RAW measurement noise, not
    // yet propagated through Gx/Gz below -- close enough as the "fresh"
    // side of this blend, since Gx's own contribution (vehicle pose
    // uncertainty) is typically small relative to _stddev by the time a
    // landmark has already been active once (that's exactly the regime
    // this blend applies in).
    //
    // fusedVarX/Y are ALSO applied to Pll/Pcross below (not just used here
    // for the position blend) -- blending only the position and leaving
    // Pll at its full fresh-detection width was tried first and confirmed
    // insufficient: the landmark's reported uncertainty still looked
    // barely-constrained, so the very NEXT correction (matched or another
    // reactivation) applied just as large a Kalman gain to the next noisy
    // detection as if this were genuinely brand new, and visible jitter
    // persisted. fusedVar is guaranteed <= min(priorVar, freshVar) (a
    // basic property of inverse-variance weighting), so substituting it
    // into Pll's diagonal can only ever tighten, never invalidate, the
    // landmark's uncertainty.
    bool hasPrior = _priorEstimate != nullptr;
    double fusedVarX = 0.0, fusedVarY = 0.0, scaleX = 1.0, scaleY = 1.0;
    if (hasPrior)
    {
        const double freshVar = _stddev * _stddev;
        fusedVarX = 1.0 / (1.0 / _priorEstimate->varX + 1.0 / freshVar);
        fusedVarY = 1.0 / (1.0 / _priorEstimate->varY + 1.0 / freshVar);
        lmX = fusedVarX * (_priorEstimate->x / _priorEstimate->varX + lmX / freshVar);
        lmY = fusedVarY * (_priorEstimate->y / _priorEstimate->varY + lmY / freshVar);
    }

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
    Eigen::MatrixXd Pcross = Gx * m_P.topRows(3);  // 2 x n
    Eigen::Matrix2d Pll =
        Gx * m_P.topLeftCorner(3, 3) * Gx.transpose() + Gz * R * Gz.transpose();

    if (hasPrior)
    {
        // Scale factors < 1 (fusedVar is always <= the fresh-only
        // variance). Off-diagonal/cross terms are scaled by
        // sqrt(scaleX*scaleY) rather than left alone, so the correlation
        // COEFFICIENT (cov / sqrt(varX*varY)) is exactly preserved --
        // shrinking the diagonal while leaving covariance entries
        // unchanged would inflate the implied correlation coefficient,
        // potentially past the Cauchy-Schwarz bound StabilizeCovariance
        // enforces elsewhere in this file, for no reason grounded in the
        // actual math (the prior contributes zero cross-correlation
        // information, so there's no basis to change the correlation
        // STRUCTURE here, only its overall magnitude).
        scaleX = fusedVarX / Pll(0, 0);
        scaleY = fusedVarY / Pll(1, 1);
        const double offDiagScale = std::sqrt(scaleX * scaleY);
        Pll(0, 0) = fusedVarX;
        Pll(1, 1) = fusedVarY;
        Pll(0, 1) *= offDiagScale;
        Pll(1, 0) *= offDiagScale;
        Pcross.row(0) *= scaleX;
        Pcross.row(1) *= scaleY;
    }

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
    StabilizeCovariance(kVehicleDims);
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
        m_retiredLandmarks.push_back(LandmarkEstimate{
            m_x(li), m_x(li + 1), m_landmarkColors[staleIndex],
            m_P(li, li), m_P(li + 1, li + 1)});
        RemoveActiveLandmark(staleIndex);
    }
}

void Ekf::PruneStaleRetiredDuplicates()
{
    for (size_t r = 0; r < m_retiredLandmarks.size();)
    {
        bool isDuplicate = false;
        for (size_t a = 0; a < m_landmarkColors.size(); ++a)
        {
            if (m_landmarkColors[a] != m_retiredLandmarks[r].color)
            {
                continue;
            }
            const int li = kVehicleStateDim + 2 * static_cast<int>(a);
            const double dx = m_x(li) - m_retiredLandmarks[r].x;
            const double dy = m_x(li + 1) - m_retiredLandmarks[r].y;
            if (dx * dx + dy * dy < kDuplicatePruneRadius * kDuplicatePruneRadius)
            {
                isDuplicate = true;
                break;
            }
        }
        if (isDuplicate)
        {
            m_retiredLandmarks.erase(m_retiredLandmarks.begin() + static_cast<long>(r));
        }
        else
        {
            ++r;
        }
    }
}

void Ekf::PruneStaleActiveDuplicates()
{
    for (size_t a = 0; a < m_landmarkColors.size();)
    {
        bool removed = false;
        const int liA = kVehicleStateDim + 2 * static_cast<int>(a);
        for (size_t b = a + 1; b < m_landmarkColors.size(); ++b)
        {
            if (m_landmarkColors[b] != m_landmarkColors[a])
            {
                continue;
            }
            const int liB = kVehicleStateDim + 2 * static_cast<int>(b);
            const double dx = m_x(liA) - m_x(liB);
            const double dy = m_x(liA + 1) - m_x(liB + 1);
            if (dx * dx + dy * dy < kDuplicatePruneRadius * kDuplicatePruneRadius)
            {
                // Keep whichever was seen more recently -- same recency
                // signal EvictStaleIfOverCapacity already uses.
                const size_t toRemove = (m_landmarkLastSeen[a] < m_landmarkLastSeen[b]) ? a : b;
                RemoveActiveLandmark(toRemove);
                removed = true;
                break;
            }
        }
        if (!removed)
        {
            ++a;
        }
        // If removed, re-scan from the SAME index a: RemoveActiveLandmark
        // shifted every later index down by one, so whatever is now at
        // position a (or, if a itself was removed, whatever slid into it)
        // still needs to be checked against the rest of the list.
    }
}

} // namespace fsd
