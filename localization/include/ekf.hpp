#pragma once

#include <cstdint>
#include <unordered_map>
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
        // Positional variance (Pll diagonal) AT THE MOMENT OF RETIREMENT --
        // zero/unused for still-active landmarks. Needed because the
        // retired-landmark re-match gate (see CorrectOrAddLandmark) was
        // using the VEHICLE's own current position variance as a stand-in
        // for "how uncertain am I about this landmark", which collapses to
        // ~1e-5 within the first second or two of any run (see
        // StabilizeCovariance's confirmed P00/P11 convergence) regardless
        // of whether the landmark itself was ever well-constrained -- an
        // effective re-match tolerance of a few tenths of a meter, tighter
        // than realistic re-detection noise, which was silently forcing
        // every re-observation of a retired landmark to fail the gate and
        // get re-added as brand new. Carrying the landmark's OWN retained
        // uncertainty forward gives an honest tolerance instead.
        double varX = 0.0;
        double varY = 0.0;
        // Stable identity, assigned once when a landmark is FIRST created
        // (see AddLandmark) and preserved across every eviction/
        // reactivation cycle after that -- unlike the index into
        // m_x/m_landmarkColors (which shifts every time ANY OTHER active
        // landmark is added/removed) or list position within
        // m_retiredLandmarks, this is the only thing that reliably
        // identifies "the same physical landmark" across time from OUTSIDE
        // this class. Needed because /estimated_landmarks (a plain
        // Pose_V, one entry per CURRENTLY discovered landmark, no index
        // stability guaranteed between messages) made rigorous jitter
        // measurement impossible: matching landmarks between two
        // consecutive polls by nearest-same-color-position is exactly the
        // kind of thing that silently confuses "this landmark moved" with
        // "a DIFFERENT nearby landmark got matched instead" once landmark
        // density is high -- which is precisely the regime jitter needs to
        // be measured in. See localization.cpp's publishLandmarks for how
        // this gets exposed externally.
        uint64_t uid = 0;
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
    // Symmetrizes m_P and floors its diagonal to a small positive value --
    // called after every m_P update. The simplified covariance-update form
    // used throughout this file (P - K*(H*P), chosen for its O(n^2) cost
    // vs. O(n^3) for a naive dense update) is NOT guaranteed to stay
    // symmetric/positive-semi-definite under floating-point rounding error
    // the way the mathematically-equivalent-in-exact-arithmetic Joseph
    // form is -- confirmed as a REAL, not theoretical, failure mode here:
    // diagnostic instrumentation caught P(1,1) going negative at the
    // SECOND EVER GNSS correction, a direct result of the startup P=1e6
    // sitting right next to an already-converged ~1e-4 entry in the same
    // matrix (huge dynamic range = the exact condition that amplifies
    // rounding error). A full Joseph-form rewrite of every correction
    // function would also stay O(n^2) with the same sparse-extraction
    // technique already used elsewhere in this file, but touches 5
    // separate, structurally-different functions; this is a smaller,
    // easier-to-verify-correct fix that directly targets the confirmed
    // symptom.
    //
    // The off-diagonal Cauchy-Schwarz clamp added alongside the diagonal
    // floor (see the confirmed-negative-P00 story above -- the SAME
    // instrumentation later caught an off-diagonal entry going invalid
    // too, exploited by CorrectHeading/CorrectYawRate's rank-1 corrections
    // into a 400+ meter single-call position jump) is deliberately scoped
    // to only the dimensions THIS correction actually touched
    // (_touchedIndices), NOT a full n x n scan -- confirmed directly as a
    // real, severe regression when it WAS a full scan: at n=166
    // (kMaxActiveLandmarks=80), a dense scan is ~14000 sqrt() calls, and
    // with this called from every correction (hundreds of landmark
    // corrections per /cone_detections message alone), per-message
    // /timing/localization went from ~0.2ms to 7-17ms -- an order of
    // magnitude past the ~1-2ms budget available between GNSS/ground-speed
    // messages at their ~500-1000Hz rate, which starved localization of
    // real-time and was the actual cause of /estimated_pose appearing
    // "frozen" during real driving (it wasn't frozen, it was falling
    // further and further behind processing a growing backlog of stale
    // messages). Restricting the clamp to just the touched dims keeps it
    // at the same sparse O(k^2) (k = 6 or 8) scale as every other
    // correction in this file, while still directly guarding the exact
    // mechanism that produced the confirmed corruption (a corrupted
    // vehicle-vehicle or vehicle-landmark entry gets exploited by a LATER
    // correction whose H touches those same dims -- which necessarily
    // means that later correction's OWN _touchedIndices call already
    // covers it, so this loses no real protection against the failure
    // mode that was actually observed).
    //
    // The symmetrize step is ALSO scoped to _touchedIndices now (see the
    // .cpp for the same reasoning applied there) -- scoping just the
    // clamp wasn't enough on its own: the full O(n^2) symmetrize was
    // confirmed as the dominant REMAINING cost even after that fix.
    void StabilizeCovariance(const std::vector<int> &_touchedIndices);

    // Applies one EKF correction step (K = P*H^T*S^-1, x += K*y, P -= K*H*P)
    // for a measurement whose Jacobian H is zero everywhere EXCEPT the
    // columns in _touchedIndices -- _Hsub is just those nonzero columns
    // (measDim x _touchedIndices.size()), in the same order. Every
    // correction function in this file used to build a full,
    // mostly-zero kVehicleStateDim/landmark-wide H and hand it to Eigen's
    // generic dense matrix multiply for K = P*H^T and (later) H*P -- Eigen
    // has no way to know most of H is zero, so both of those were genuinely
    // OPERATION-BY-OPERATION dense O(n^2) multiplies despite H's sparsity
    // being "obvious" from the code around it. This was a confirmed, severe
    // regression (7-17ms per /cone_detections message, when the ~1-2ms
    // budget between GNSS/ground-speed messages at their ~500-1000Hz rate
    // requires sub-millisecond corrections) that was NOT fixed by scoping
    // StabilizeCovariance alone -- that only addressed a much cheaper part
    // of the same per-correction cost. This method does the same
    // extract-the-relevant-rows-of-P technique LandmarkInnovation already
    // used for its S computation (P is symmetric, so P's touched COLUMNS
    // equal its touched ROWS), reusing ONE such extraction for both S and
    // K instead of Eigen silently doing the full O(n^2) multiply two or
    // three times over across a correction's S/K/P-update. The final P
    // update (P -= K*H*P) stays O(n^2) -- that part is unavoidable, since
    // the RESULT is genuinely a dense n x n matrix -- but every step
    // feeding into it drops from O(n^2) to O(k*n) (k = _touchedIndices.size(),
    // a small constant: 6 for a vehicle-only correction, 8 for a landmark
    // one). Also calls StabilizeCovariance(_touchedIndices) internally, so
    // callers don't need a separate call. Returns the full n x measDim K
    // (not reduced to just the touched rows) so callers can still read
    // whatever entries their diagnostic prints reference.
    Eigen::MatrixXd ApplyCorrection(const std::vector<int> &_touchedIndices,
                                     const Eigen::MatrixXd &_Hsub,
                                     const Eigen::VectorXd &_y,
                                     const Eigen::MatrixXd &_R);

    // The innovation y = z - h(x) and the measurement Jacobian's 5 nonzero
    // columns (_Hcols, in column order {vehicle x, vehicle y, vehicle yaw,
    // landmark x, landmark y} -- see ApplyCorrection for why this is just
    // the nonzero columns rather than a full n-wide H) for a given EXISTING
    // landmark against a fresh detection -- shared by CorrectMatchedLandmark
    // (which needs y/H to actually apply the correction, via
    // ApplyCorrection) and CorrectOrAddLandmark's active-landmark search
    // loop (which only needs y/S to compute a Mahalanobis distance for
    // gating). Factored out specifically because this Jacobian is fiddly
    // enough that duplicating it in two places would be a real correctness
    // risk if they ever drifted out of sync.
    void LandmarkInnovation(int _landmarkIndex, double _measuredBodyX, double _measuredBodyY,
                             double _stddev, Eigen::Vector2d &_y, Eigen::Matrix2d &_S,
                             Eigen::MatrixXd &_Hcols) const;
    void CorrectMatchedLandmark(int _landmarkIndex, double _measuredBodyX,
                                 double _measuredBodyY, double _stddev);
    // _priorEstimate is non-null when this is a REACTIVATION (see
    // CorrectOrAddLandmark) -- the retired landmark's own retained
    // (x, y, varX, varY) from before it was evicted. When present, the
    // new landmark's position is a proper inverse-variance-weighted blend
    // of the prior and this fresh detection, not just the fresh detection
    // alone -- confirmed as a real, direct cause of visible landmark
    // jitter: with kMaxActiveLandmarks forcing frequent eviction/
    // reactivation churn, blindly reseeding from a single fresh detection
    // every time (discarding whatever positional confidence had already
    // accumulated before eviction) means the reported position effectively
    // resets to raw single-detection noise on every reactivation, which
    // happens far more often than a genuinely new landmark is created.
    // Deliberately only the POSITION is blended this way, not Pll/Pcross
    // (the covariance/cross-correlation this landmark enters the joint
    // state with) -- those stay derived purely from the fresh detection's
    // Jacobians, same as the no-prior case, since the prior carries no
    // cross-correlation information to fuse in and a from-scratch blend
    // of two covariance representations is a materially bigger, riskier
    // change than the confirmed symptom (position jitter, not a
    // covariance-validity problem) calls for.
    void AddLandmark(double _measuredBodyX, double _measuredBodyY,
                      ConeColor _color, double _stddev,
                      const LandmarkEstimate *_priorEstimate = nullptr);
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
    // Discards any retired landmark sitting within kDuplicatePruneRadius of
    // a SAME-COLOR active landmark. Needed because CorrectOrAddLandmark's
    // matching order (active search first, unconditionally returning on a
    // hit -- see step 1/2 there) means a retired entry only ever gets a
    // chance to be reconsidered when NO active match exists for a fresh
    // detection; if an active landmark representing the same physical cone
    // already exists (however that duplicate arose -- e.g. an earlier
    // detection matched a slightly-off active candidate instead of this
    // one's retired copy), every future detection of that cone keeps
    // matching the active one, and the retired twin is never looked at
    // again -- confirmed directly: a retired entry sat completely frozen,
    // unchanged, right next to a continuously-updating active landmark for
    // 27+ seconds straight in a live test. This is separate cleanup, not
    // folded into the matching gate itself, specifically because it must
    // run regardless of whether a NEW detection ever triggers a retired
    // search at all. kDuplicatePruneRadius is a plain Euclidean check (no
    // live covariance available for a retired-vs-active landmark pair, and
    // none needed): comfortably under the >=5m real cone spacing this file
    // already relies on elsewhere, so it can only catch genuine duplicates
    // of the same physical cone, never two distinct real ones.
    //
    // Throttled to every 20th call (see the .cpp) -- cost is O(retired *
    // active), and retired count isn't bounded the way active is, so on a
    // long-running drive this became a confirmed, real regression
    // (/timing/localization climbing to 1.5-3.7ms as the retired list grew
    // into the hundreds over several minutes).
    void PruneStaleRetiredDuplicates();
    // Same idea as PruneStaleRetiredDuplicates, but for TWO ACTIVE
    // landmarks of the same color sitting on top of each other -- confirmed
    // as a real, SEPARATE failure mode: raising kMaxActiveLandmarks (to
    // reduce eviction churn and let landmarks actually converge -- see the
    // jitter-debugging story elsewhere in this file) made duplicate counts
    // get WORSE, not better, even though individual jitter dropped. Once
    // two active entries for the same physical cone both exist (however
    // that happened -- e.g. two near-simultaneous detections in an early,
    // still-uncorrelated frame that didn't gate against each other), NEW
    // detections always just match whichever ONE has the lower Mahalanobis
    // distance -- the other sits there getting no further corrections,
    // starved, and previously would eventually get evicted (forcing it
    // through the retired-list path, where PruneStaleRetiredDuplicates or
    // a later reactivation could catch it) -- but with a large enough
    // active cap, eviction may never happen at all, so nothing EVER cleans
    // it up. When a duplicate active pair is found, keeps whichever was
    // seen more recently (m_landmarkLastSeen) -- the same recency signal
    // EvictStaleIfOverCapacity already uses elsewhere -- and removes the
    // other via RemoveActiveLandmark.
    //
    // Also throttled to every 20th call (see the .cpp) -- O(active^2),
    // bounded by kMaxActiveLandmarks unlike the retired one above, but
    // still a measurable avoidable cost run on every single correction.
    void PruneStaleActiveDuplicates();

    // Packs the grid cell containing world position (_x, _y) into a single
    // key for m_retiredGrid -- see that member's comment for the grid this
    // indexes into and the coverage guarantee it relies on.
    static int64_t RetiredGridCellKey(double _x, double _y);
    // Adds m_retiredLandmarks[_index] to its grid cell's bucket. Called
    // once when a landmark is newly retired (EvictStaleIfOverCapacity) and
    // once when EraseRetiredLandmark relocates the former last element into
    // a freed slot.
    void RetiredGridInsert(size_t _index);
    // Removes m_retiredLandmarks[_index] via swap-with-last-element +
    // pop_back (O(1)) instead of .erase() (O(retired), and -- worse for
    // the grid specifically -- silently invalidates every index the grid
    // has stored for elements originally past _index, since .erase() shifts
    // them all down by one without the grid ever finding out). Also
    // updates the grid: removes _index from its own bucket, and if a
    // relocation happened, repoints the moved (formerly-last) element's
    // grid entry from its old index to _index.
    void EraseRetiredLandmark(size_t _index);
    // Gathers indices into m_retiredLandmarks for every entry within the
    // 3x3 grid-cell neighborhood around world position (_x, _y) -- see
    // m_retiredGrid's comment for the coverage guarantee. Candidates are
    // NOT pre-filtered by color or further distance here; callers still
    // apply the real Mahalanobis/variance gate exactly as before -- this
    // only narrows which retired landmarks are even considered, the same
    // role the active-search loop's own coarse Euclidean pre-filter plays.
    void RetiredGridQuery(double _x, double _y, std::vector<size_t> &_candidates) const;

    Eigen::VectorXd m_x;
    Eigen::MatrixXd m_P;
    // m_landmarkColors[i] / m_landmarkLastSeen[i] correspond to state
    // indices kVehicleStateDim + 2*i (x) and kVehicleStateDim + 2*i + 1
    // (y) -- ACTIVE landmarks only. m_landmarkLastSeen[i] is the m_tick
    // value at the most recent match or add for that landmark.
    std::vector<ConeColor> m_landmarkColors;
    std::vector<uint64_t> m_landmarkLastSeen;
    // Parallel to m_landmarkColors -- see LandmarkEstimate::uid's comment
    // for why this exists. Assigned once (m_nextLandmarkUid++) the first
    // time a landmark is genuinely newly created, then carried forward
    // through every eviction/reactivation via LandmarkEstimate::uid (see
    // AddLandmark).
    std::vector<uint64_t> m_landmarkUid;
    uint64_t m_nextLandmarkUid = 1;
    uint64_t m_tick = 0;
    // Evicted landmarks -- no longer part of the joint state (see the
    // class comment), kept only for Landmarks() reporting and for
    // CorrectOrAddLandmark's reactivation check.
    std::vector<LandmarkEstimate> m_retiredLandmarks;
    // Spatial hash grid over m_retiredLandmarks, keyed by a packed (cellX,
    // cellY) grid cell (see RetiredGridCellKey) -- see kRetiredGridCellSize
    // in ekf.cpp for the cell size and the proof that a 3x3 cell
    // neighborhood search around any query point is guaranteed not to miss
    // a valid candidate. Added to fix a confirmed O(retired) scaling
    // problem: CorrectOrAddLandmark's retired-landmark search (step 2, and
    // PruneStaleRetiredDuplicates) used to linearly scan the ENTIRE
    // m_retiredLandmarks list on every detection that didn't match an
    // active landmark, and unlike the active list (bounded by
    // kMaxActiveLandmarks), the retired list grows without bound over a
    // long drive (only the overall kMaxLandmarks backstop limits it, sized
    // for a full track's worth of cones -- hundreds). This turns that
    // search from O(retired) into O(1) average (a small, bounded number of
    // candidates per query, regardless of total retired count), the same
    // complexity class every other per-detection cost in this file is
    // already held to.
    //
    // Values are INDICES into m_retiredLandmarks, not landmark data itself
    // -- kept valid only because every removal from m_retiredLandmarks now
    // goes through EraseRetiredLandmark (swap-and-pop + grid maintenance)
    // instead of a raw .erase() (which would silently invalidate every
    // index past the erased one, without the grid ever finding out).
    std::unordered_map<int64_t, std::vector<size_t>> m_retiredGrid;
};

} // namespace fsd
