#pragma once

// Detects when the vehicle has completed one full lap of the track, so
// planning.cpp can switch the landmark-based racing-line pipeline from its
// windowed (forward-facing, local) query to a full-track CLOSED-LOOP one
// (2026-08-31 user request: once the whole track is mapped, the racing
// line / corridor bounds / midpoints / spline should visibly close the
// loop instead of only ever showing a local window around the car).
//
// Leave-then-return detection, GATED on actual cumulative distance
// traveled -- not leave/return proximity alone. Record the vehicle's world
// position the first time this is updated (the start line, approximately
// -- there's no explicit start/finish gate in this sim, so "wherever the
// car was when the landmark pipeline first went ready" is used as a
// proxy). Accumulate the path length actually driven (sum of consecutive-
// update displacements) every call. The lap is declared complete only once
// BOTH: (a) accumulated distance exceeds kMinLapDistance, AND (b) current
// position is back within kReturnRadius of the start point.
//
// Distance-traveled is the PRIMARY gate, not a secondary one -- confirmed
// directly (2026-08-31) as necessary, not defensive: an earlier version of
// this class used only a small leave-radius (15m) before allowing a return
// check, on a track whose real scale spans 100+ meters. The car reliably
// got back within a small return radius of its own start point within the
// first minute of driving (a hairpin or return leg sitting close to the
// start line, not an actual full lap), declaring "lap complete" while the
// accumulated landmark map still only covered a small partial arc of
// track. Feeding that sparse partial arc into planning's closed-loop
// spline (which wraps its first and last points together) produced a
// bogus shortcut straight through the track interior, corrupting the
// corridor/racing line badly enough that downstream safety clamps
// stripped every waypoint -- published an empty path and left the car
// stuck. kMinLapDistance requires the car to have actually covered
// meaningful ground before a return-to-start proximity hit is trusted.
//
// One-way latch, same convention as planning.cpp's own
// landmarkPipelineActive: once a lap completes, stays completed --  a
// momentary GPS-noise-driven dip back near the start mid-lap (unlikely at
// kReturnRadius's scale, but not impossible near a track section that
// happens to pass close to the start) shouldn't un-complete it.
namespace fsd
{
class LapDetector
{
public:
    // Call every cycle a fresh vehicle world position is available (same
    // cadence as the landmark pipeline's own recompute -- see
    // planning.cpp's onEstimatedLandmarks).
    void Update(double _worldX, double _worldY);

    bool LapComplete() const { return m_lapComplete; }

    // Diagnostic accessor -- lets a caller (planning.cpp's own log/debug
    // output) confirm how much ground was actually covered before
    // trusting a "lap complete" result, without needing a separate topic.
    double DistanceTraveled() const { return m_distanceTraveled; }

private:
    bool m_haveStart = false;
    double m_startX = 0.0;
    double m_startY = 0.0;
    bool m_haveLast = false;
    double m_lastX = 0.0;
    double m_lastY = 0.0;
    double m_distanceTraveled = 0.0;
    bool m_lapComplete = false;
};
}  // namespace fsd
