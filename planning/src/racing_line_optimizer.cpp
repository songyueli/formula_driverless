#include "racing_line_optimizer.hpp"

#include <algorithm>
#include <cmath>

namespace fsd
{
namespace
{
// Cap on how far a point's LONGITUDINAL (along-tangent) component can
// drift from its own corridor sample's fixed position -- see
// ClampToCorridor's own header comment in racing_line_optimizer.hpp for
// why this component was originally left unclamped, and this file's own
// history below for why that turned out unsafe once neighborRadius grew
// past 1.
//
// BUG FIX (2026-09-04, user report: "it's going outside the boundaries of
// the left and right corridor" -- confirmed live at a third distinct
// stuck location, same session as the gate/hairpin fixes above).
// ClampToCorridor only ever clamps the LATERAL component against
// _sample's own leftBound/rightBound -- the longitudinal component passes
// through completely unbounded, by original design ("shifting a little
// along-track is exactly what smoothing is expected to do"). That was a
// reasonable call at neighborRadius=1 (immediate-neighbor pulls stay
// small), but at neighborRadius=8 (4.0m reach) the neighbor-midpoint pull
// itself can be several meters away along-track, and nothing brings a
// point's LONGITUDINAL position back toward its own assigned sample
// across 30 iterations -- only the lateral axis is corrected every pass.
// The result: a point can converge several meters along-track from
// _sample's own position while still being clamped against THAT sample's
// leftBound/rightBound -- satisfying a corridor bound that was measured
// somewhere else on the track entirely. Wherever the corridor's real
// width changes meaningfully over that drifted distance (both the gate
// and any other locally-narrow section qualify), the point ends up
// geometrically outside the corridor's ACTUAL width at its own resulting
// position, even though it never violated the (stale) bound it was
// checked against.
//
// 1.0m is a conservative starting cap -- comfortably more than
// kSplineSampleSpacing (0.5m) so real local smoothing still has room to
// work, comfortably less than neighborRadius=8's own 4.0m reach so a
// point can no longer drift far enough for its assigned sample to go
// meaningfully stale relative to nearby real corridor-width changes. Not
// yet live-validated at a range of values -- retune from a live capture
// showing racing-line-vs-corridor-bound violations the same way this
// session's other first-attempt constants have been, if 1.0m turns out to
// still leave visible violations or feels overly conservative once this
// specific failure mode is confirmed gone.
constexpr double kMaxLongitudinalDrift = 1.0;  // meters

// Optimizer-side edge buffer (2026-09-08, user report: recurring "stuck on
// a cone" wedges at ~0.95-1.0m from the actual cone, persisting essentially
// UNCHANGED across a live kCorridorSafetyMargin raise from 0.85 to 0.95m
// that same session -- if raising the margin genuinely bought more real
// clearance, the wedge distance should have grown by roughly the same
// 0.10m; it didn't, it stayed ~0.96-0.97m both before and after. Root
// cause: the SOR clamp two lines below uses the corridor's FULL nominal
// leftBound/rightBound with zero reserved slack of its own, so whenever
// curvature-minimization wants to hug an edge (a real hairpin apex
// legitimately does), it converges to EXACTLY that boundary -- moving the
// boundary just moves where the zero-slack hug point sits, it never
// creates any actual margin, since the optimizer is happy to consume
// 100% of whatever width the corridor offers. Reserving a small buffer
// HERE, inside the optimizer's own achievable range, is what the margin
// raises were actually trying (and structurally couldn't) achieve: even
// the theoretically-optimal least-curvature line can now only reach to
// within kOptimizerEdgeBuffer of the corridor's own (already
// safety-margin-adjusted) edge, so there's real slack left over for
// whatever tracking error remains, no matter how tightly the optimizer
// itself wants to hug a genuine apex. 0.15m is a first, not yet live-
// validated value -- small relative to kSplineSampleSpacing-scale
// corridor widths (~1.0-1.5m typical on this track) so it shouldn't
// meaningfully blunt real curvature reduction, but large enough to matter
// against the ~0.95-1.0m wedge distances actually observed. Only ever
// SHRINKS the achievable range (via std::max(0.0, bound - buffer) below),
// never widens it past what the corridor's own bound already allows.
constexpr double kOptimizerEdgeBuffer = 0.15;  // meters

// INSIDE-CURVE BIAS -- tried and REVERTED (2026-09-09, same session, user
// report: "that's the opposite of what we want. it's as if the planned
// path is going even wider"). Root cause of the reversal, not just a
// guess: a real racing line's own optimal shape is an S-curve -- yStar's
// sign legitimately FLIPS between corner entry (curvature-minimization
// pulls toward the OUTSIDE, to set up a wide, smooth arc) and the apex
// (pulls toward the INSIDE, to cut the corner). A bias that amplifies
// whichever direction yStar CURRENTLY points, with no notion of which
// phase the sample is in, reinforces the entry-phase outward pull just as
// much as the apex-phase inward one -- confirmed live as making the
// entry-phase widening MORE pronounced, the opposite of the intended
// effect, not a subtle side effect. Do not re-add a same-sign-as-yStar
// bias again; a real fix needs to distinguish entry from apex (e.g. by
// the local curvature's own sign relative to the corridor's net
// direction, or by biasing toward the corridor's OWN measured apex
// location) rather than blindly following yStar's instantaneous sign.

// DISTANCE PENALTY (2026-09-09, user request: "the racing line should be a
// combination of least curvature but also least distance" -- after the
// same-sign bias attempt above was confirmed to make things worse, not a
// fresh guess in a different direction). This is a proper joint
// optimization, not a heuristic: adds lambda*y_i^2 (squared lateral
// deviation from this sample's own fixed centerline point) to the
// existing curvature-cost objective -- a standard Tikhonov/ridge
// regularization, and a well-established proxy for "extra path length"
// in this literature (for a corridor whose fixed samples already sit at
// roughly even arc-length spacing along a reference curve, deviating
// laterally by y adds path length on the order of y^2/(2R), so penalizing
// y^2 directly discourages exactly the excess distance a pure detour
// would add, without needing a separate, harder-to-linearize arc-length
// term). Re-deriving this sample's own closed-form stationary point with
// the new term included (same coordinate-descent approach as the pure-
// curvature yStar, see this function's own header comment): the y_i^2
// coefficient in the objective goes from 6 (three residual terms, each
// contributing a coefficient of 1, 4, 1 respectively -- see the derivation
// in OptimizeRacingLine's own comment) to (6 + kDistancePenaltyWeight),
// and the stationary point becomes yStar_new = (2*dotI - dotIm1 - dotIp1)
// / (6 + kDistancePenaltyWeight) -- i.e. the ORIGINAL yStar formula with
// this constant simply added to its own denominator. This uniformly
// SCALES DOWN however far the pure curvature-minimizer wants to deviate
// from centerline, symmetrically in both directions -- unlike the
// reverted same-sign bias, it can't selectively amplify one phase of the
// S-curve over the other, since it acts identically regardless of
// yStar's own sign. kDistancePenaltyWeight=0 recovers the original
// pure-curvature behavior exactly; 2.5 (roughly 6/8.5=~0.71x the
// pure-curvature deviation) is a first, moderate, not yet independently
// live-validated starting point -- large enough to meaningfully damp the
// entry-phase widening this session's own live captures confirmed was
// reaching the corridor's real edge, small enough that genuine apex
// curvature reduction (still the dominant term, 6 vs 2.5) isn't lost.
// Retune from a live measurement of both real apex clearance AND actual
// cornering speed (a too-large value defeats the entire point of
// curvature minimization -- see this file's own header comment for why
// minimum-curvature, not minimum-distance, is the literature's own
// standard racing-line objective in the first place) before raising
// further.
constexpr double kDistancePenaltyWeight = 2.5;
}  // namespace
PathPoint ClampToCorridor(const PathPoint &point, const CorridorSample &sample)
{
    const double dx = point.x - sample.point.x;
    const double dy = point.y - sample.point.y;
    const double longitudinal = dx * sample.tangentX + dy * sample.tangentY;
    const double lateral = -dx * sample.tangentY + dy * sample.tangentX;  // left-positive
    const double clampedLongitudinal = std::clamp(longitudinal, -kMaxLongitudinalDrift, kMaxLongitudinalDrift);
    // See kOptimizerEdgeBuffer's own comment above -- keeps this clamp
    // consistent with the SOR sweep's own buffered range, so a point
    // re-clamped here (e.g. after cross-cycle blending) can't drift back
    // out to the corridor's full, zero-slack nominal edge either.
    const double bufferedRight = std::max(0.0, sample.rightBound - kOptimizerEdgeBuffer);
    const double bufferedLeft = std::max(0.0, sample.leftBound - kOptimizerEdgeBuffer);
    const double clampedLateral = std::clamp(lateral, -bufferedRight, bufferedLeft);
    const double leftNormalX = -sample.tangentY;
    const double leftNormalY = sample.tangentX;

    PathPoint result;
    result.x = sample.point.x + clampedLongitudinal * sample.tangentX + clampedLateral * leftNormalX;
    result.y = sample.point.y + clampedLongitudinal * sample.tangentY + clampedLateral * leftNormalY;
    return result;
}

std::vector<PathPoint> OptimizeRacingLine(const std::vector<CorridorSample> &corridor, int sweeps,
                                           double omega, bool closed)
{
    const int n = static_cast<int>(corridor.size());
    if (n < 3)
    {
        std::vector<PathPoint> result;
        result.reserve(corridor.size());
        for (const auto &c : corridor)
        {
            result.push_back(c.point);
        }
        return result;  // nothing with an "interior" to solve
    }

    // Per-sample left-normal (perpendicular to tangent, left-positive) --
    // each point's ENTIRE degree of freedom is a scalar offset along this
    // fixed direction, from this sample's own fixed point. See this
    // function's own header comment for why that (not a free 2D candidate
    // position projected/clamped after the fact) is the structural fix.
    std::vector<double> normalX(static_cast<size_t>(n));
    std::vector<double> normalY(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        normalX[static_cast<size_t>(i)] = -corridor[static_cast<size_t>(i)].tangentY;
        normalY[static_cast<size_t>(i)] = corridor[static_cast<size_t>(i)].tangentX;
    }

    std::vector<double> y(static_cast<size_t>(n), 0.0);  // lateral offset per sample, starts at centerline

    auto pointAt = [&](int i) -> PathPoint
    {
        const auto &s = corridor[static_cast<size_t>(i)];
        const double yi = y[static_cast<size_t>(i)];
        return PathPoint{s.point.x + yi * normalX[static_cast<size_t>(i)],
                          s.point.y + yi * normalY[static_cast<size_t>(i)]};
    };

    // Open: index 0 and n-1 stay fixed at y=0 (their own corridor
    // centerline) -- same anchor contract the old code had. Closed: every
    // index is solved, neighbors wrap modulo n.
    const int startIdx = closed ? 0 : 1;
    const int endIdx = closed ? n : n - 1;  // exclusive

    for (int sweep = 0; sweep < sweeps; ++sweep)
    {
        for (int i = startIdx; i < endIdx; ++i)
        {
            int im1, ip1, im2, ip2;
            if (closed)
            {
                im1 = (i - 1 + n) % n;
                ip1 = (i + 1) % n;
                im2 = (i - 2 + n) % n;
                ip2 = (i + 2) % n;
            }
            else
            {
                // Mirrors the Python prototype's open-path guard: only
                // solve where a full +/-2 neighborhood exists, leaving
                // points right next to the fixed anchors untouched by this
                // pass (they still start at y=0, same as the anchors).
                if (i - 1 < 1 || i + 1 > n - 2)
                {
                    continue;
                }
                im1 = i - 1;
                ip1 = i + 1;
                im2 = std::max(0, i - 2);
                ip2 = std::min(n - 1, i + 2);
            }

            const PathPoint pim2 = pointAt(im2);
            const PathPoint pim1 = pointAt(im1);
            const PathPoint &pi = corridor[static_cast<size_t>(i)].point;  // this sample's own FIXED point
            const PathPoint pip1 = pointAt(ip1);
            const PathPoint pip2 = pointAt(ip2);

            // Closed-form stationary point of this sample's own
            // contribution to the total curvature cost sum_i |P_{i-1} -
            // 2*P_i + P_{i+1}|^2, holding every OTHER point fixed at its
            // current value (standard coordinate-descent derivation for a
            // quadratic: differentiate w.r.t. y_i, set to zero, solve).
            // See this function's own header comment for the full
            // derivation reference.
            const double bim1X = pim2.x - 2.0 * pim1.x + pi.x;
            const double bim1Y = pim2.y - 2.0 * pim1.y + pi.y;
            const double biX = pim1.x - 2.0 * pi.x + pip1.x;
            const double biY = pim1.y - 2.0 * pi.y + pip1.y;
            const double bip1X = pi.x - 2.0 * pip1.x + pip2.x;
            const double bip1Y = pi.y - 2.0 * pip1.y + pip2.y;

            const double nx = normalX[static_cast<size_t>(i)];
            const double ny = normalY[static_cast<size_t>(i)];
            const double dotIm1 = bim1X * nx + bim1Y * ny;
            const double dotI = biX * nx + biY * ny;
            const double dotIp1 = bip1X * nx + bip1Y * ny;

            // Denominator is (6 + kDistancePenaltyWeight), not the pure-
            // curvature 6, per the Tikhonov/ridge derivation in this
            // file's own comment above kDistancePenaltyWeight's
            // declaration.
            const double yStar =
                (2.0 * dotI - dotIm1 - dotIp1) / (6.0 + kDistancePenaltyWeight);

            // SOR: move partway (scaled by omega) from the current value
            // toward the exact coordinate-wise minimizer, rather than
            // jumping straight there -- see this function's own header
            // comment for why plain Gauss-Seidel (omega=1) converges too
            // slowly for this to be practical at closed-loop scale.
            const double yCur = y[static_cast<size_t>(i)];
            const double yNew = yCur + omega * (yStar - yCur);

            const auto &sample = corridor[static_cast<size_t>(i)];
            // See kOptimizerEdgeBuffer's own comment above.
            const double bufferedRight = std::max(0.0, sample.rightBound - kOptimizerEdgeBuffer);
            const double bufferedLeft = std::max(0.0, sample.leftBound - kOptimizerEdgeBuffer);
            y[static_cast<size_t>(i)] = std::clamp(yNew, -bufferedRight, bufferedLeft);
        }
    }

    std::vector<PathPoint> result;
    result.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        result.push_back(pointAt(i));
    }
    return result;
}
}  // namespace fsd
