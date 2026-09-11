#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

#include <gz/transport/Node.hh>
#include <gz/msgs/clock.pb.h>
#include <gz/msgs/double.pb.h>
#include <gz/msgs/double_v.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/uint64.pb.h>

#include <common/frame_transform.hpp>
#include <common/scoped_timer.hpp>
#include <common/types.hpp>

#include "centerline_extractor.hpp"
#include "corridor.hpp"
#include "landmark_map.hpp"
#include "lap_detector.hpp"
#include "path_generator.hpp"
#include "path_utils.hpp"
#include "racing_line_cache.hpp"
#include "racing_line_optimizer.hpp"
#include "spline.hpp"
#include "track_boundaries.hpp"

// Path planning process
// ----------------------
// TWO parallel pipelines, gated by a one-way readiness latch:
//
//   1. REACTIVE (original, unmodified): every /cone_detections message
//      (body frame, one camera cycle) recomputes a short centerline path
//      from scratch using only cones visible in that single instant -- no
//      memory, no localization dependency. See path_generator.hpp/
//      track_boundaries.hpp. Serves as the cold-start fallback: it's what
//      publishes /planned_path until the landmark pipeline below has
//      enough data to take over, and keeps running (cheaply) afterward so
//      it's never left stale, matching this project's established
//      "swappable-controller"-style convention of keeping a simpler prior
//      implementation available (see pure_pursuit_controller.hpp) rather
//      than deleting it.
//
//   2. LANDMARK-BASED RACING LINE (new): draws from the EKF's ACCUMULATED,
//      world-frame landmark map (/estimated_landmarks, /estimated_pose --
//      both already published by localization.cpp) instead of one
//      instant's detections. Confirmed live as the fix for the reactive
//      pipeline's own real failure mode: a path built from only what's
//      visible RIGHT NOW has no memory of cones seen a moment ago and now
//      out of view, which was causing the car to visibly run off the
//      actual track. This is a DELIBERATE break from the reactive
//      pipeline's "decoupled from localization quality" principle above --
//      an accepted tradeoff: from here on, localization drift can surface
//      as apparent planning bugs.
//
//      Stages (each independently testable, own file):
//        a. landmark_map.hpp    -- windowed query of the landmark cache
//        b. centerline_extractor.hpp -- ordered midpoints (swappable, see
//           its own FUTURE EXTENSION POINT comment for Delaunay)
//        c. spline.hpp          -- centripetal Catmull-Rom, densely sampled
//        d. corridor.hpp        -- per-sample track-width bound
//        e. racing_line_optimizer.hpp -- least-curvature line within that
//           corridor (generalizes path_generator.cpp's MinimizeCurvature)
//      Recomputes on /estimated_landmarks arrival (perception-cycle rate,
//      ~9-10Hz), NOT /estimated_pose (up to ~50Hz) -- re-running this
//      whole pipeline every pose tick would waste CPU recomputing against
//      landmark data that hasn't actually changed. The single world->body
//      conversion uses one pose snapshot taken at the top of that
//      callback, so it stays internally consistent even though pose
//      updates faster than landmarks do.
//
//      Readiness latch (one-way: once true, stays true -- a momentary dip
//      in nearby landmark count shouldn't flap back to the reactive
//      pipeline mid-drive): the window around the vehicle needs at least 2
//      blue + 2 yellow landmarks and a pose to have arrived at least once.
//
// Inputs (subscribe):
//   /cone_detections    gz.msgs.Pose_V  (body frame -- reactive pipeline)
//   /estimated_pose     gz.msgs.Pose    (world frame -- landmark pipeline)
//   /estimated_landmarks gz.msgs.Pose_V (world frame, color in name() --
//                        landmark pipeline; recompute trigger)
//
// Output (publish):
//   /planned_path       gz.msgs.Pose_V  (ordered waypoints, BODY frame,
//                        nearest-ahead first, position only -- unchanged
//                        shape regardless of which pipeline produced it,
//                        so control.cpp needs no changes)

namespace
{
const fsd::BoundaryExtractorFn kActiveBoundaryExtractor = fsd::ColorSplitBoundaries;
const fsd::PathGeneratorFn kActivePathGenerator = fsd::NearestPairMidpointPath;
const fsd::MidpointExtractorFn kActiveMidpointExtractor = fsd::TwoPointMidpointExtractor;

// Last-line-of-defense filter, applied to the open pipeline's final body-
// frame output right before publish: drops any waypoint that ends up
// BEHIND the vehicle (negative body-frame x) -- confirmed directly
// (2026-08-31) as a real, user-reported symptom ("the planned line gets
// drawn behind the car"). Rather than chase down every possible upstream
// cause across two pipelines and several stages each (a spline's phantom-
// endpoint extrapolation, a smoothing pass's pull, a stale pose used for
// the world->body conversion mid-turn, etc. -- any of which COULD
// occasionally produce one), this guarantees the invariant unconditionally
// at the one place both pipelines' output has to pass through anyway. A
// small negative tolerance (not a hard x>=0) absorbs ordinary floating-
// point noise around the origin without discarding a legitimately-just-
// ahead point.
//
// WIDENED -0.05 -> -3.0 (2026-09-09, user report: /planned_path "still
// gets shorter and shorter, then pops back up by luck" even after the
// all-negative safeguard above was added). Root cause: this near-zero
// tolerance was never given the SAME fix already applied to the
// structurally IDENTICAL problem in landmark_map.cpp's own
// kBehindMargin -- a genuinely-ahead point can read as negative body-
// frame x purely from the vehicle's heading lagging the track's local
// curvature mid-turn (see kBehindMargin's own comment for the full
// mechanism and its own confirmed live case), not because the car
// actually passed it. kBehindMargin was widened from -3.0 to -8.0 for
// this exact reason; this constant was left at a near-zero cutoff the
// whole time, so the landmark stage generously kept forward data
// (confirmed live: /planning/debug_racing_line stayed healthy throughout)
// while THIS filter stripped it right back out on every cycle the
// heading lag was large enough -- the fluctuating lag crossing this
// razor-thin threshold cycle to cycle is exactly the "shrinks, then
// recovers by luck" pattern reported. -3.0 (not the full -8.0) is a
// deliberately smaller step: pure_pursuit_controller.cpp's own reactive
// target search is gated on TOTAL distance, not body-frame x sign, so a
// kept point that's both negative-x AND farther than the lookahead
// distance could in principle get picked as the steering target -- -3.0
// keeps that risk small (this array's own leading points are the
// closest-along-track ones, typically well under the 2.0m lookahead
// before any sign ambiguity matters) while still meaningfully covering
// ordinary heading-lag effects. Retest specifically for a "car aims at a
// point behind it" symptom before widening further toward -8.0.
constexpr double kBehindCarTolerance = -3.0;  // meters

// PREFIX-ONLY removal, not a global filter (2026-09-05, user report: the
// car stalling at the hairpin/tight turns traced to /planned_path getting
// truncated -- confirmed the mechanism: this used to be a global
// std::remove_if, stripping EVERY negative-x point anywhere in the array,
// not just a leading run of already-passed ones. That's fine on a straight
// or gentle curve, where "behind the car" and "negative body-frame x" are
// the same thing everywhere in the array -- but at a sharp/hairpin turn,
// if the vehicle's own heading hasn't caught up with the track's local
// curvature yet (mid-turn, nose still pointed closer to the OLD direction
// than the tighter curve ahead), a genuinely-forward-along-the-track
// stretch of the racing line can read as negative-x in the car's CURRENT
// heading frame purely from that heading lag, not because the car actually
// passed it. The old global filter stripped that whole stretch out
// wherever it fell in the array, which can gut the published path down to
// just the short leading run that happened to still read as positive-x --
// exactly the "reaches the end of /planned_path and stops" symptom
// reported live. Restricting removal to a genuine LEADING PREFIX (stop at
// the first point that's already positive-x, keep everything from there
// onward regardless of what any LATER point's own x sign is) preserves the
// original intent (still trims genuinely-just-passed points at the very
// start of the array) without discarding real, still-ahead track data
// later in the sequence just because of a transient heading/curvature
// mismatch. Only ever applied to the open pipeline's own short, forward-
// windowed output (never the closed loop, which has its own reason not to
// use this filter at all -- see that call site's own comment) -- that
// window doesn't loop back near the car the way a closed loop's seam does,
// so a prefix-only rule doesn't need the same "far side of the track, not
// actually passed" caveat the closed-loop case has.
// ALL-NEGATIVE SAFEGUARD (2026-09-09, user report + direct live capture:
// "it just did the same thing... stopped because there's no planned path
// at all" -- confirmed live via a 250-cycle side-by-side trace,
// /planning/debug_racing_line steady at 11-13 points / ~27.7m reach the
// ENTIRE time while /planned_path sat at EXACTLY 0 points, the whole time,
// starting within the first few cycles of the capture). Root cause: this
// function's own 2026-09-05 fix (the prefix-only restriction above)
// already correctly diagnosed the mechanism -- a heading/curvature
// mismatch can make genuinely-ahead points read as negative-x -- but only
// handled the case where SOME prefix of the array is affected. When the
// mismatch is severe enough that EVERY point reads negative-x (confirmed
// live: firstAhead walked all the way to the array's own end), the
// prefix-walk still empties the whole array, exactly the "reaches the end
// of /planned_path and stops" symptom this function was originally
// written to fix, just via the one case its own 2026-09-05 comment didn't
// yet cover. Matches this session's own established principle (see
// EnforceMinTurnRadius's matching 2026-09-09 fix in path_utils.cpp): the
// published path should have the same reach as the debug racing line --
// there's no reason to publish nothing when world-frame data this good
// still exists. If trimming would remove the ENTIRE array, don't trim at
// all -- publishing a few genuinely-already-passed points at the front
// (pure pursuit's own lookahead search just walks past them to the first
// point far enough away, see its own header comment) is a far smaller
// cost than publishing nothing and stopping the car dead.
std::vector<fsd::PathPoint> RemoveBehindCarPoints(std::vector<fsd::PathPoint> _waypoints)
{
    size_t firstAhead = 0;
    while (firstAhead < _waypoints.size() && _waypoints[firstAhead].x < kBehindCarTolerance)
    {
        ++firstAhead;
    }
    if (firstAhead >= _waypoints.size())
    {
        return _waypoints;  // every point read as "behind" -- don't empty the array, see comment above
    }
    _waypoints.erase(_waypoints.begin(), _waypoints.begin() + static_cast<long>(firstAhead));
    return _waypoints;
}

// Landmark-pipeline tuning constants. Flagged, same as this project's
// other first-attempt constants: not yet tuned against a second
// independent live measurement -- reasonable starting points, expect to
// retune from real driving, not to have guessed harder up front.
constexpr double kWindowRadius = 20.0;          // meters
constexpr double kSplineSampleSpacing = 0.5;    // meters of arc length
// Was fsd::kMinCarClearance directly (1.35m) from 2026-09-02 through
// 2026-09-03 -- that DID fix a real bug (see the history this comment used
// to carry, preserved in git blame: a smaller 0.3m corridor margin let the
// optimizer route the racing line somewhere EnforceMinTurnRadius's own
// zero-cone-awareness clamp couldn't safely follow, a confirmed live wedge
// crash). But 1.35m itself is the WRONG DIMENSION for this use -- confirmed
// live (2026-09-03, user report: "debug left and debug right corridor...
// not consistent near the hairpin, and they go wide by a lot"): a live
// /planning/debug_corridor_left vs _right snapshot measured a mean corridor
// width of just 0.41m on this track's own measured 3.00m real width, with
// occasional spikes (e.g. 4.00m right at the hairpin) that only LOOK
// dramatic because the baseline they're swinging against is a near-zero
// sliver -- not a hairpin-specific bug, a track-wide one the hairpin's
// tighter chain-tracking just exposes hardest. Root cause: kMinCarClearance
// is explicitly derived from the car's HALF-LENGTH (0.9m, see its own
// comment in path_utils.cpp -- sized for EnforceMinClearance's job of
// pushing a waypoint away from a cone the car might drive straight INTO,
// nose-on) -- but a corridor sample's blue/yellow distance is measured
// PERPENDICULAR TO THE LOCAL TANGENT, i.e. genuinely LATERAL, where the
// car's HALF-WIDTH (0.42m, same comment) is the physically correct
// dimension, not half-length. Using the longer dimension here reserves
// roughly 2x the margin actually needed on EACH side independently (2x
// kMinCarClearance = 2.70m taken from a 3.00m track), which is why the
// corridor came out pinned to a sliver almost everywhere.
// kCorridorLateralClearance is a NEW, separate constant using the correct
// dimension: 0.42m (half-width) + 0.1425m (largest real cone's own base
// radius, same value kMinCarClearance's own derivation uses) + a
// comparable proportional tracking-error buffer to the one that pushed
// kMinCarClearance itself from 1.2 to 1.35 (~13%) -- rounds to 0.65m.
// kMinCarClearance ITSELF is left unchanged at 1.35m (still correct for
// EnforceMinClearance's own nose-on, direction-agnostic use, downstream of
// this corridor).
//
// RESIDUAL RISK, not fully closed by this change: EnforceMinClearance still
// runs AFTER the corridor-constrained racing line is built, and still
// enforces the full 1.35m EUCLIDEAN radius against every cone regardless of
// approach direction -- so a racing-line point placed near this new,
// smaller 0.65m corridor edge can still get pushed further in by
// EnforceMinClearance afterward. This is a smaller, more bounded version of
// the ORIGINAL 2026-09-02 wedge bug (which happened at a 0.3m corridor
// margin, a ~1.05m gap to close; this leaves only a ~0.7m gap), not a fully
// eliminated risk -- needs live validation specifically at the hairpin and
// any other tight section, watching for the same wedge signature (a
// published point matching EnforceMinTurnRadius's clamp boundary exactly).
// Raised 0.65 -> 0.85m (2026-09-04, user report: "we crashed into a cone
// again... decrease the width of the corridor" -- right after
// EnforceMinClearance was removed from the corridor-based racing-line path
// entirely, see that removal's own comment above). This constant is now the
// SOLE clearance guarantee on that path (there is no more downstream push
// to lean on), so the margin needs to cover tracking error + cone radius on
// its own rather than being a belt to EnforceMinClearance's suspenders.
// +0.20m leaves 3.00 - 2*0.85 = 1.30m of drivable width on this track's
// nominal 3.00m gates -- still comfortably more than the car's own 0.84m
// physical width -- while meaningfully increasing the buffer against the
// exact failure just reported. kCorridorMinHalfWidth's own 0.5m floor is
// unchanged, so an already-tight section is no more constrained than
// before; this only tightens sections that had margin to spare.
// RAISED 0.85 -> 0.95 (2026-09-05, user goal: 10 laps, zero stuck events --
// confirmed live, the car STILL wedged near a cone at ~0.96-0.97m even
// after corridor.cpp's own safety-floor fix that same night, which
// addressed a real, confirmed bug (the floor claiming more space than
// truly available) but evidently wasn't the whole story -- a racing line
// that legitimately, intentionally hugs an apex at exactly this margin's
// own distance leaves very little room for any additional real-world
// tracking error before actual contact. +0.10m leaves 3.00 - 2*0.95 =
// 1.10m of drivable width on this track's nominal 3.00m gates -- still
// more than the car's own 0.84m physical width, but with less slack than
// before, so this can't grow indefinitely without conflicting with
// kCorridorMinHalfWidth's own 0.5m floor (1.0m combined) in genuinely
// narrow sections -- if wedging persists at this value, the next lever is
// tracking fidelity (control-side), not squeezing this further.
constexpr double kCorridorLateralClearance = 0.95;  // meters
constexpr double kCorridorSafetyMargin = kCorridorLateralClearance;  // meters
constexpr double kCorridorMinHalfWidth = 0.5;   // meters
constexpr double kCorridorMaxHalfWidth = 2.0;   // meters

// Dynamic closed-loop corridor margin (2026-09-04, user request: "make the
// track width dynamic, between a min and max value... first lap would be
// the min width, and then as the confidence of the cone locations increase
// lap after lap... we can increase the left and right widths accordingly,
// up to the max"). Applies ONLY to the CLOSED-loop corridor build -- the
// full-map recompute done at every lap completion (see the closedLoop
// block below, now re-run every lap rather than once) -- not to
// kCorridorLateralClearance above, which stays fixed for the open/fallback
// pipeline: that pipeline is inherently a "map still forming" case with no
// accumulated confidence signal yet to lean on.
//
// kConfidenceObsCountConservative/Optimistic bound the INPUT: the mean
// per-landmark OBSERVATION COUNT across the whole discovered map, published
// by localization.cpp's landmarksConfidencePub (see its own comment for the
// full derivation). NOT variance/stddev-based, despite that being the
// first, more obvious-looking choice -- confirmed LIVE, immediately, that
// mean position stddev is floor-saturated in this EKF (captured at
// ~0.540-0.542m repeatedly within the first several seconds of a run and
// never moved, since kLandmarkVarianceFloor clamps nearly every landmark to
// the same value on its very first correction, not gradually -- see
// localization.cpp's own comment at the actual computation for the full
// explanation). obsCount has no such floor and genuinely accumulates with
// repeated observation.
//
// RELATIVE to lap 1's own measured baseline, NOT a fixed absolute number
// (2026-09-04, same day, changed before this ever shipped): a first attempt
// hardcoded conservative=3.0/optimistic=20.0 as guesses, but a live check
// (before even one lap had completed) already showed the whole-map mean at
// 19.4 within ~15s and 47.2 by ~40s -- both already past the guessed
// "optimistic" ceiling, meaning the corridor would reach max width almost
// immediately, defeating "lap 1 = min width" entirely. The real growth
// rate depends on THIS track's own size, cone density, and driving speed
// (all things that also change between runs/tracks), so any fixed absolute
// number is fragile the same way -- instead, kFirstLapMeanObsCount (below)
// captures whatever the mean genuinely IS the moment lap 1's own line is
// first computed, and that becomes the CONSERVATIVE anchor by construction
// -- lap 1 always reads confidence==its own baseline (t=0, exactly
// kCorridorMarginConservative), regardless of this track's specific
// numbers. kConfidenceGrowthMultiplier=3.0 (OPTIMISTIC = 3x the lap-1
// baseline) is the one still-first-guess, not-yet-live-validated part of
// this mapping -- retune from an actual multi-lap /estimated_landmarks_
// confidence trace (this file's own per-lap log line already reports the
// mean at every recompute) if the corridor still widens too fast/slow
// relative to how many laps a real race actually runs.
//
// kCorridorMarginConservative/Aggressive bound the OUTPUT: Conservative
// matches kCorridorLateralClearance's own current value -- used at lap 1
// itself, so lap 1's own margin is UNCHANGED from right before this
// feature, not a fresh regression risk on the lap that matters most (least
// map confidence, first time through every corner).
//
// Aggressive TEMPORARILY PINNED EQUAL TO CONSERVATIVE (2026-09-05, user
// goal: 10 laps, zero stuck events). This feature's whole point is to
// let the margin shrink (corridor widen) on later laps -- but the SAME
// night this was built, wedging was confirmed live at kCorridorLateralClearance
// =0.85m, i.e. AT-OR-ABOVE the old 0.55m aggressive value, meaning 0.55m
// is now known to be less safe than a value already shown to wedge, not
// more. Letting the margin narrow further while the underlying wedge
// mechanism is still being run down would work directly against tonight's
// reliability goal. Pinning both to the same (currently 0.95m) value
// disables the narrowing behavior WITHOUT removing the feature -- once
// kCorridorLateralClearance is confirmed to reliably avoid wedging across
// many clean laps, re-derive a genuinely lower, live-validated aggressive
// value rather than restoring 0.55m blindly.
constexpr double kConfidenceGrowthMultiplier = 3.0;  // optimistic = this many x the lap-1 baseline
constexpr double kCorridorMarginConservative = kCorridorLateralClearance;  // meters
constexpr double kCorridorMarginAggressive = kCorridorLateralClearance;    // meters (see comment above)

double MarginForConfidence(double _meanObsCount, double _lap1BaselineMeanObsCount)
{
    // Floored at 1.0 (obsCount's own minimum, "initial add counts as 1") --
    // guards the divide below for the pathological case of a lap 1 that
    // somehow discovered zero landmarks, which would otherwise be a 0/0.
    const double baseline = std::max(_lap1BaselineMeanObsCount, 1.0);
    const double t = std::clamp(
        (_meanObsCount / baseline - 1.0) / (kConfidenceGrowthMultiplier - 1.0), 0.0, 1.0);
    return kCorridorMarginConservative + t * (kCorridorMarginAggressive - kCorridorMarginConservative);
}

// RACING LINE ALGORITHM REPLACED 2026-09-04 (user report: "it doesn't
// appear to be properly stretching the curve to the left and right debug
// corridors, which is the whole point of those corridors"). The previous
// Laplacian neighbor-pull approach went through FOUR distinct live-tuning
// cycles the same day (radius=1 too timid, ~5.7% corridor-width
// utilization on average -> wider radius fixed the hairpin but drifted
// unwarranted on a straight -> clamping the drift fixed THAT but produced
// a confirmed-jagged, non-smooth line -> reverted entirely) without ever
// landing on something both smooth and corridor-respecting -- see
// racing_line_optimizer.hpp's own header comment for the full postmortem
// and why a heuristic neighbor-pull-then-clamp was the wrong tool for this
// job. OptimizeRacingLine is now a proper minimum-curvature QP (per-sample
// scalar lateral offset, box-constrained, solved via SOR) -- see that same
// header comment for the algorithm itself and its own offline validation
// numbers (zero corridor violations across the full captured closed loop,
// heading-direction sign flips back down near the old timid baseline's
// own smoothness while corridor-width utilization climbed well past it).
//
// kClosedRacingLineSweeps/kClosedRacingLineOmega: this runs ONCE per lap
// completion, so cost is essentially free -- 2000 sweeps is what the
// offline validation was actually run at (see racing_line_optimizer.hpp),
// not a smaller/faster-but-less-converged guess.
constexpr int kClosedRacingLineSweeps = 2000;
constexpr double kClosedRacingLineOmega = 1.9;
// kOpenRacingLineSweeps/kOpenRacingLineOmega: the open/windowed pipeline
// recomputes fresh every cycle (unlike the closed loop's one-time solve),
// so this needs to fit inside a real per-cycle budget.
//
// LOWERED omega 1.9 -> 1.0 (2026-09-04, user report: "look at how
// zigzagged it is now... it's much more warped than the debug lines").
// omega=1.9 was carried over from the closed-loop value UNVALIDATED at
// this scale (the offline validation that picked 1.9 only checked zero-
// violations and utilization for the open case, never a smoothness/
// heading-change metric the way the closed-loop case's own jaggedness
// investigation did) -- confirmed live as a real problem, not a guess: a
// captured /planned_path showed genuine heading-direction reversals over
// 100 degrees, TWICE in a row, with the underlying corridor confirmed
// smooth at a nearby capture, pointing at the solver itself rather than
// noisy input data. SOR's textbook 0<omega<2 convergence guarantee is for
// the UNCONSTRAINED problem -- once a large over-relaxed step overshoots
// past a sample's own corridor bound and gets clipped every sweep (far
// more likely on this pipeline's smaller, less pre-smoothed per-cycle
// corridor than the closed loop's own heavily-smoothed 560-sample one),
// that guarantee doesn't cleanly carry over, and the repeated overshoot-
// then-clip cycle is a plausible direct mechanism for the observed
// oscillation. 1.0 is plain Gauss-Seidel -- slower per-sweep progress, but
// monotonically convergent even through constraint clipping, which is what
// actually matters here. kOpenRacingLineSweeps raised 200->600 to
// compensate for the slower per-sweep rate -- there is enormous headroom
// for this: this pipeline's own /timing/planning was measured at
// 235-267us per cycle at 200 sweeps/omega=1.9, a tiny fraction of the
// ~70-75ms cycle budget, so 3x the sweep count at a cheaper (no
// overshoot-driven extra iterations to converge past) omega is still
// nowhere close to a real cost concern. Needs live validation specifically
// for the zigzag symptom being gone, not just re-confirming timing.
//
// RAISED 600 -> 3000 (2026-09-09, user report: "at the hairpin, we're not
// drawing a planned path that starts wide and goes along the inner part
// of the turn... clip[ping] the outer cone"). Confirmed live, precisely:
// a captured /planning/debug_racing_line through this exact hairpin
// measured its OWN lateral offset from the raw spline centerline at
// 0.000-0.03m through the entire tight section (only growing past the
// apex, well after the part that matters), while the corridor there
// measured ~1.0-1.2m wide -- meaning ~0.35-0.45m of real, unused box room
// on each side (kOptimizerEdgeBuffer already subtracted). That's not a
// hard constraint being respected, it's under-convergence: unlike the
// closed loop (which solves ONCE per lap and can afford to fully
// converge), this open/pre-lap window restarts from y=0 EVERY cycle (see
// OptimizeRacingLine's own `y(n, 0.0)` init) with a brand-new corridor,
// and 600 sweeps of plain Gauss-Seidel evidently isn't enough to move
// meaningfully away from that cold start before the array gets thrown
// away and rebuilt next cycle -- effectively publishing raw centerline,
// never a real racing line, for the entire pre-lap-completion phase.
// 3000 is a large step, not a timid one, matching the same "there is
// enormous headroom" reasoning as the 200->600 raise above (a few more
// milliseconds at most, still nowhere near the ~70-75ms cycle budget) --
// deliberately closer to the closed loop's own validated 2000 than to the
// stale 600, since this pipeline's cold-restart-every-cycle handicap
// argues for MORE sweeps than the closed loop needs, not fewer. Needs
// live validation specifically for real apex-hugging appearing at the
// hairpin (nonzero, meaningfully-sized offset_from_center through the
// tight section), not just re-confirming timing.
constexpr int kOpenRacingLineSweeps = 3000;
constexpr double kOpenRacingLineOmega = 1.0;

fsd::ConeColor ConeColorFromName(const std::string &_name)
{
    if (_name == "blue") return fsd::ConeColor::Blue;
    if (_name == "yellow") return fsd::ConeColor::Yellow;
    // "orange" and "large_orange" are two DISTINCT classes perception's
    // own model outputs (see perception.cpp's kClassNames) -- the FSAE
    // start/finish gate uses large orange cones specifically, a real,
    // regularly-detected class, not a hypothetical one. BUG FIX
    // (2026-09-01): "large_orange" had no case here, so every such cone
    // fell through to ConeColor::Unknown, which LandmarkMap's own query
    // functions silently drop entirely (see their switch statements'
    // `default: break;`) -- confirmed directly: this track's only two
    // orange cones (the start/finish gate, per
    // simulation/worlds/trackdrive.sdf) sit right at the recurring
    // trouble spot several closed-loop pipeline bugs were traced back to
    // this session, and were completely invisible to planning the whole
    // time -- zero clearance protection, no influence on anything.
    // Treated identically to "orange" here: neither is a centerline
    // color, both matter only for clearance, and nothing downstream needs
    // to distinguish the two sizes.
    if (_name == "orange" || _name == "large_orange") return fsd::ConeColor::Orange;
    return fsd::ConeColor::Unknown;
}
}  // namespace

int main()
{
    gz::transport::Node node;

    auto pathPub = node.Advertise<gz::msgs::Pose_V>("/planned_path");
    auto timingPub = node.Advertise<gz::msgs::UInt64>("/timing/planning");

    // Debug topics for the landmark-based pipeline's own intermediate
    // stages -- all WORLD frame (unlike /planned_path, which is body
    // frame), one per stage, so each can be inspected directly in
    // Foxglove instead of only trusting the final published path. Added
    // specifically to debug this pipeline's own live behavior (a real
    // forward/backward-window bug already found and fixed this way, plus
    // a reported oscillation still being chased down) -- not meant to be
    // permanent instrumentation the way /timing/planning is; fine to
    // remove once this pipeline is trusted, same spirit as the FOXGLOVE
    // BRIDGE's own TEMPORARY debug landmark topic.
    auto debugMidpointsPub = node.Advertise<gz::msgs::Pose_V>("/planning/debug_midpoints");
    auto debugSplinePub = node.Advertise<gz::msgs::Pose_V>("/planning/debug_spline");
    auto debugCorridorLeftPub = node.Advertise<gz::msgs::Pose_V>("/planning/debug_corridor_left");
    auto debugCorridorRightPub = node.Advertise<gz::msgs::Pose_V>("/planning/debug_corridor_right");
    auto debugRacingLinePub = node.Advertise<gz::msgs::Pose_V>("/planning/debug_racing_line");

    // Lap timing (2026-09-03 user request, deferred until after the
    // crossover fix -- see the closed-loop pipeline's own history above for
    // why: no point timing laps against a path that wasn't a real closed
    // loop yet). "Live elapsed time" + "first lap time", scoped per the
    // user's own choice. Sim time (not wall-clock) via /world/trackdrive/
    // clock, gz.msgs.Clock -- same source and .sim().sec()/.sim().nsec()
    // pattern foxglove_bridge.cpp's own onClock already uses, so this stays
    // correct under pause/step/non-realtime playback the way a wall-clock
    // std::chrono timer wouldn't. World name hardcoded ("trackdrive") --
    // this file's main() takes no argv today (unlike foxglove_bridge.cpp,
    // which gets it from dev_sim.sh's own $WORLD_NAME), and every launch
    // this project has ever used is trackdrive; revisit if that changes.
    auto lapElapsedPub = node.Advertise<gz::msgs::Double>("/lap_time/elapsed");
    auto lapFirstLapPub = node.Advertise<gz::msgs::Double>("/lap_time/first_lap");
    // /lap_time/last_lap (2026-09-04, user request: "modify the lap
    // counter so that it includes a last lap time") -- the MOST RECENTLY
    // COMPLETED lap's own duration, updating every time a new lap
    // finishes, not just the first. Backed by a SEPARATE LapDetector
    // instance (lapTimer, declared alongside the existing pipeline-switch
    // lapDetector below) that gets explicitly Reset() after each
    // completion it observes -- the existing lapDetector must stay a
    // one-way latch (it drives the closed-loop pipeline switch, a genuine
    // one-time event), so it can't be reused for repeating per-lap timing
    // without breaking that.
    auto lapLastLapPub = node.Advertise<gz::msgs::Double>("/lap_time/last_lap");
    // /lap_time/laps (2026-09-04, user request: "post the lap times as a
    // fixed size array of length 10, since laps never exceed 10") -- every
    // completed lap's own duration, index 0 = lap 1, kept in one place so
    // a Foxglove table/plot can bind to it without resizing. Fixed at 10
    // slots always (not grown to match however many laps have actually
    // completed) -- slots for laps that haven't happened yet stay 0.0,
    // same "unset means 0" convention firstLapTimeSeconds/lastLapTimeSeconds
    // already use before their own first completion. gz::msgs::Double_V
    // (repeated double) rather than 10 separate scalar topics -- this is
    // genuinely one array-shaped value, not 10 independent ones.
    constexpr size_t kMaxTrackedLaps = 10;
    auto lapTimesPub = node.Advertise<gz::msgs::Double_V>("/lap_time/laps");
    std::array<std::atomic<double>, kMaxTrackedLaps> lapTimes{};
    std::atomic<size_t> lapCount{0};
    std::atomic<bool> haveRaceStart{false};
    std::atomic<uint64_t> raceStartSimTimeNs{0};
    std::atomic<uint64_t> lastSimTimeNs{0};
    // Set once, the first cycle the closed loop completes (see the
    // lapJustCompletedForRecompute block below, the same moment
    // storedClosedRacingLine etc. are FIRST computed -- that block now
    // re-runs every lap, see its own comment, but this flag's own
    // haveFirstLapTime guard still only ever fires once) -- read back here
    // every clock tick so /lap_time/first_lap keeps
    // republishing the same value continuously rather than a single
    // one-shot message a not-yet-subscribed bridge could miss.
    std::atomic<bool> haveFirstLapTime{false};
    std::atomic<double> firstLapTimeSeconds{0.0};
    // Set/updated every time lapTimer (below) observes a completion --
    // same continuous-republish reasoning as haveFirstLapTime above.
    std::atomic<bool> haveLastLapTime{false};
    std::atomic<double> lastLapTimeSeconds{0.0};
    std::function<void(const gz::msgs::Clock &)> onClock =
        [&lapElapsedPub, &lapFirstLapPub, &lapLastLapPub, &lapTimesPub, &lapTimes, &haveRaceStart,
         &raceStartSimTimeNs, &lastSimTimeNs, &haveFirstLapTime, &firstLapTimeSeconds, &haveLastLapTime,
         &lastLapTimeSeconds](const gz::msgs::Clock &_msg)
    {
        const uint64_t ns = static_cast<uint64_t>(_msg.sim().sec()) * 1000000000ULL
            + static_cast<uint64_t>(_msg.sim().nsec());
        lastSimTimeNs.store(ns, std::memory_order_relaxed);
        if (!haveRaceStart.load(std::memory_order_relaxed))
        {
            raceStartSimTimeNs.store(ns, std::memory_order_relaxed);
            haveRaceStart.store(true, std::memory_order_relaxed);
        }
        const double elapsedSeconds =
            static_cast<double>(ns - raceStartSimTimeNs.load(std::memory_order_relaxed)) * 1e-9;
        gz::msgs::Double elapsedMsg;
        elapsedMsg.set_data(elapsedSeconds);
        lapElapsedPub.Publish(elapsedMsg);
        if (haveFirstLapTime.load(std::memory_order_relaxed))
        {
            gz::msgs::Double firstLapMsg;
            firstLapMsg.set_data(firstLapTimeSeconds.load(std::memory_order_relaxed));
            lapFirstLapPub.Publish(firstLapMsg);
        }
        if (haveLastLapTime.load(std::memory_order_relaxed))
        {
            gz::msgs::Double lastLapMsg;
            lastLapMsg.set_data(lastLapTimeSeconds.load(std::memory_order_relaxed));
            lapLastLapPub.Publish(lastLapMsg);
        }
        gz::msgs::Double_V lapTimesMsg;
        for (auto &slot : lapTimes)
        {
            lapTimesMsg.add_data(slot.load(std::memory_order_relaxed));
        }
        lapTimesPub.Publish(lapTimesMsg);
    };
    if (!node.Subscribe("/world/trackdrive/clock", onClock))
    {
        std::cerr << "planning: failed to subscribe to /world/trackdrive/clock -- lap timing disabled\n";
    }

    fsd::LandmarkMap landmarkMap;
    std::atomic<bool> landmarkPipelineActive{false};
    // Detects lap completion so the pipeline below can switch from a
    // windowed (local, forward-facing) query to the full-track CLOSED-LOOP
    // one -- see lap_detector.hpp's own header comment. Lives here (not
    // inside the onEstimatedLandmarks callback) so it persists its
    // leave/return state across every call, same as landmarkPipelineActive.
    fsd::LapDetector lapDetector;

    // Separate instance for REPEATING per-lap timing (2026-09-04, user
    // request: "modify the lap counter so that it includes a last lap
    // time") -- see lap_detector.hpp's own Reset() comment for why this
    // can't share lapDetector above (that one must stay a one-way latch
    // for the pipeline switch). Explicitly Reset() after each completion
    // this instance observes, in the same rising-edge block lapDetector's
    // own one-time completion is already handled in below.
    fsd::LapDetector lapTimer;
    // Elapsed-since-race-start (seconds) at the moment of the PREVIOUSLY
    // completed lap, so each new completion's own duration is computable
    // as a simple difference -- 0.0 for lap 1 (a lap's duration measured
    // from the race start IS just its own elapsed-time-at-completion).
    double previousLapCompletionSeconds = 0.0;

    // RacingLineCache/corridorLeftCache/corridorRightCache/kBlendAlpha
    // (originally added 2026-08-31 to damp cycle-to-cycle jitter in the
    // open pipeline's own recomputed-every-cycle racing line and debug
    // corridor arrays) REMOVED ENTIRELY 2026-09-03 -- see the open
    // pipeline's own racing-line block below (where the blend used to be
    // applied) for the full reasoning: the same 0.5m grid-cell-collision
    // mechanism already found and removed for the closed-loop case (this
    // comment block's own next paragraph, unchanged below) was ALSO live in
    // the open pipeline the whole time, and got directly implicated in a
    // real stuck-car report ("stuck, ghst cone") via a captured
    // /planned_path showing the same scrambled-ordering signature. No
    // replacement jitter-damping added here -- see the open pipeline's own
    // comment for what a real fix would need instead (an index-keyed or
    // radius-searched cache, not exact grid bucketing).
    //
    // Closed-loop /planned_path cross-cycle blending -- ADDED 2026-09-02,
    // REMOVED 2026-09-03. Two implementations were tried and both actively
    // corrupted the output rather than smoothing it:
    //
    // 1. RacingLineCache, spatially keyed by each point's world position --
    // confirmed live to collide, since its grid cell size (0.5m) exactly
    // matched this closed-loop line's own point spacing, hashing adjacent-
    // but-genuinely-different points into the same cell.
    //
    // 2. A direct per-stored-index cache (fixed the collision, since
    // storedClosedRacingLine has permanently stable indices once computed)
    // -- but confirmed live to have a DIFFERENT, worse problem: the source
    // array is completely static once computed (byte-identical across
    // cycles, verified directly), so there was never any real jitter to
    // smooth in the first place. The cache's only actual effect was a
    // failure mode -- if any stored index was ever cached with a bad value
    // (e.g. from a cycle where an upstream stage hadn't fully settled),
    // every later cycle revisiting that index would blend toward that
    // stale value indefinitely, since the blended output was itself written
    // back into the cache. Confirmed directly: traced every published
    // /planned_path point back to its nearest stored-line index and found
    // the tail of the window jumping backward to indices already used
    // EARLIER in the same message (e.g. output position 58 landing on the
    // same world position as position 36's index) -- not drift, a cache
    // returning wrong data.
    //
    // No replacement needed: publishing the window directly, with no
    // cross-cycle blend at all, is correct given the source is already
    // static -- see this file's own diagnostic dump confirming a max
    // consecutive-point gap of well under 1m across the entire stored line.

    // Closed-loop pipeline state, RECOMPUTED once per lap completion (not
    // every ~10Hz cycle) -- see this file's 2026-08-31 postmortem comment
    // at the closedLoop branch below for why per-CYCLE recompute is unsafe
    // (that reasoning is about live, momentarily-noisy vehicle pose being
    // re-decided 10-14 times a SECOND, an entirely different timescale from
    // "once every time the car crosses the start/finish line," a handful
    // of times per race at most -- see lapJustCompletedForRecompute's own
    // comment below for the 2026-09-04 change from a one-way latch to a
    // per-lap trigger).
    std::vector<fsd::PathPoint> storedClosedMidpoints;
    std::vector<fsd::PathPoint> storedClosedSpline;
    std::vector<fsd::PathPoint> storedClosedCorridorLeft;
    std::vector<fsd::PathPoint> storedClosedCorridorRight;
    std::vector<fsd::PathPoint> storedClosedRacingLine;
    // Persistent "where on the loop is the car right now" cursor -- see
    // the per-cycle nearest-point lookup below for why this can't be a
    // fresh global nearest-distance scan every cycle. Initialized to 0
    // when the loop is first computed (the vehicle IS at/near index 0 at
    // that moment, by construction of ClosedLoopMidpointExtractor's own
    // ordering cursor).
    size_t closedLoopCursor = 0;

    std::function<void(const gz::msgs::Pose_V &)> onConeDetections =
        [&pathPub, &timingPub, &landmarkPipelineActive](const gz::msgs::Pose_V &_msg)
    {
        // Still computed every cycle even once the landmark pipeline takes
        // over publishing -- see this file's header comment for why: an
        // always-warm fallback, never left stale.
        fsd::ScopedTimer timer([&timingPub, &landmarkPipelineActive](int64_t _us)
        {
            if (landmarkPipelineActive.load(std::memory_order_relaxed))
            {
                return;  // don't double-publish /timing/planning against the landmark pipeline's own
            }
            gz::msgs::UInt64 msg;
            msg.set_data(static_cast<uint64_t>(_us));
            timingPub.Publish(msg);
        });

        std::vector<fsd::ClassifiedCone> cones;
        cones.reserve(static_cast<size_t>(_msg.pose_size()));
        for (const auto &pose : _msg.pose())
        {
            cones.push_back(fsd::ClassifiedCone{pose.position().x(), pose.position().y(), pose.name()});
        }

        const fsd::TrackBoundaries boundaries = kActiveBoundaryExtractor(cones);
        const std::vector<fsd::PathPoint> waypoints = RemoveBehindCarPoints(kActivePathGenerator(boundaries));

        if (landmarkPipelineActive.load(std::memory_order_relaxed))
        {
            return;  // landmark pipeline is now the sole publisher
        }

        gz::msgs::Pose_V pathMsg;
        for (const auto &wp : waypoints)
        {
            gz::msgs::Pose *p = pathMsg.add_pose();
            p->mutable_position()->set_x(wp.x);
            p->mutable_position()->set_y(wp.y);
        }
        pathPub.Publish(pathMsg);
    };
    if (!node.Subscribe("/cone_detections", onConeDetections))
    {
        std::cerr << "Failed to subscribe to /cone_detections\n";
        return 1;
    }

    std::function<void(const gz::msgs::Pose &)> onEstimatedPose =
        [&landmarkMap](const gz::msgs::Pose &_msg)
    {
        const double yaw = fsd::YawFromPlanarQuaternion(_msg.orientation().z(), _msg.orientation().w());
        landmarkMap.UpdatePose(fsd::Pose2D{_msg.position().x(), _msg.position().y(), yaw});
    };
    if (!node.Subscribe("/estimated_pose", onEstimatedPose))
    {
        std::cerr << "Failed to subscribe to /estimated_pose\n";
        return 1;
    }

    // Map-wide landmark confidence (2026-09-04, user request: "make the
    // track width dynamic, between a min and max value... as the
    // confidence of the cone locations increase lap after lap... increase
    // the left and right widths accordingly"). See localization.cpp's
    // landmarksConfidencePub for exactly what this scalar is (mean
    // per-landmark OBSERVATION COUNT across the whole discovered map,
    // HIGHER = more confident -- not position variance, see that
    // publisher's own comment for why variance was tried first and
    // confirmed live not to work here) and MarginForConfidence's own
    // comment for how this is used RELATIVE to a lap-1 baseline, not
    // against a fixed absolute number. Defaults to 0.0 -- harmless even if
    // a closed-loop recompute somehow ran before this topic's first real
    // message ever arrived, since MarginForConfidence's own baseline
    // capture (see the recompute site) would then also capture 0.0 as
    // lap 1's baseline, and the very next real message updates this to a
    // genuine value before the SECOND recompute (lap 2) ever reads it.
    std::atomic<double> landmarkMeanObsCount{0.0};
    std::function<void(const gz::msgs::Double &)> onLandmarksConfidence =
        [&landmarkMeanObsCount](const gz::msgs::Double &_msg)
    { landmarkMeanObsCount.store(_msg.data(), std::memory_order_relaxed); };
    if (!node.Subscribe("/estimated_landmarks_confidence", onLandmarksConfidence))
    {
        std::cerr << "Failed to subscribe to /estimated_landmarks_confidence\n";
        return 1;
    }
    // Captured ONCE, at the very first closed-loop recompute (lap 1) -- see
    // MarginForConfidence's own comment for why this run's own lap-1 value,
    // not a fixed absolute number, is the right conservative anchor. Same
    // one-way-latch convention as haveFirstLapTime/firstLapTimeSeconds
    // above.
    std::atomic<bool> haveFirstLapMeanObsCount{false};
    std::atomic<double> firstLapMeanObsCount{0.0};

    std::function<void(const gz::msgs::Pose_V &)> onEstimatedLandmarks =
        [&landmarkMap, &landmarkPipelineActive, &lapDetector, &lapTimer, &previousLapCompletionSeconds,
         &pathPub, &timingPub, &debugMidpointsPub,
         &debugSplinePub, &debugCorridorLeftPub, &debugCorridorRightPub, &debugRacingLinePub,
         &storedClosedMidpoints, &storedClosedSpline, &storedClosedCorridorLeft,
         &storedClosedCorridorRight, &storedClosedRacingLine, &closedLoopCursor,
         &haveRaceStart, &raceStartSimTimeNs, &lastSimTimeNs, &haveFirstLapTime,
         &firstLapTimeSeconds, &haveLastLapTime, &lastLapTimeSeconds, &lapTimes, &lapCount,
         &landmarkMeanObsCount, &haveFirstLapMeanObsCount, &firstLapMeanObsCount,
         kMaxTrackedLaps](const gz::msgs::Pose_V &_msg)
    {
        std::vector<fsd::WorldCone> landmarks;
        landmarks.reserve(static_cast<size_t>(_msg.pose_size()));
        for (const auto &pose : _msg.pose())
        {
            landmarks.push_back(fsd::WorldCone{
                pose.position().x(), pose.position().y(), ConeColorFromName(pose.name())});
        }
        landmarkMap.UpdateLandmarks(std::move(landmarks));

        const fsd::LandmarkMap::WindowResult window = landmarkMap.QueryWindow(kWindowRadius);
        const bool ready = window.poseValid && window.blue.size() >= 2 && window.yellow.size() >= 2;
        if (!ready && !landmarkPipelineActive.load(std::memory_order_relaxed))
        {
            return;  // not ready yet -- reactive pipeline stays the sole publisher
        }
        if (ready)
        {
            landmarkPipelineActive.store(true, std::memory_order_relaxed);  // one-way latch
        }

        fsd::ScopedTimer timer([&timingPub](int64_t _us)
        {
            gz::msgs::UInt64 msg;
            msg.set_data(static_cast<uint64_t>(_us));
            timingPub.Publish(msg);
        });

        // window.poseValid is guaranteed true here: either `ready` required
        // it directly, or landmarkPipelineActive was already latched true
        // by an earlier cycle where it was (poseValid itself never resets
        // to false once set -- see LandmarkMap::UpdatePose).
        lapDetector.Update(window.vehiclePose.x, window.vehiclePose.y);

        // Repeating per-lap timing (2026-09-04) -- independent of the
        // pipeline-switch block below, since this needs to keep firing for
        // every lap, not just the first. Update() is a no-op once
        // m_lapComplete is set (see lap_detector.hpp's own one-way-latch
        // comment), so this correctly does nothing between the moment
        // lapTimer completes and the Reset() call a few lines down runs --
        // there's no window where a stale "complete" could be double-
        // counted.
        lapTimer.Update(window.vehiclePose.x, window.vehiclePose.y);
        // Captured BEFORE the timing block's own lapTimer.Reset() below --
        // see lapJustCompletedForRecompute's own use further down (2026-09-
        // 04) for why this needs to survive past that Reset() call, unlike
        // the timing logic's own lapTimer.LapComplete() check just below,
        // which is fine reading it fresh since it runs first.
        const bool lapJustCompletedForRecompute = lapTimer.LapComplete();
        if (lapTimer.LapComplete() && haveRaceStart.load(std::memory_order_relaxed))
        {
            const uint64_t ns = lastSimTimeNs.load(std::memory_order_relaxed);
            const uint64_t startNs = raceStartSimTimeNs.load(std::memory_order_relaxed);
            const double nowElapsed = static_cast<double>(ns - startNs) * 1e-9;
            const double thisLapDuration = nowElapsed - previousLapCompletionSeconds;
            lastLapTimeSeconds.store(thisLapDuration, std::memory_order_relaxed);
            haveLastLapTime.store(true, std::memory_order_relaxed);
            previousLapCompletionSeconds = nowElapsed;
            // Store into the fixed 10-slot array (2026-09-04) -- see
            // lapTimes/lapCount's own declaration comment above. Silently
            // stops recording past the 10th lap rather than wrapping or
            // growing -- "laps never exceed 10" was stated as a known
            // constraint of this track/race format, not something to
            // defensively handle beyond simply not writing out of bounds.
            const size_t idx = lapCount.load(std::memory_order_relaxed);
            if (idx < kMaxTrackedLaps)
            {
                lapTimes[idx].store(thisLapDuration, std::memory_order_relaxed);
                lapCount.store(idx + 1, std::memory_order_relaxed);
            }
            lapTimer.Reset();
        }

        // Once a lap completes, switch from the windowed (local, forward-
        // facing) landmark set to every landmark ever seen (QueryAll(),
        // which legitimately does return the FULL discovered map --
        // confirmed directly in ekf.cpp's Landmarks(): active landmarks
        // plus m_retiredLandmarks are already merged before ever reaching
        // /estimated_landmarks). `window` itself (fetched above) stays the
        // windowed query regardless -- it's still what lapDetector and the
        // clearance safety-net below use, since clearance is an inherently
        // LOCAL concept.
        //
        // ARCHITECTURE: the closed-loop midpoints/spline/corridor/racing-
        // line are recomputed once EVERY LAP (lapJustCompletedForRecompute,
        // 2026-09-04 -- was a one-way latch, computed exactly once ever,
        // before the dynamic-corridor-width feature below needed a way to
        // apply an improved confidence-based margin on lap 2, 3, etc.), not
        // every cycle -- re-deciding the whole track's topology from
        // scratch off live, momentarily-noisy vehicle pose every ~10Hz
        // cycle was the root source of most stuck-car bugs found here
        // (2026-08-31/09-01), but that risk is about a 10-14Hz timescale,
        // not a "once every lap" one -- a lap takes many seconds, and the
        // vehicle pose at a lap boundary is no noisier than at any other
        // moment this same recompute already had to tolerate the one time
        // it used to run. Every cycle BETWEEN recomputes still looks up the
        // nearest point on the CURRENTLY stored line to the vehicle's
        // CURRENT position (see the lookup below) and walks forward from
        // there -- "map once [per lap], localize within it", the same
        // split every SLAM-adjacent system uses, just with "once" now
        // meaning "once per lap" instead of "once ever".
        //
        // STATUS (2026-09-01, end of a long multi-session debugging
        // effort -- see chat history for the full blow-by-blow): SEVEN
        // real, distinct bugs were found and fixed in this pipeline,
        // each confirmed via live testing and/or a direct dump of the
        // actual frozen array: (1) premature lap detection (fixed via
        // distance-gating in lap_detector.cpp), (2) global mutual-NN
        // midpoint pairing not scaling to full-track candidate counts
        // (fixed via ClosedLoopMidpointExtractor's per-color-chain
        // approach), (3) a chain hop-distance cap silently truncating
        // the track (fixed by raising centerline_extractor.cpp's
        // kClosedChainMaxHop, confirmed run-to-run variable), (4) the
        // midpoint chain's walk direction being undecidable without a
        // heading reference (fixed by ClosedLoopMidpointExtractor's own
        // reverse+rotate check), (5) a global nearest-distance lookup
        // jumping to an unrelated branch of the loop wherever the track
        // passes close to itself (fixed via a persistent cursor + local
        // window), (6) that same window unconditionally wrapping across
        // the array's own seam right at the ambiguous switchover point
        // (fixed by clamping wraparound until the cursor is legitimately
        // past the loop's midpoint), (7) the window's nearest-by-distance
        // choice not actually leading to real forward progress in body
        // frame near a recurring hard-turn area close to the switchover
        // point (fixed by directly checking a point well into each
        // candidate's own forward continuation, with progressive window
        // widening when the immediate neighborhood has no usable
        // candidate at all).
        //
        // With all seven fixes in place, the pipeline reliably produced
        // long, clean autonomous laps (400+m of continuous driving in
        // the best validation run, including successfully continuing
        // through the seam into a second lap) with only rare, self-
        // recovering thin-path cycles (down from constant failure). The
        // LAST thing observed, twice, was the car ending up physically
        // WEDGED (frozen, normal chassis height, no active collision
        // signature) after a mid-severity bump (a brief chassis-height
        // spike) somewhere later in a long run -- diagnostics showed the
        // published path was healthy at the time (not empty/thin), so
        // this was a SEPARATE issue from everything above, in
        // pure_pursuit_controller.cpp's own stuck-detection watchdog, not
        // a planning-data defect: the watchdog correctly detected zero
        // progress but reset its own cycle counter to throttle repeated
        // log spam, which ALSO let the very next cycle fall through to
        // full normal driving again -- the car oscillated between one
        // stop cycle and ~90 cycles of normal commands forever, instead
        // of actually holding position. Fixed 2026-09-01 with a one-way
        // m_permanentlyStuck latch (pure_pursuit_controller.hpp/.cpp) --
        // confirmed live (233 consecutive zero /cmd_ackermann commands
        // during a reproduced wedge, vs. continuous pulsing before).
        //
        // Re-enabled (2026-09-01) now that both the planning-side bugs
        // and the controller-side wedge-handling bug are fixed and
        // validated. A genuine physical wedge event can still happen
        // occasionally (a separate question of how often the car
        // contacts something, not addressed by either fix) -- but the
        // car now safely holds position when it does, rather than
        // unpredictably resuming full-speed commands into whatever it's
        // stuck against.
        const bool closedLoop = lapDetector.LapComplete();
        const fsd::LandmarkMap::WindowResult activeSet = closedLoop ? landmarkMap.QueryAll() : window;

        if (closedLoop && lapJustCompletedForRecompute)
        {
            // Lap timing: on the very FIRST lap this is the exact rising
            // edge for /lap_time/first_lap; on every lap after that this
            // whole outer block re-runs (see its own comment above) but
            // this specific inner check is already a no-op via its own
            // haveFirstLapTime guard, so /lap_time/first_lap still only
            // ever gets set once, correctly. Only captured if a clock tick
            // has actually arrived yet (haveRaceStart) -- true in practice
            // well before a lap finishes, this just avoids a bogus 0.0s
            // reading in the pathological case of a lap completing before
            // this process's very first /world/trackdrive/clock message.
            if (haveRaceStart.load(std::memory_order_relaxed) && !haveFirstLapTime.load(std::memory_order_relaxed))
            {
                const uint64_t ns = lastSimTimeNs.load(std::memory_order_relaxed);
                const uint64_t startNs = raceStartSimTimeNs.load(std::memory_order_relaxed);
                firstLapTimeSeconds.store(static_cast<double>(ns - startNs) * 1e-9, std::memory_order_relaxed);
                haveFirstLapTime.store(true, std::memory_order_relaxed);
            }
            storedClosedMidpoints =
                fsd::ClosedLoopMidpointExtractor(activeSet.blue, activeSet.yellow, activeSet.orange, activeSet.vehiclePose);
            storedClosedSpline = fsd::FitAndSampleClosedSpline(storedClosedMidpoints, kSplineSampleSpacing);
            // Dynamic margin (2026-09-04) -- see MarginForConfidence's own
            // comment for the full derivation and why this is relative to
            // THIS run's own lap-1 baseline, not a fixed absolute number.
            // Baseline captured ONCE, right here, the first time this block
            // ever runs (lap 1) -- current mean read fresh on every
            // recompute (including this same one), so a later lap's line
            // genuinely gets to use whatever the map's confidence has
            // improved to relative to that fixed lap-1 anchor.
            const double currentMeanObsCount = landmarkMeanObsCount.load(std::memory_order_relaxed);
            if (!haveFirstLapMeanObsCount.load(std::memory_order_relaxed))
            {
                firstLapMeanObsCount.store(currentMeanObsCount, std::memory_order_relaxed);
                haveFirstLapMeanObsCount.store(true, std::memory_order_relaxed);
            }
            const double dynamicCorridorMargin =
                MarginForConfidence(currentMeanObsCount, firstLapMeanObsCount.load(std::memory_order_relaxed));
            const std::vector<fsd::CorridorSample> corridor = fsd::ComputeCorridor(
                storedClosedSpline, activeSet.blue, activeSet.yellow, activeSet.orange, dynamicCorridorMargin,
                kCorridorMinHalfWidth, kCorridorMaxHalfWidth, /*closed=*/true);
            storedClosedRacingLine = fsd::OptimizeRacingLine(corridor, kClosedRacingLineSweeps,
                                                              kClosedRacingLineOmega, /*closed=*/true);
            storedClosedCorridorLeft.clear();
            storedClosedCorridorRight.clear();
            storedClosedCorridorLeft.reserve(corridor.size());
            storedClosedCorridorRight.reserve(corridor.size());
            for (const auto &c : corridor)
            {
                const double leftNormalX = -c.tangentY;
                const double leftNormalY = c.tangentX;
                storedClosedCorridorLeft.push_back(
                    fsd::PathPoint{c.point.x + c.leftBound * leftNormalX, c.point.y + c.leftBound * leftNormalY});
                storedClosedCorridorRight.push_back(fsd::PathPoint{c.point.x - c.rightBound * leftNormalX,
                                                                     c.point.y - c.rightBound * leftNormalY});
            }
            closedLoopCursor = 0;
            // Max consecutive gap in the stored racing line -- diagnostic
            // only, printed once here rather than requiring another
            // offline-harness round-trip if this recurs. A large value
            // means some stretch of the loop is sparse/discontinuous
            // (missed cone detections, not a code bug) -- see
            // centerline_extractor.cpp's kClosedChainMaxHop comment for
            // the confirmed real variability here run to run.
            double maxGap = 0.0;
            for (size_t i = 1; i < storedClosedRacingLine.size(); ++i)
            {
                const double dx = storedClosedRacingLine[i].x - storedClosedRacingLine[i - 1].x;
                const double dy = storedClosedRacingLine[i].y - storedClosedRacingLine[i - 1].y;
                maxGap = std::max(maxGap, std::sqrt(dx * dx + dy * dy));
            }
            // lapCount already reflects THIS just-completed lap (incremented
            // by the timing block above, same cycle) -- more meaningful
            // here than lapDetector.DistanceTraveled(), which is frozen at
            // lap 1's own value from lap 2 onward (lapDetector is a
            // one-way latch, see closedLoop's own declaration; Update()
            // becomes a no-op once complete).
            std::cerr << "planning: lap " << lapCount.load(std::memory_order_relaxed)
                      << " complete -- recomputed closed-loop pipeline (blue=" << activeSet.blue.size()
                      << " yellow=" << activeSet.yellow.size() << " orange=" << activeSet.orange.size()
                      << ", racing line points=" << storedClosedRacingLine.size()
                      << ", max consecutive gap=" << maxGap << "m, mean landmark obsCount="
                      << currentMeanObsCount << " (lap-1 baseline="
                      << firstLapMeanObsCount.load(std::memory_order_relaxed)
                      << "), dynamic corridor margin=" << dynamicCorridorMargin << "m)\n";
            // Dump the ACTUAL frozen array to disk -- diagnostic only.
            // Confirmed necessary (2026-09-01): re-running the pipeline
            // offline against a LATER capture of /estimated_landmarks
            // does NOT reproduce the same array a live failure actually
            // froze, since landmark estimates keep changing after the
            // one-time computation -- an offline replay's own numbers
            // (line heading, corridor width) don't match what the live
            // process was actually using when it got stuck. Writing the
            // real, exact array out here removes that gap entirely.
            std::ofstream dump("/tmp/planning_closed_loop_dump.csv");
            if (!dump.is_open())
            {
                std::cerr << "planning: FAILED to open dump file, errno=" << errno << " (" << strerror(errno)
                          << ")\n";
            }
            dump << "idx,x,y,dirDeg,turnDeg,leftBound,rightBound\n";
            double prevDir = 0.0;
            bool havePrevDir = false;
            const size_t dn = storedClosedRacingLine.size();
            for (size_t i = 0; i < dn; ++i)
            {
                const fsd::PathPoint &p = storedClosedRacingLine[i];
                const fsd::PathPoint &next = storedClosedRacingLine[(i + 1) % dn];
                const double dir = std::atan2(next.y - p.y, next.x - p.x) * 180.0 / M_PI;
                double turn = 0.0;
                if (havePrevDir)
                {
                    turn = dir - prevDir;
                    while (turn > 180.0) turn -= 360.0;
                    while (turn < -180.0) turn += 360.0;
                }
                havePrevDir = true;
                prevDir = dir;
                const double lb = i < corridor.size() ? corridor[i].leftBound : -1.0;
                const double rb = i < corridor.size() ? corridor[i].rightBound : -1.0;
                dump << i << "," << p.x << "," << p.y << "," << dir << "," << turn << "," << lb << "," << rb
                     << "\n";
            }
        }

        // Open (pre-lap-completion) pipeline: unchanged, recomputed every
        // cycle from the windowed query, same as always.
        std::vector<fsd::PathPoint> openMidpoints;
        std::vector<fsd::PathPoint> openSpline;
        std::vector<fsd::CorridorSample> openCorridor;
        std::vector<fsd::PathPoint> openRacingLine;
        if (!closedLoop)
        {
            openMidpoints = kActiveMidpointExtractor(activeSet.blue, activeSet.yellow, activeSet.orange, activeSet.vehiclePose);
            openSpline = fsd::FitAndSampleSpline(openMidpoints, kSplineSampleSpacing);
            openCorridor = fsd::ComputeCorridor(openSpline, activeSet.blue, activeSet.yellow, activeSet.orange,
                                                 kCorridorSafetyMargin, kCorridorMinHalfWidth, kCorridorMaxHalfWidth,
                                                 /*closed=*/false);
            openRacingLine = fsd::OptimizeRacingLine(openCorridor, kOpenRacingLineSweeps,
                                                      kOpenRacingLineOmega, /*closed=*/false);
            // Cross-cycle blend REMOVED (2026-09-03, user report: "stuck,
            // ghst cone" -- confirmed live via direct offline analysis of a
            // captured /planned_path: the published open-pipeline array
            // showed the exact same non-monotonic, locally-scrambled point
            // ordering signature already diagnosed and fixed for the
            // CLOSED-loop pipeline on 2026-09-03 (see this file's own
            // removed-blend-cache history for the closed case) -- two array
            // positions a few indices apart landing at nearly-identical
            // world positions, consistent with RacingLineCache's grid-cell
            // key (0.5m, same as this pipeline's own point spacing)
            // colliding: a point's blend can pick up a STALE cached value
            // from a different, unrelated point that happens to hash into
            // the same or an adjacent cell. That closed-loop fix removed
            // the cache entirely because the closed source is static (byte-
            // identical cycle to cycle, so there's no real jitter to damp in
            // the first place) -- that specific justification does NOT
            // apply here (the open pipeline genuinely recomputes fresh every
            // cycle from a live, changing window, so SOME cycle-to-cycle
            // jitter is real and this cache was originally added to damp
            // it). Removed anyway: a scrambled path that gets the car
            // permanently stuck (this pipeline runs BEFORE lap completion,
            // so a stuck car here can never even reach the point where the
            // closed-loop pipeline takes over) is a strictly worse outcome
            // than visible jitter, and this collision mechanism is not a
            // theoretical risk -- it's the same one already confirmed live
            // once this session. ClampToCorridor is kept (still a real,
            // needed safety re-clamp on its own, unrelated to the cache) --
            // only the blend/cache round-trip is removed. If jitter turns
            // out to be a real problem again without it, the fix is a
            // smaller/collision-safe cache key (e.g. keyed by array index
            // instead of world position, or a proper radius search instead
            // of exact grid bucketing), not reintroducing this one as-is.
            for (size_t i = 0; i < openRacingLine.size() && i < openCorridor.size(); ++i)
            {
                openRacingLine[i] = fsd::ClampToCorridor(openRacingLine[i], openCorridor[i]);
            }
        }

        // Debug topics for every intermediate stage -- see their
        // declarations above for why. All world frame. Closed: publish the
        // STORED (frozen) values -- identical every cycle, so zero jitter
        // by construction, not just damped. Open: publish this cycle's
        // fresh values, same as always.
        auto publishWorldPath = [](gz::transport::Node::Publisher &_pub, const std::vector<fsd::PathPoint> &_pts,
                                    bool _closeLoop = false)
        {
            gz::msgs::Pose_V msg;
            for (const auto &p : _pts)
            {
                gz::msgs::Pose *pose = msg.add_pose();
                pose->mutable_position()->set_x(p.x);
                pose->mutable_position()->set_y(p.y);
            }
            // Visually close the loop for Foxglove's own LINE_STRIP
            // rendering (foxglove_bridge.cpp's generic kDebugPathTopics
            // handling), which never connects its own last point back to
            // the first -- 2026-09-02 user report: these closed-loop debug
            // topics never actually LOOKED like a closed loop despite the
            // underlying data genuinely representing one (storedClosed*
            // really does wrap via modulo indexing everywhere else in this
            // file, e.g. the cursor lookup below). Only when _closeLoop is
            // set (the STORED, closed-loop arrays) -- the open pipeline's
            // own debug paths are genuinely forward-only, and closing THEM
            // would draw a spurious segment straight across the track.
            if (_closeLoop && _pts.size() >= 3)
            {
                gz::msgs::Pose *closing = msg.add_pose();
                closing->mutable_position()->set_x(_pts.front().x);
                closing->mutable_position()->set_y(_pts.front().y);
            }
            _pub.Publish(msg);
        };
        publishWorldPath(debugMidpointsPub, closedLoop ? storedClosedMidpoints : openMidpoints, closedLoop);
        publishWorldPath(debugSplinePub, closedLoop ? storedClosedSpline : openSpline, closedLoop);
        publishWorldPath(debugRacingLinePub, closedLoop ? storedClosedRacingLine : openRacingLine, closedLoop);
        if (closedLoop)
        {
            publishWorldPath(debugCorridorLeftPub, storedClosedCorridorLeft, true);
            publishWorldPath(debugCorridorRightPub, storedClosedCorridorRight, true);
        }
        else
        {
            std::vector<fsd::PathPoint> corridorLeft, corridorRight;
            corridorLeft.reserve(openCorridor.size());
            corridorRight.reserve(openCorridor.size());
            for (const auto &c : openCorridor)
            {
                const double leftNormalX = -c.tangentY;
                const double leftNormalY = c.tangentX;
                corridorLeft.push_back(fsd::PathPoint{c.point.x + c.leftBound * leftNormalX,
                                                        c.point.y + c.leftBound * leftNormalY});
                corridorRight.push_back(fsd::PathPoint{c.point.x - c.rightBound * leftNormalX,
                                                         c.point.y - c.rightBound * leftNormalY});
            }
            // Cross-cycle blend removed here too (2026-09-03) -- same
            // grid-cell-collision mechanism as openRacingLine's own blend
            // above (see its own comment for the full reasoning), just for
            // the debug visualization arrays instead of the array that
            // actually feeds /planned_path. Debug-only and never consumed
            // by control, but a collision here would still show up as a
            // visibly wrong/jumping corridor boundary in Foxglove -- worth
            // fixing for the same reason, at no behavioral risk since
            // nothing downstream of these two arrays affects driving.
            publishWorldPath(debugCorridorLeftPub, corridorLeft);
            publishWorldPath(debugCorridorRightPub, corridorRight);
        }

        // What actually gets published to /planned_path.
        // Closed: nearest-point lookup into the STORED (already
        // correctly-oriented, never re-decided) racing line, then the
        // FULL loop published starting from that point, walking in the
        // array's own fixed index order all the way around -- no direction
        // re-decision happens here at all, which is the whole point of
        // computing the loop once (see the closedLoop comment above).
        // Published whole, not truncated to a short window (2026-09-03) --
        // see the publish site below for why that's safe.
        //
        // The CURSOR LOOKUP (finding where "nearest" is) still uses a
        // PERSISTENT CURSOR + local search window, NOT a fresh global
        // nearest-distance scan every cycle -- this is about efficiently
        // and safely finding the START of the published loop, unrelated to
        // how much of the loop gets published from there --
        // confirmed live (2026-08-31) as a real, severe bug the global-
        // scan version had: wherever a closed loop passes close to
        // itself (a start/finish area next to another section, a hairpin
        // fold, any real track has spots like this), the globally-
        // nearest-by-raw-distance index can belong to a completely
        // different, unrelated branch of the loop than the one the car
        // is actually driving on -- confirmed directly via added
        // diagnostics: nearestIdx landed at index 529 of a 532-point loop
        // (right near the array's OTHER end) while the car was still
        // only partway along its actual route from index ~0, because
        // that far-away-in-ARC-LENGTH point happened to be physically
        // close by EUCLIDEAN distance. The resulting forward slice then
        // pointed somewhere geometrically unrelated to the car's real
        // heading -- every one of the 60 points ended up behind it,
        // empty /planned_path. This is the exact same class of failure
        // path_utils.hpp's own OrderWaypointsByTraversal (hop cap) and
        // corridor.cpp's NearestChainLateralDistance (local search
        // window) already had to solve elsewhere in this file's own
        // pipeline -- global nearest-distance search is fundamentally
        // unsafe on a shape that folds back near itself; only a window
        // anchored to where the car was LAST known to be is safe. Window
        // half-width chosen generously past normal per-cycle movement
        // (kMaxSpeed=5-8m/s / ~9-10Hz cycle rate is under 1m, i.e. under
        // 2 index steps at 0.5m sample spacing) while staying far short
        // of the whole-loop jump that caused this bug.
        // Open: this cycle's freshly-computed racing line, used whole (it
        // was already scoped to the windowed/forward-facing query, so
        // never needs this kind of bounding).
        constexpr int kClosedLoopCursorWindow = 30;  // index units (~15m at 0.5m spacing)
        std::vector<fsd::PathPoint> racingLineForPublish;
        size_t diagNearestIdx = 0;
        double diagNearestDist = -1.0;
        if (closedLoop)
        {
            if (storedClosedRacingLine.empty())
            {
                racingLineForPublish.clear();
            }
            else
            {
                const int n = static_cast<int>(storedClosedRacingLine.size());
                // Wrap the window past the array's own seam (index n-1
                // back to index 0) ONLY once the cursor has legitimately
                // advanced well past the halfway point -- confirmed live
                // (2026-08-31) as necessary, not defensive: right at
                // computation time the cursor starts at 0, and index 0's
                // own physical neighborhood is genuinely, ambiguously
                // close to index n-1's neighborhood too (they're the two
                // ends of the SAME point on a closed loop -- the vehicle's
                // switchover position). An unconditionally-wrapping window
                // search at that moment can lock onto n-1 just as easily
                // as 0, and the stored line's own local direction walking
                // FROM n-1 reads as ~170-180 degrees opposite the
                // vehicle's actual heading there (confirmed directly via
                // the storedLineHeadingVsVehicleDeg diagnostic) -- a
                // forward slice from the wrong side of the seam, nearly
                // every point behind the car. Clamping (no wraparound)
                // near the start removes that ambiguity entirely: index
                // n-1 simply isn't a candidate until the cursor has
                // covered enough real distance that "wrap back toward 0"
                // can only mean "genuinely completing another lap", not
                // "confused about which side of the start line this is".
                const bool nearEndOfLoop = closedLoopCursor > static_cast<size_t>(n) / 2;
                // Prefer the closest candidate that actually leads to real
                // forward progress -- checked DIRECTLY (does the point
                // kForwardCheckSteps further along this candidate's own
                // index order land ahead of the car in BODY frame?), not
                // via a local-tangent proxy. A local-tangent-alignment
                // version of this check was tried first and confirmed
                // live (2026-09-01) as insufficient on its own: a
                // candidate can pass a generous (100 degree) tangent
                // check at its OWN point while the array still curves
                // away over the next several samples, producing a
                // forward slice that's still entirely behind the car in
                // body frame. Checking the actual body-frame position of
                // a point well INTO the forward slice is a direct test of
                // the thing that actually matters (RemoveBehindCarPoints'
                // own criterion), not an indirect proxy for it.
                const double vehicleCosYaw = std::cos(activeSet.vehiclePose.yaw);
                const double vehicleSinYaw = std::sin(activeSet.vehiclePose.yaw);
                constexpr int kForwardCheckSteps = 20;  // ~10m ahead at 0.5m spacing
                constexpr double kForwardCheckMinBodyX = 1.0;  // meters
                // Search a given [lo,hi] index range for the closest
                // candidate passing the forward-progress check; returns
                // whether one was found, its index, and its distSq.
                auto searchRange = [&](int lo, int hi, size_t &outIdx, double &outDistSq) -> bool
                {
                    bool found = false;
                    for (int off = lo; off <= hi; ++off)
                    {
                        const size_t i = static_cast<size_t>(((off % n) + n) % n);
                        const size_t ahead = (i + static_cast<size_t>(kForwardCheckSteps)) % static_cast<size_t>(n);
                        const double adx = storedClosedRacingLine[ahead].x - activeSet.vehiclePose.x;
                        const double ady = storedClosedRacingLine[ahead].y - activeSet.vehiclePose.y;
                        // Body-frame x of the "ahead" point, using the same
                        // rotation WorldToBody applies (see
                        // common/frame_transform.hpp) -- inlined here
                        // rather than calling it, since only x is needed.
                        const double aheadBodyX = adx * vehicleCosYaw + ady * vehicleSinYaw;
                        if (aheadBodyX < kForwardCheckMinBodyX)
                        {
                            continue;  // this candidate doesn't lead anywhere ahead of the car
                        }
                        const double dx = storedClosedRacingLine[i].x - activeSet.vehiclePose.x;
                        const double dy = storedClosedRacingLine[i].y - activeSet.vehiclePose.y;
                        const double distSq = dx * dx + dy * dy;
                        if (!found || distSq < outDistSq)
                        {
                            outDistSq = distSq;
                            outIdx = i;
                            found = true;
                        }
                    }
                    return found;
                };
                // Progressive widening -- confirmed live (2026-09-01) as
                // necessary: near the recurring hard-turn area right by
                // the switchover point, EVERY candidate within the normal
                // +-30 window can fail the forward-progress check (the
                // whole local neighborhood curves away from the car's
                // current heading there), and falling back to plain
                // nearest at that point just recreates the original bug
                // (confirmed directly: nearestIdx pinned at 0-1,
                // storedLineHeadingVsVehicleDeg ~157-162 degrees, empty
                // /planned_path, car stuck). Widening the search --
                // still clamped/wrap-guarded by nearEndOfLoop exactly as
                // the normal window is -- lets the lookup skip PAST a
                // locally-bad stretch to a genuinely usable point further
                // along, rather than accepting a known-bad nearest one.
                size_t nearestIdx = closedLoopCursor;
                double bestDistSq = 0.0;
                bool foundUsable = false;
                for (int window : {kClosedLoopCursorWindow, 100, 250})
                {
                    int lo, hi;
                    if (nearEndOfLoop)
                    {
                        lo = static_cast<int>(closedLoopCursor) - window;
                        hi = static_cast<int>(closedLoopCursor) + window;
                    }
                    else
                    {
                        lo = std::max(0, static_cast<int>(closedLoopCursor) - window);
                        hi = std::min(n - 1, static_cast<int>(closedLoopCursor) + window);
                    }
                    if (searchRange(lo, hi, nearestIdx, bestDistSq))
                    {
                        foundUsable = true;
                        break;
                    }
                }
                // SANITY RE-CHECK (2026-09-03 user report: "stuck" -- chassis
                // height normal, /planned_path non-empty, but confirmed live
                // via direct offline replay that the published rotation did
                // NOT start anywhere near the vehicle's true nearest point --
                // the true nearest (world-verified, no closer candidate
                // exists anywhere on the whole loop) was clearly identifiable
                // and easily passed its own forward-progress check, yet the
                // cursor had settled on something else entirely). Root cause:
                // the loop above breaks on the FIRST window width that finds
                // ANY passing candidate -- "foundUsable" only ever meant "a
                // candidate exists in THIS window," never "this is actually
                // close to the car." If closedLoopCursor has drifted enough
                // that even the narrowest window (kClosedLoopCursorWindow=30)
                // already contains SOME point that happens to pass the
                // forward check (entirely plausible on a loop -- most stretches
                // have SOME forward-facing point somewhere), the search stops
                // right there and never tries a wider window, even though
                // that "found" point can be many meters from the car's real
                // position. This is a DIFFERENT gap from the "Global resync"
                // case just below (which only fires when NO window finds
                // anything at all) -- this one fires when a window finds
                // something, just not the RIGHT something. kCursorSanityDist
                // (5.0m) is comfortably above normal per-cycle tracking
                // distance (the corridor itself averages ~1.7m half-width, so
                // a genuinely-tracking cursor's own result should rarely
                // exceed a few meters) but well under the scale of an actual
                // divergence (the 2026-09-02 report above measured a 320+
                // index gap) -- only escalates to the O(n) global check (same
                // cost concern as the comment below, but only paid when the
                // windowed result already looks suspicious, not every cycle).
                constexpr double kCursorSanityDist = 5.0;  // meters
                if (foundUsable && bestDistSq > kCursorSanityDist * kCursorSanityDist)
                {
                    size_t globalIdx = nearestIdx;
                    double globalDistSq = bestDistSq;
                    if (searchRange(0, n - 1, globalIdx, globalDistSq) && globalDistSq < bestDistSq)
                    {
                        nearestIdx = globalIdx;
                        bestDistSq = globalDistSq;
                    }
                }
                // Global resync (2026-09-02 user report): confirmed live as
                // a real, distinct failure mode from the "locally sparse
                // stretch" case the sparse-stretch fallback below already
                // handles -- the cursor can fall arbitrarily far behind the
                // car's true position (observed: cursor pinned at index
                // 192, car's true nearest index at 515, on a 552-point
                // loop -- 320+ indices apart, far past even the widest
                // window=250 search above), with NO way to recover:
                // whenever the ABOVE windowed search fails, the fallback
                // below computes a fresh LIVE path but deliberately never
                // touches closedLoopCursor (see its own comment), so the
                // gap between the frozen cursor and the car's real position
                // only ever grows, cycle after cycle, forever, once it
                // first exceeds the window. This is NOT a plain global
                // nearest-distance search (see this whole block's own
                // opening comment for why that's unsafe on a self-crossing
                // loop) -- it's the EXACT SAME forward-progress-verified
                // searchRange() used above, just given the whole array
                // (0..n-1, already self-wrapping via searchRange's own
                // modulo) instead of a window clamped/anchored to the
                // stale cursor. The forward-progress check is what actually
                // rejects fold-back candidates (confirmed live 2026-09-01,
                // see its own comment) -- the window was an efficiency/
                // extra-safety layer on top of it, not the sole source of
                // correctness, so extending the SAME check globally is safe
                // as a last resort, not a reversion to the original bug.
                // Deliberately tried only after the local widening above
                // has already exhausted itself (an O(n) full scan every
                // cycle would be wasteful when the cheap local search
                // already succeeds the vast majority of the time).
                if (!foundUsable)
                {
                    foundUsable = searchRange(0, n - 1, nearestIdx, bestDistSq);
                }
                if (!foundUsable)
                {
                    // Even the global resync above found nothing usable in
                    // the STORED array -- confirmed live (2026-09-01) as a
                    // real, recurring case, not just a theoretical one:
                    // some track sections end up genuinely sparse in the
                    // one-time snapshot (perception simply didn't detect/
                    // localize enough cones there by lap-completion time --
                    // a real data gap, not a lookup bug; see
                    // centerline_extractor.cpp's kClosedChainMaxHop
                    // comment for the same root cause elsewhere). No
                    // widening of the SAME frozen data fixes a genuine
                    // hole in it. Falling back to "plain nearest anyway"
                    // here (an earlier version of this fix) just
                    // republishes a known-bad point -- confirmed live: the
                    // car sat sweeping in place for minutes because the
                    // underlying gap can't be searched around by turning.
                    //
                    // Instead, fall back to a FRESH reactive-pipeline
                    // computation from the CURRENT windowed live cone
                    // view (`window`, always fetched above regardless of
                    // closedLoop) -- the same math the pre-lap-completion
                    // pipeline already uses, just invoked for this one
                    // cycle. This sidesteps the frozen array's gap
                    // entirely: live detections exist right now even
                    // where the ONE-TIME snapshot didn't capture enough.
                    // Doesn't touch storedClosedRacingLine or
                    // closedLoopCursor -- once the car drives past this
                    // sparse stretch, the normal stored-line lookup
                    // resumes on its own next cycle.
                    const std::vector<fsd::PathPoint> fallbackMidpoints =
                        kActiveMidpointExtractor(window.blue, window.yellow, window.orange, window.vehiclePose);
                    const std::vector<fsd::PathPoint> fallbackSpline =
                        fsd::FitAndSampleSpline(fallbackMidpoints, kSplineSampleSpacing);
                    const std::vector<fsd::CorridorSample> fallbackCorridor = fsd::ComputeCorridor(
                        fallbackSpline, window.blue, window.yellow, window.orange, kCorridorSafetyMargin,
                        kCorridorMinHalfWidth, kCorridorMaxHalfWidth, /*closed=*/false);
                    racingLineForPublish = fsd::OptimizeRacingLine(fallbackCorridor, kOpenRacingLineSweeps,
                                                                    kOpenRacingLineOmega, /*closed=*/false);
                    diagNearestIdx = closedLoopCursor;  // unchanged -- didn't move the cursor this cycle
                    diagNearestDist = -1.0;             // sentinel: fallback path was used, not a lookup
                }
                else
                {
                    closedLoopCursor = nearestIdx;
                    diagNearestIdx = nearestIdx;
                    diagNearestDist = std::sqrt(bestDistSq);
                    // Publish the FULL loop, rotated to start at nearestIdx,
                    // not a truncated window (2026-09-03, explicit user
                    // request: "/planned_path" should visually BE the closed
                    // loop, with target selection handling "pick the point
                    // in front" rather than pre-truncating the array).
                    // Safe to do: pure_pursuit_controller.cpp's own target
                    // search is already DISTANCE-gated, not count-gated (it
                    // breaks on the first point past lookaheadDistance, and
                    // the forward-preview scan just skips anything past
                    // kBrakePreviewDistance) -- searching a longer array
                    // costs a few more skipped iterations, not a behavior
                    // change. RemoveBehindCarPoints (downstream) still trims
                    // whatever ends up behind the car once converted to body
                    // frame, same as it always did for the windowed case.
                    const size_t un = storedClosedRacingLine.size();
                    racingLineForPublish.reserve(un);
                    for (size_t k = 0; k < un; ++k)
                    {
                        racingLineForPublish.push_back(storedClosedRacingLine[(nearestIdx + k) % un]);
                    }
                }
            }
        }
        else
        {
            racingLineForPublish = openRacingLine;
        }

        // Single world->body conversion, right before publish -- the whole
        // pipeline above stays in world frame throughout (see this file's
        // header comment).
        std::vector<fsd::PathPoint> racingLineBody;
        racingLineBody.reserve(racingLineForPublish.size());
        for (const auto &wp : racingLineForPublish)
        {
            const fsd::Point2D body = fsd::WorldToBody(window.vehiclePose, wp.x, wp.y);
            racingLineBody.push_back(fsd::PathPoint{body.x, body.y});
        }

        // No cross-cycle blending for the closed-loop published window --
        // the source (storedClosedRacingLine) is a frozen, static array
        // with no real jitter to smooth, and a from-scratch cache-based
        // attempt was confirmed live to corrupt the output instead. The
        // open pipeline's own racing line no longer blends either (2026-09-
        // 03, see that block's own declaration comment for why -- the same
        // corruption mechanism, confirmed live there too).

        double diagPreClampMinX = 0.0, diagPreClampMaxX = 0.0;
        if (!racingLineBody.empty())
        {
            diagPreClampMinX = diagPreClampMaxX = racingLineBody.front().x;
            for (const auto &p : racingLineBody)
            {
                diagPreClampMinX = std::min(diagPreClampMinX, p.x);
                diagPreClampMaxX = std::max(diagPreClampMaxX, p.x);
            }
        }

        // Safety-net clamp, same as the reactive pipeline's own final step
        // (path_utils.hpp) -- reused unmodified, since EnforceMinTurnRadius's
        // geometry is already correctly defined relative to the vehicle at
        // the body-frame origin. Needs ALL nearby cones (including orange,
        // which the centerline extraction itself deliberately excludes)
        // converted to body frame too, for EnforceMinTurnRadius's own
        // post-clamp clearance drop-check (EnforceMinClearance itself is no
        // longer called on this corridor-based path -- see below).
        std::vector<fsd::ClassifiedCone> allConesBody;
        allConesBody.reserve(window.blue.size() + window.yellow.size() + window.orange.size());
        for (const auto *coneList : {&window.blue, &window.yellow, &window.orange})
        {
            for (const auto &c : *coneList)
            {
                const fsd::Point2D body = fsd::WorldToBody(window.vehiclePose, c.x, c.y);
                allConesBody.push_back(fsd::ClassifiedCone{body.x, body.y, ""});
            }
        }
        // No re-ordering needed here: WorldToBody is a rigid (distance- and
        // order-preserving) transform, so the world-frame ordering
        // OptimizeRacingLine/the corridor/spline already carried forward
        // (ultimately from centerline_extractor's OrderWaypointsByTraversal
        // starting at the vehicle's world position) is still correct in
        // body frame.
        //
        // EnforceMinTurnRadius runs LAST -- historical reasoning from when
        // this call site still ran EnforceMinClearance first (removed
        // 2026-09-04, see below): confirmed directly (2026-08-31) as a
        // real, live bug the other way around, EnforceMinClearance's
        // cone-avoidance push has no awareness of the turn-radius
        // constraint, so it can shove a waypoint back OUTSIDE the vehicle's
        // achievable curvature after EnforceMinTurnRadius had just pulled
        // it in -- confirmed live via the actual picked pure-pursuit target
        // implying curvature nearly 2x the physical max, right at a
        // hairpin apex where the racing line deliberately hugs the inside
        // boundary (see path_generator.cpp's own NearestPairMidpointPath
        // for the fuller explanation -- that reactive-pipeline call site
        // still runs both, in this order, for the same reason).
        // Closed-loop: only clamp a GEOMETRICALLY-determined near prefix of
        // the array (the car's actual upcoming path), NOT the whole
        // 555-point loop (2026-09-03, two iterations to get right --
        // history below). EnforceMinTurnRadius's engagement test is body-
        // frame PROXIMITY (|x| < kMinTurnRadius), not array position -- on
        // a track that passes close to the car's current spot more than
        // once (confirmed real, not a bug: /planning/debug_racing_line
        // shows the same underlying data with no artifact), a point that's
        // far away in the array (a DIFFERENT lap/leg of the track) can
        // still land inside that small radius purely because it's
        // geographically nearby, and get incorrectly squashed toward the
        // vehicle's own centerline as if it were the car's own next
        // waypoint -- confirmed live as the exact mechanism behind a
        // visible "hook"/crossover in /planned_path that doesn't exist in
        // /debug_racing_line's unclamped version of the same data.
        //
        // First attempt: a FIXED array-position cutoff (e.g. "first 80/350
        // positions"). Two real problems, both confirmed live: (1) if
        // clamping drops several near points (car genuinely close to a
        // cone), the fixed cutoff still reattaches raw data starting at
        // the ORIGINAL fixed position regardless, leaving a gap -- a
        // published path whose first point was 11.8m from the vehicle
        // origin, a real stuck event. (2) even after making that boundary
        // adaptive, a fixed position range still isn't safe: WHERE the
        // track happens to fold back near itself varies with the car's
        // current position on the loop -- one capture showed a fold-back
        // at array position 96, well inside what should have been a "safe"
        // near zone, so no fixed count/range is universally correct.
        //
        // Correct approach: determine the near zone GEOMETRICALLY instead
        // of by a fixed count. Scan forward from k=0 (RAW, pre-clamp body
        // positions) and stop the very first time a point's range exceeds
        // kWindowRadius -- the same radius allConesBody itself is limited
        // to, so points beyond it have no live cone data to clamp against
        // anyway. Critically this is a ONE-WAY, MONOTONIC scan: once the
        // raw path has left the vehicle's immediate vicinity, it never
        // re-enters "near" classification even if a later point (a
        // different, unrelated leg of the track folding back close by)
        // happens to have a small range of its own -- that's exactly the
        // self-crossing case this whole fix exists to exclude. Points
        // within the resulting prefix still get the normal
        // EnforceMinClearance/TurnRadius treatment (including drops for
        // genuinely unsafe ones); the boundary is deliberately NOT grown
        // back out to backfill drops the way the first fix attempt did --
        // growing past the geometric near-zone boundary would just
        // reintroduce the fold-back risk. A drop-heavy near zone
        // publishing fewer usable points is an accepted, rare degraded
        // case (the same limitation the original small fixed-window design
        // always had), not something to patch further here.
        const size_t diagPreClampCount = racingLineBody.size();
        if (closedLoop)
        {
            size_t cutoff = racingLineBody.size();
            for (size_t k = 0; k < racingLineBody.size(); ++k)
            {
                const double range = std::hypot(racingLineBody[k].x, racingLineBody[k].y);
                if (range > kWindowRadius)
                {
                    cutoff = k;
                    break;
                }
            }
            std::vector<fsd::PathPoint> nearPart(racingLineBody.begin(),
                                                   racingLineBody.begin() + static_cast<long>(cutoff));
            std::vector<fsd::PathPoint> farPart(racingLineBody.begin() + static_cast<long>(cutoff),
                                                 racingLineBody.end());
            // EnforceMinClearance REMOVED from this path entirely (2026-09-
            // 04, user reports "the blue planned path... zigzagged" and
            // separately "fails to stay in track... crossover... near the
            // hairpin" -- STILL present after a first attempt that kept the
            // push but re-clamped with ClampToCorridor afterward). That
            // first attempt only bounded the damage, it didn't remove it:
            // ClampToCorridor's own tolerance (+-1m longitudinal via
            // kMaxLongitudinalDrift, plus the full corridor lateral width)
            // is generous relative to this pipeline's 0.5m sample spacing
            // by design -- it exists to absorb a SMALL cross-cycle blend
            // drift, not a multi-meter push -- so a point could still land
            // anywhere in a ~1-2m cloud independent of its neighbors',
            // which is still enough scatter, relative to 0.5m spacing, to
            // read as a zigzag. Confirmed directly: a live capture with the
            // re-clamp already deployed still showed a 177 degree turn and
            // max per-point displacement of 1.66m from the optimizer's own
            // output.
            //
            // The actual fix is removing EnforceMinClearance's push, not
            // bounding it further. It's now fully redundant: the corridor
            // (corridor.cpp's ComputeCorridor) already reserves
            // kCorridorSafetyMargin on every sample against the nearest
            // blue/yellow/orange boundary cone, and OptimizeRacingLine's
            // SOR solve keeps every point inside that bound BY
            // CONSTRUCTION -- this push, layered on top, can only ever
            // break that guarantee (as just confirmed), never improve on
            // it. It was designed for the OLD reactive pipeline's raw
            // centerline (path_generator.cpp), which had no corridor
            // concept and no other source of clearance guarantee at all --
            // that justification doesn't carry over to a pipeline where the
            // clearance guarantee already exists upstream, smoothly, by
            // construction. EnforceMinTurnRadius is KEPT: it enforces a
            // genuinely different constraint (vehicle kinematics) that
            // corridor width alone says nothing about, and its own clamp is
            // a simple, smooth, monotonic function of body-frame x (not a
            // sum of independent per-cone pushes), so it doesn't reintroduce
            // this same neighbor-incoherence failure mode.
            nearPart = fsd::EnforceMinTurnRadius(std::move(nearPart));
            racingLineBody = std::move(nearPart);
            racingLineBody.insert(racingLineBody.end(), farPart.begin(), farPart.end());
        }
        else
        {
            // See the closed-loop branch's own comment just above for why
            // EnforceMinClearance is no longer called here.
            racingLineBody = fsd::EnforceMinTurnRadius(std::move(racingLineBody));
        }
        const size_t diagPostClampCount = racingLineBody.size();
        // NOT applied to the closed-loop case (2026-09-03) -- this filter
        // drops any point with negative body-frame x, which is exactly
        // right for a short forward-only window (negative x there really
        // does mean "just passed"), but wrong once the FULL loop is
        // published: roughly half of any closed loop's points naturally
        // fall behind the car's CURRENT instantaneous heading simply
        // because they're on the far/opposite side of the track, not
        // because the car passed them. Confirmed live as the cause of the
        // published loop rendering as a chopped/partial shape instead of
        // the actual track outline. The open pipeline's own short window
        // still needs this exactly as before.
        if (!closedLoop)
        {
            racingLineBody = RemoveBehindCarPoints(std::move(racingLineBody));
        }

        // Diagnostic only, rate-limited to once every ~2s (not every empty
        // cycle) -- added specifically to pin down a confirmed live empty-
        // /planned_path failure (2026-08-31) that an offline harness
        // couldn't reproduce (the harness always recomputes fresh at the
        // CURRENT vehicle position, which trivially places index 0 right
        // next to it -- the actual bug only shows up querying a FROZEN
        // array from a position that's drifted away from where it was
        // computed, which only the live process's own real per-cycle
        // state can exhibit).
        // Loosened from "empty" to "thin" (< 10 points) -- confirmed live
        // (2026-08-31) that this pipeline can get stuck with a NON-empty
        // but too-thin path (2 points), which the original empty-only
        // check never caught. Also now reports the stored line's own
        // local heading near the cursor vs. the vehicle's actual current
        // heading -- testing the hypothesis that at a sharp turn the
        // vehicle's nose can be un-aligned enough with the stored line's
        // local direction there that most of the forward window reads as
        // body-frame "behind" even though it's the genuinely correct
        // upcoming path (a case the windowed/open pipeline never faces,
        // since its own forward-facing landmark filter keeps everything
        // roughly aligned with current heading before it ever reaches
        // this stage).
        if (closedLoop && racingLineBody.size() < 10)
        {
            static int thinLogCounter = 0;
            if (thinLogCounter++ % 10 == 0)
            {
                double storedLineHeadingDeg = 0.0;
                if (!storedClosedRacingLine.empty())
                {
                    const size_t n2 = storedClosedRacingLine.size();
                    const fsd::PathPoint &a = storedClosedRacingLine[diagNearestIdx % n2];
                    const fsd::PathPoint &b = storedClosedRacingLine[(diagNearestIdx + 10) % n2];
                    const double lineHeading = std::atan2(b.y - a.y, b.x - a.x);
                    double diff = lineHeading - activeSet.vehiclePose.yaw;
                    while (diff > M_PI) diff -= 2 * M_PI;
                    while (diff < -M_PI) diff += 2 * M_PI;
                    storedLineHeadingDeg = diff * 180.0 / M_PI;
                }
                std::cerr << "planning: closed-loop /planned_path THIN (" << racingLineBody.size()
                          << " pts) -- nearestIdx=" << diagNearestIdx << " nearestDist=" << diagNearestDist
                          << "m preClampCount=" << diagPreClampCount << " preClampBodyX=[" << diagPreClampMinX
                          << "," << diagPreClampMaxX << "]" << " postClampCount=" << diagPostClampCount
                          << " allConesBody=" << allConesBody.size() << " vehiclePose=("
                          << activeSet.vehiclePose.x << "," << activeSet.vehiclePose.y << ","
                          << activeSet.vehiclePose.yaw << ")"
                          << " storedLineHeadingVsVehicleDeg=" << storedLineHeadingDeg << "\n";
            }
        }

        gz::msgs::Pose_V pathMsg;
        for (const auto &wp : racingLineBody)
        {
            gz::msgs::Pose *p = pathMsg.add_pose();
            p->mutable_position()->set_x(wp.x);
            p->mutable_position()->set_y(wp.y);
        }
        pathPub.Publish(pathMsg);
    };
    if (!node.Subscribe("/estimated_landmarks", onEstimatedLandmarks))
    {
        std::cerr << "Failed to subscribe to /estimated_landmarks\n";
        return 1;
    }

    std::cout << "planning: /cone_detections -> /planned_path (reactive, cold-start)\n";
    std::cout << "planning: /estimated_landmarks + /estimated_pose -> /planned_path "
                 "(landmark-based racing line, once ready)\n";

    gz::transport::waitForShutdown();
    return 0;
}
