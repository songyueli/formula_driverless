#include "pure_pursuit_controller.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace
{
// Straight-line target speed -- confirmed to hold up at full active-landmark
// capacity (kMaxActiveLandmarks=80) once the EKF-side scaling fixes (sparse
// corrections, spatial-grid retired-landmark search) were in place.
//
// Speed scaling history (2026-08-31): the original curvature-based scheme
// (instantaneous + EMA + a forward preview) was removed at user request --
// felt like it slowed "tremendously" for no clear benefit. Removing it
// entirely was then confirmed live to let the car drift off the intended
// path on corners (the EMA/preview machinery existed for a reason, just
// not one this simpler replacement needs the same complexity for). Speed
// is now a direct linear function of the STEERING ANGLE the current
// curvature implies (see kWheelBase/kSteeringLimit below), not curvature
// itself -- the angle naturally saturates at the vehicle's own physical
// steering_limit, so there's no need for a separate artificial curvature
// cap the way the old kMaxExpectedCurvature was.
// Tried 6.0/7.0 before (2026-08-31) and both got stuck -- traced at the
// time to two real bugs since fixed: (1) the EnforceMinClearance/
// EnforceMinTurnRadius ordering bug (a clearance push could shove a
// waypoint back outside the vehicle's achievable curvature after the
// turn-radius clamp had just pulled it in -- see path_generator.cpp's own
// NearestPairMidpointPath), and (2) the AckermannSteering plugin's
// steer_p_gain being an unset, soft default (see model.sdf's own comment
// -- steering took 1.5-2+ seconds to converge on a held command). Retried
// 7.0 again after both fixes (plus the behind-car path filter and
// speed-proportional adaptive lookahead, added the same session) -- still
// crashed hard, but traced THAT to a third, genuinely different bug: at
// speed=7.0, the adaptive lookahead formula wanted 7.0m but kMaxLookahead
// was still 6.0 (a leftover from before this constant existed), silently
// giving LESS than a full second of lookahead exactly where more was
// needed most -- see kMaxLookahead's own comment. Tried 8.0 next (same
// session) with kMaxLookahead raised to 9.0 for headroom -- still crashed,
// this time a rollover, WHILE the racing-line cache (racing_line_cache.cpp)
// was also active for the first time in the same test. Reverted to 5.0
// (last confirmed-clean full lap) to isolate: is it the speed, the cache,
// or their interaction? Retry 8.0 again only after the cache itself is
// separately confirmed clean at 5.0.
// Reverted from 20.0 (2026-09-01): confirmed live as a real crash, not a
// theoretical risk -- chassis z spiked to 1.26m (vs. the normal ~0.31m
// baseline and every prior confirmed collision's own worst case, ~0.56m)
// within the first 30 seconds of driving. 20 m/s matches real FSAE top
// speed, but this software (lookahead/corridor/turn-radius tuning) has
// never been validated anywhere near it -- 5.0 is the last speed with a
// full confirmed-clean run.
//
// Retrying 8.0 (2026-09-01, target is eventually 15.0): the precondition
// this file's own comment above was waiting on -- "the cache itself
// separately confirmed clean at 5.0" -- is now satisfied. This same
// session ran extensive live testing at 5.0 with racing_line_cache.cpp AND
// the closed-loop pipeline both active (multiple soak tests, both sharp
// corners, the hairpin specifically) and found zero rollover/dynamics
// crashes -- every issue traced to landmark-map/stuck-watchdog bugs, now
// separately fixed, not vehicle dynamics. kMaxLookahead=9.0 already has
// headroom above 8.0 (see its own comment), so no lookahead change needed
// for this step. Going straight to 15.0 is NOT attempted here -- that's a
// 3x jump past the last confirmed-clean speed with no incremental
// validation, the same mistake the 20.0 attempt made. Step up from here
// (8 -> 11 -> 15) only after each step is confirmed clean, specifically
// through both sharp corners and the hairpin -- raise kMaxLookahead
// alongside kMaxSpeed once it stops having headroom (kLookaheadTimeConstant
// =1.0s means the adaptive formula wants exactly kMaxSpeed meters of
// lookahead at full speed -- see kMaxLookahead's own comment for why
// under-clamping this was already a confirmed real bug once, at 7.0).
//
// Stepping to 11.0 (2026-09-01): 8.0 was confirmed clean (chassis stayed
// at the 0.31m baseline through both sharp corners and the hairpin, with
// kBrakePreviewDistance's forward-preview braking added the same session)
// AFTER the actual wedge root cause -- OrderWaypointsByTraversal's
// hairpin-fold-back bug, see path_utils.cpp's own comment -- was found and
// fixed, confirmed by a full clean lap (311.836m, zero stuck events) plus
// continued clean driving into lap 2. kMaxLookahead's own headroom no
// longer covers this step (9.0 < 11.0, the exact under-clamping pattern
// already confirmed as a real bug once before at 7.0/6.0) -- raised
// alongside it, see kMaxLookahead's own comment for the new value.
// Stepping to 15.0 (2026-09-02): 11.0 confirmed clean after re-enabling the
// forward-preview braking/adaptive lookahead/speed smoothing that had been
// accidentally left disabled from the prior session's oscillation bisection
// (see kEnableForwardPreviewBraking's own comment in Compute()) -- 138+
// seconds of continuous driving, zero stuck events, including straight
// through the (-21.7,-11.2) hairpin that had been failing every run before
// that fix. This is the final planned step (8 -> 11 -> 15, see this
// constant's own history above) -- not a fresh 3x jump, the incremental
// validation this comment's own history insisted on has already happened.
// LOWERED 25.0 -> 15.0 (2026-09-04, user request: "too fast going into the
// turn. let's turn the max speed down until we fully optimize the racing
// line"). 25.0 was the user's own manual edit, never separately step-
// validated against closed-loop driving the way every value below it was
// (8->11->15, see this constant's own history above) -- and the racing
// line itself is now back to the safe, barely-optimized (neighborRadius=1)
// centerline-hugging shape (planning.cpp, 2026-09-04) after repeated
// live failures at a wider, more aggressive shape -- a car essentially
// following raw centerline through corners needs more margin, not less,
// than one following a properly widened line. 15.0 is not a fresh guess --
// it's this file's own last INCREMENTALLY validated closed-loop speed,
// confirmed clean specifically against the same class of conservative line
// this file's own history describes. Raise again only after the racing
// line is properly re-optimized (see planning.cpp's own deferred-goal
// comment) and speed is stepped back up incrementally, not jumped.
//
// LOWERED 20.0 -> 11.0 (2026-09-04, user request: "decrease the max speed
// in the mean time while we debug the racing line issues"). 20.0 was the
// user's own manual edit past the 15.0 this file's history above had just
// argued for, and the racing-line-shape investigation (a background
// redesign, offline-validated before anything gets deployed live) is
// still actively in progress -- safety margin during that window matters
// more than speed. 11.0 is not a fresh guess -- it's this file's own
// second-to-last incrementally-validated step (8->11->15), one full step
// more conservative than 15.0 itself. Raise back to 15.0, then further,
// only once the racing-line redesign is actually deployed and confirmed
// clean -- not before.
// LOWERED 18.0 -> 15.0 (2026-09-05, user report: two separate rollovers on
// the FIRST two real closed-loop attempts this session, at two different,
// only-moderate-curvature corners -- not even the hardest hairpin). 18.0
// was raised past 15.0 ("this file's own last INCREMENTALLY validated
// closed-loop speed", see the paragraph above) without the same step-and-
// revalidate discipline every other constant in this file has otherwise
// required -- confirmed today as a real, not theoretical, gap: the
// corridor's own margin (kCorridorSafetyMargin=0.95m leaves only
// ~0.5-0.55m of real clearance per side on an ordinary, non-drifted
// section) is a fixed, speed-independent budget, so the faster the car
// drives, the less of that budget any real tracking error can consume
// before it becomes a cone strike. This exact "slower driving doesn't fix
// a genuine tracking/planning imprecision, but it directly buys more
// real-world margin against it" tradeoff is why kFirstLapMaxSpeed/
// kMinSpeed were lowered the same way earlier this session. Back to 15.0,
// not further -- the closed loop has never yet completed a single clean
// lap at ANY speed this session, so 15.0 (an anchor with SOME live
// history, even if against an earlier racing-line shape) is the more
// defensible first step down than an untested lower guess. Paired with
// kClosedLoopMaxLookahead=6.0 above: 6.0/15.0=0.4s reaction time, matching
// the open pipeline's own validated kMaxLookahead/kFirstLapMaxSpeed ratio
// (2.0/5.0=0.4s) exactly.
// LOWERED 15.0 -> 10.0 (2026-09-05, same session, third distinct closed-
// loop incident in a row -- this one a short-term stuck/stall rather than
// a rollover, right at the start/finish orange gate, the single tightest
// corridor point on the track (orange-gate tightening in corridor.cpp
// stacks on top of the already-thin ~0.5m/side blue/yellow margin there,
// confirmed live: 0.95m combined width, tighter than the ~1.0-1.4m typical
// elsewhere). Each of these three incidents has been at a DIFFERENT
// location, which points at a general tracking-margin deficit at this
// speed rather than one fixable local bug -- the closed loop has now run
// three times at 15.0-18.0 m/s and never once completed lap 2. 10.0 sits
// between this file's own historical "8" and "11" incremental steps
// (8->11->15, see above), prioritizing actually finishing laps (the user's
// explicit goal) over top speed while re-establishing this file's own
// step-and-revalidate discipline. Raise back toward 15.0 only after a
// confirmed-clean run at 10.0.
// LOWERED 10.0 -> 5.0 (2026-09-05, same session, FOURTH closed-loop
// incident in a row -- a rollover during lap 3 itself, after two full
// CLEAN laps at 10.0). Directly ruled out landmark drift this time
// (nearest blue landmark to the real cone was only 0.27m off, ordinary EKF
// noise) and ruled out a corridor-computation bug (width a consistent,
// correctly-computed ~1.05m, matching this track's normal ~3.0m width
// minus 2*kCorridorSafetyMargin=1.9m, not a floor artifact). Same
// signature as the two earlier rollovers: a genuinely, by-design tight
// corridor (~0.5m real clearance per side after the margin) that leaves no
// slack for whatever the car's actual tracking error is at closed-loop
// speed -- three successive, real speed cuts (18->15->10) each still hit
// this same wall at a different location, which is evidence the mismatch
// is not simply "still slightly too fast" but a genuine gap between
// nominal margin and real tracking precision that a bigger, more decisive
// cut is needed to bracket. 5.0 removes speed as a variable entirely: it's
// the SAME value already proven, repeatedly, to complete clean laps on the
// open/reactive pipeline (kFirstLapMaxSpeed/kMinSpeed, see their own
// history) -- if the closed loop still can't complete 3 laps at its own
// pipeline's tracking-precision baseline speed, that points at the
// racing-line/corridor GEOMETRY itself, not speed, and further speed cuts
// past this point stop being a meaningful lever. Raise back up
// incrementally (10, then 15) only after a confirmed-clean multi-lap run
// at 5.0 -- this is now the same one-known-good-anchor-first discipline
// kFirstLapMaxSpeed/kMinSpeed already follow.
// RAISED straight to 15.0 by manual edit (author unclear), skipping the
// 10.0 intermediate step this comment explicitly calls for.
// LOWERED BACK to 10.0 (2026-09-08, user report: "we're oscillating the
// steering", confirmed live -- car stalled from an oscillation-induced
// no-forward-progress stop, same signature as the earlier 15.0/18.0
// incidents this same session). 10.0 is not a fresh guess -- it previously
// completed 2 full clean laps in a row before this exact re-raise; restore
// that and re-validate incrementally from there rather than jumping back
// to the value that just failed again.
// RAISED 10.0 -> 13.0 (2026-09-08, later same session, user request:
// "increase the max speed after the first lap"). Different context than
// every earlier attempt at 15.0/18.0: the two mechanisms actually
// responsible for the oscillation/stall incidents that forced this value
// back down to 10.0 have since been root-caused and fixed, not worked
// around by capping speed -- (1) the steering joints' own P-only,
// undamped controller (simulation/models/fsd_car/model.sdf, now damped +
// retuned gain, see that file's own comment), and (2) centerline_extractor
// .cpp's midpoint-pairing fallback tier, which could silently accept a
// wrong-leg cross-pair at the hairpin (see TwoPointMidpointExtractor's own
// BUG FIX comment) -- fixed and live-validated the SAME session, 4
// consecutive clean laps (lap 1 through lap 4 logged complete) at
// kMaxSpeed=10.0 with zero STUCK events and zero collision-height bumps.
// 13.0 is a moderate step (~30%), matching this file's own historical
// increment size (5->8->11->15 before oscillation forced retreat), not a
// jump back to the never-safely-validated 15.0/18.0.
//
// LOWERED BACK 13.0 -> 10.0 (2026-09-09, same session, confirmed live: TWO
// separate rollovers at 13.0, at two DIFFERENT track locations, across two
// DIFFERENT steering-joint damping values (0.5 and the previously-"safe"
// 0.8) -- see model.sdf's own steer_p_gain/damping history for the first
// rollover, which was (incorrectly, it now looks like) attributed solely
// to that damping change at the time. A rollover recurring at the SAME
// damping value (0.8) that had 4 clean laps behind it at kMaxSpeed=10.0
// means damping wasn't the (sole) variable -- the one thing common to
// both incidents is this speed increase itself, raised the same session
// without the incremental per-step revalidation this file's own history
// otherwise insists on (13.0 was tested live for the first time only in
// the same session as both rollovers). Reverted to 10.0, the last value
// with real multi-lap validation behind it, rather than keep testing at a
// speed that's now produced two crashes. Re-raise only after confirming
// clean laps at 10.0 again post-revert, and step up in smaller increments
// than the original 10->13 jump, watching specifically for a third
// rollover before going any further.
constexpr double kMaxSpeed = 10.0;  // m/s

// First-lap cap (2026-09-03, user request: "couldn't get past the hairpin"
// at kMaxSpeed=25.0 -- add something that makes the first lap slower). The
// FIRST lap runs the open/reactive pipeline: no full racing line exists
// yet, only a live windowed view of whatever cones are currently visible
// (see planning.cpp's own doc comment) -- inherently less certain than the
// closed loop's own pre-optimized, turn-radius/clearance-validated line.
// kMaxSpeed itself was validated incrementally (8->11->15, see its own
// history above) specifically against CLOSED-LOOP driving; the open
// pipeline never went through that same step-by-step process. Detected via
// inputs.path's own size rather than a new topic/field: the closed-loop
// pipeline now publishes the FULL ~550-point loop on /planned_path
// (planning.cpp, 2026-09-03), while the open pipeline's own windowed
// output stays in the tens -- already the exact signal that matters (is
// the car actually following the validated closed-loop line right now),
// with no new plumbing needed.
//
// LOWERED 10.0 -> 5.0 (2026-09-03, same day): 10.0 didn't prevent a crash
// traced to a real root cause this cap alone can't fully fix -- the
// published path going completely EMPTY (confirmed live, /planned_path had
// 0 points at the moment of a stuck event), meaning localization/
// perception weren't providing enough cone data for planning to produce a
// line at all. An empty path bypasses this whole speed law regardless of
// its value (see the early-return on inputs.path.empty() near the top of
// Compute()), so no cap fully solves that data-availability gap -- but
// driving slower during the inherently less-certain first lap still helps
// as a stopgap (more time per unit distance for perception to catch up,
// smaller consequences from a brief empty-path moment). 5.0 is not a fresh
// guess -- it's this file's own longest-standing, most validated clean
// speed (see kMaxSpeed's own history above: "5.0 is the last speed with a
// full confirmed-clean run"). The actual perception/localization data-gap
// this is working around is a separate, unaddressed issue.
// LOWERED 10.0 -> 5.0 (2026-09-05, user goal: 10 laps, zero stuck events).
// Matches kMinSpeed exactly -- the open pipeline now drives at a CONSTANT
// 5.0 m/s regardless of curvature, the most conservative choice available
// without further live tuning. Confirmed live tonight: the car reliably
// (over 48+ consecutive attempts) never survives the open/first-lap
// pipeline at all -- physically wedging against a boundary cone (commanded
// to move, real displacement zero) at multiple different track locations,
// all at completely normal (~3.00m) track width, ruling out a narrow-
// section/margin explanation. Corridor margin and lookahead adjustments
// made the same night did not resolve it. 5.0 is not a fresh guess -- it's
// this exact constant's own previously-validated value (see the history
// just above: "the last speed with a full confirmed-clean run"), raised
// back to 10.0 at some point since without the same incremental
// revalidation every other speed constant in this file has otherwise
// required. Slower driving doesn't fix a genuine tracking/planning
// imprecision, but it directly buys more real-world margin against it
// (more reaction time per unit distance, smaller consequences from the
// same positional error) -- the most reliable lever available under time
// pressure. Raise again only after confirming clean laps at this value
// first.
// LOWERED 5.0 -> 3.0 (2026-09-05, same day as kMinSpeed's own matching
// change just below -- see its own comment for the full reasoning: five
// separate, confirmed-real logic fixes across this session (corridor
// safety-floor, landmark-window heading margin, midpoint-ordering
// heading-reversal threshold, stuck-watchdog timing, yaw-rate smoothing)
// still hadn't gotten the car past this exact hairpin, and the remaining
// suspected cause -- perception's own near-field detection gap -- isn't
// something safe to patch under time pressure without reading and
// understanding that pipeline properly first. Slower driving doesn't fix
// that gap, but it directly buys the open pipeline more real time per
// unit of track distance to work through it before the car's own position
// outruns what's been reliably mapped. Kept equal to kMinSpeed, same as
// before this change -- still a constant speed through the whole open
// pipeline, not reintroducing curvature-based variation that was never
// validated at this lower value.
// RAISED 3.0 -> 5.0 (2026-09-05, user report: "too slow"). The actual root
// cause this 3.0 floor was hedging against turned out to be a real,
// confirmed-and-fixed planning bug -- TwoPointMidpointExtractor's mutual-
// nearest-neighbor cross-pair rejection dropping legitimate cones at the
// hairpin with no fallback, truncating the path at exactly that apex (see
// centerline_extractor.cpp's own comment) -- not a perception gap needing
// an open-ended speed hedge. Restored to 5.0, this constant's own longest-
// standing confirmed-clean value (see the "10.0 -> 5.0" history above),
// not all the way back to the never-revalidated 10.0. Kept equal to
// kMinSpeed, same invariant as before.
// RAISED to 8.0 by manual edit (author unclear), breaking the "kept equal
// to kMinSpeed" invariant this comment itself states (kMinSpeed=5.0).
// LOWERED BACK to 5.0 (2026-09-08, user report: "we're oscillating the
// steering", confirmed live) -- this exact value was already flagged, in
// its own right, as "needs live validation, specifically watching for the
// steering feels lagged/oscillating symptom" the day it was first tried,
// and that symptom is exactly what recurred. Restoring the kMinSpeed
// invariant, not guessing at a new number.
// RAISED 5.0 -> 6.5 (2026-09-08, later same day, user request: "increase
// the first lap speed"). Different context than the two attempts above:
// the oscillation that killed both the 8.0 attempt AND the 18.0/15.0
// closed-loop attempts was since root-caused and fixed at the actual
// mechanism (simulation/models/fsd_car/model.sdf's steering joints --
// steer_p_gain was an unset/then untested-stiff, undamped P-only
// controller; now damped and re-tuned, see that file's own comment), not
// worked around by capping speed. A moderate step, not a jump straight
// back to the previously-risky 8.0 -- also intentionally ABOVE kMinSpeed
// now (breaking the "kept equal" invariant on purpose this time), so the
// open pipeline's own grip-based cornering law can actually vary speed by
// curvature again instead of being pinned constant. Not yet validated
// across multiple clean laps with the new steering fix in place -- watch
// for a recurrence of oscillation specifically before raising further.
// LOWERED BACK 6.5 -> 5.0 (2026-09-08, minutes later, user reports: "the
// planned path gave up again at the hairpin" and "we're also still
// oscillating around the centerline" -- both landed right after this
// raise, and a live capture right then confirmed the hairpin's own
// windowed data was genuinely thin (only 4 raw midpoints) at that exact
// moment. Not conclusively PROVEN this speed raise caused it (thin
// windowed data at the hairpin has independent causes too), but the
// timing is suspicious enough, and the hairpin is fragile enough already,
// that trading the requested speed bump for one less variable while still
// actively debugging is the safer call -- restore once the hairpin holds
// up cleanly across multiple runs at 5.0 first.
// RAISED BACK 5.0 -> 6.5 (2026-09-08, later same session, explicit user
// request: "increase speed on the first lap"), before the "multiple clean
// runs at 5.0" precondition just above was actually met -- an explicit
// override, not a lapse in discipline. Different context than the same
// jump earlier today: both suspected mechanisms behind the original
// 6.5 rollback (steering-joint P-gain/damping ringing, and
// path_generator.cpp's MinimizeCurvature eating this track's own
// ~0.10-0.20m real per-side clearance margin) have real, live-diagnosed
// fixes in place now, not just a fresh guess at the same number. Still
// the same 6.5 step as before, not a bigger one -- watch specifically for
// the hairpin and the "oscillating around centerline" symptom recurring
// before raising further.
constexpr double kFirstLapMaxSpeed = 6.5;  // m/s
constexpr size_t kClosedLoopPathSizeThreshold = 100;  // points
// Floor speed at/beyond the vehicle's max steering angle. Raised from the
// original 1.0 (2026-08-31, user report: "too slow") -- 1.0 was tuned
// against the OLD raw-centerline path, which genuinely needed near-max
// steering angle sustained through the whole hairpin; the new landmark-
// based racing-line pipeline (see planning/include/racing_line_optimizer.hpp)
// widens exactly that corner specifically so the car rarely needs to
// approach kSteeringLimit at all anymore, so the old floor's own
// justification is largely gone. 2.0 is a first retry, not yet validated
// live the way 1.0 was -- retest at the hairpin specifically before
// trusting it the same way.
// LOWERED 5.0 -> 3.0 (2026-09-05, user report: "we didn't even make it
// past the first lap" -- the SAME hairpin section, confirmed live three
// separate times this same session after five distinct, confirmed-real
// fixes elsewhere in this pipeline (corridor.cpp's safety floor,
// landmark_map.cpp's heading margin, path_utils.cpp's heading-reversal
// threshold, this file's own stuck-watchdog timing and yaw-rate
// smoothing). This isn't a fresh guess at "the" fix -- it's a deliberate,
// last-resort SAFETY MARGIN increase given the remaining suspected root
// cause (perception's own near-field cone-detection gap at this specific
// viewing angle, confirmed live via its own per-detection log) is not
// something to patch blindly under time pressure. A lower floor speed
// doesn't fix that gap, but it's strictly conservative: kStuckDistanceThreshold
// =0.3m over kStuckCyclesBeforeLatch=270 cycles is still trivially cleared
// by genuine driving at 3.0 m/s (well over 0.3m within a handful of
// cycles), so this doesn't risk misreading real progress as stuck. Only
// affects the FLOOR everywhere (both pipelines) and the open pipeline's
// own ceiling (see kFirstLapMaxSpeed, kept equal) -- the closed loop's own
// kMaxSpeed is untouched, since the closed loop has already shown it can
// navigate this same section fine once achieved.
// RAISED 3.0 -> 5.0 (2026-09-05, same day/reason as kFirstLapMaxSpeed's own
// matching change above -- the suspected perception gap this was hedging
// against was actually the now-fixed TwoPointMidpointExtractor cross-pair
// bug, not an open-ended reason to hold speed down). Kept equal to
// kFirstLapMaxSpeed, same invariant as before.
constexpr double kMinSpeed = 5.0;  // m/s

// kSteeringSpeedPenaltyCoeff REMOVED (2026-09-04) along with the whole
// steeringAngle/steeringFraction-based cornering speed law it belonged to
// -- see speedForCurvature's own comment in Compute() for the grip-based
// replacement and why (user report: "still going too fast around the
// corners causing us to skid out"). This coefficient's entire multi-
// iteration tuning history (0.7 -> 1.3 -> 1.0) was retuning a formula with
// no direct physical relationship to actual lateral grip in the first
// place -- superseded, not generalized, by the replacement.

// Lookahead distance is now proportional to speed (m_lastSpeed, see its
// own comment in the header for why last-cycle's speed, not this cycle's),
// bounded to [kMinLookahead, kMaxLookahead] -- replaces a single fixed
// 3.0m used regardless of speed (2026-08-31, user request). A FIXED
// lookahead at a higher speed corresponds to a shorter REACTION TIME (the
// same lookahead distance is covered in less time), which is a real
// mechanism for exactly the "steering feels lagged/oscillating" symptom
// this session chased: too-short a lookahead at speed makes the geometric
// target -- and therefore the commanded curvature -- change more sharply
// from cycle to cycle than the vehicle (even with steer_p_gain now fixed)
// can track smoothly. kLookaheadTimeConstant is a "look this many seconds
// ahead" gain, the standard way adaptive pure-pursuit lookahead is tuned;
// 1.0s is a first, untested-live value -- retest and retune from a fresh
// measurement the same way every other first-attempt constant in this file
// has been.
constexpr double kLookaheadTimeConstant = 1;  // seconds
constexpr double kMinLookahead = 2.0;  // meters
// Confirmed directly (2026-08-31) as a real mismatch, not just a
// theoretical one: at kMaxSpeed=7.0 with kLookaheadTimeConstant=1.0, the
// "look 1 second ahead" formula wants 7.0m, but this was clamped to 6.0 --
// backwards from the whole point of adaptive lookahead (MORE lookahead at
// higher speed for stability), giving LESS than a full second of
// look-ahead exactly in the speed regime that needed it most. Set with
// headroom above kMaxSpeed's own value so this clamp only ever engages as
// a genuine safety bound, not a silent ceiling on the adaptive formula
// itself. Left at 9.0 (headroom above kMaxSpeed) even though kMaxSpeed
// itself was reverted to 5.0 below -- this constant was never implicated
// in the rollover, no reason to also revert it.
//
// Raised to 17.0 (2026-09-02) alongside kMaxSpeed's own step to 15.0 --
// same ~2m headroom margin as the prior 13.0-for-11.0 pairing, keeping this
// a genuine safety bound rather than a silent ceiling on the adaptive
// formula (see this comment's own opening paragraph for why that matters).
//
// LOWERED 17.0 -> 5.0 (2026-09-03, explicit user request: "it should never
// be looking that far ahead"). This is now the OPPOSITE role from the
// paragraph above -- no longer just a headroom safety bound above what the
// proportional formula would naturally ask for, but a deliberate hard
// ceiling the user wants engaged: since kMinSpeed=5.0 and the car now
// regularly cruises well above that (12-13+ m/s after the distance-aware
// preview-braking fix), kLookaheadTimeConstant(1.0)*speed exceeds 5.0m
// almost continuously, so in practice this clamp is now the ACTIVE bound
// most of the time rather than a rarely-engaged safety margin -- adaptive
// lookahead is effectively flattened to a near-constant 5.0m at normal
// driving speeds by design, not a side effect.
//
// RAISED 5.0 -> 8.0 (2026-09-03, same day, explicit user request after
// live-testing 5.0 with the preview-scan numerical-sensitivity fix and the
// first-lap speed cap both in place): needs live validation, specifically
// watching for the "steering feels lagged/oscillating" symptom
// kLookaheadTimeConstant's own comment describes -- a short lookahead at
// speed is a real mechanism for that, so this is a genuine tradeoff against
// the "never look too far ahead" request 5.0 was originally set for, not a
// free improvement.
//
// LOWERED 8.0 -> 2.0 (2026-09-04, explicit user request: "the debug target
// distance from the car is way too high. we're clipping the corners. Set
// it to max 2m"). Equal to kMinLookahead (2.0) -- together these two pin
// the lookahead distance to EXACTLY 2.0m at all times, overriding both the
// speed-proportional base distance and the curvature-based cap
// (speedForCurvature's lookaheadDistance derivation) entirely, same as
// disabling adaptive lookahead altogether (kEnableAdaptiveLookahead is
// already false, so this only affects the curvature-cap path in practice,
// which now can never bind above 2.0m either). A genuine tradeoff against
// the earlier 2026-09-03 "steering feels lagged/oscillating" concern this
// same constant's own history flags -- explicit user request, matching the
// same "corner-cutting" mechanism the curvature-based cap above was
// already trying to address, just pushed further/unconditionally rather
// than only near tight turns.
// RAISED 2.0 -> 3.5 (2026-09-05 overnight, user goal: 10 laps, zero stuck
// events), then REVERTED 3.5 -> 2.0 (2026-09-05 later that morning, user
// report: "the planned path at the hairpin is still going too wide through
// the turn causing us to hit the blue cones on the left"). The overnight
// raise was a hypothesis -- that per-cycle corridor jitter needed more
// lookahead to average over -- aimed at a wedging failure whose ACTUAL
// root cause turned out to be unrelated: RemoveBehindCarPoints in
// planning.cpp was truncating /planned_path at exactly this kind of sharp
// turn (a global negative-x filter stripping genuinely-ahead points
// whenever the vehicle's heading lagged the track's own curvature, not
// just a leading prefix of already-passed ones -- see that function's own
// fix comment). With the REAL cause fixed there, this constant's own
// 3.5m compromise no longer has a live problem to justify it, and the
// user's live report is EXACTLY the corner-cutting/wide-through-the-turn
// symptom this constant's own original 2026-09-04 change ("Set it to max
// 2m") was written to fix in the first place -- a longer lookahead
// authors a wider arc through a curve, and 3.5m was already most of the
// way back toward the pre-fix 8.0m that caused it. Back to 2.0m ==
// kMinLookahead, restoring the original pinned-minimum behavior; if
// per-cycle jitter turns out to be a genuine, separate problem later, the
// fix is damping the corridor's own sample-to-sample noise directly, not
// re-opening this constant's own range as a side effect.
constexpr double kMaxLookahead = 2.0;  // meters

// Forward-preview braking distance (2026-09-01) -- see its own use in
// Compute() for why this exists (advance warning of an upcoming tight
// corner, not just the single reactive lookahead target's own curvature).
// Deliberately well beyond kMaxLookahead: the whole point is to see a sharp
// turn coming BEFORE the reactive scheme would have picked it up on its
// own. 15m fits comfortably inside the closed-loop pipeline's own published
// horizon (planning.cpp's kClosedLoopPublishCount=60 points at 0.5m
// spacing = 30m), while still being within the ~20m range perception
// itself trusts (lidar_projector.cpp's kMaxValidRange) for the open/
// reactive pipeline's shorter published path.
//
// Raised to 20.0 (2026-09-01) alongside kMaxSpeed's step to 11.0 -- kept
// at the same ratio above kMaxLookahead as before (was 15/9 =~ 1.67x; 20/13
// =~ 1.54x, close enough) so the margin between "reactive" and "preview"
// horizons doesn't shrink as speed climbs. Capped at 20.0 rather than
// scaled further: this is lidar_projector.cpp's own kMaxValidRange ceiling
// (see the paragraph above) -- the open/reactive pipeline's published path
// realistically can't extend meaningfully past there regardless of this
// constant's own value, since no trusted cone data exists beyond it.
//
// Reduced to 10.0 (2026-09-02, user report: "debug target is wayyyy too
// close to the car all the time... we're not racing"). Root cause,
// confirmed live via a clean 15s /cmd_ackermann trace at kMaxSpeed=15.0:
// commanded speed never rose above 9.16 m/s -- the preview scan takes the
// WORST (sharpest) curvature found ANYWHERE in its whole window as if it
// applied right now, with no regard for how far along that window the
// sharp point actually is. On this track's real cone spacing, a 20m window
// essentially always contains SOME upcoming corner, so this pinned speed
// (and therefore m_lastSpeed, and therefore the speed-proportional reactive
// lookahead distance -- see kLookaheadTimeConstant) low almost
// continuously, which is exactly the "target too close" symptom: a short
// lookahead is a direct, visible consequence of a speed that never gets
// the chance to climb. 10.0 still gives meaningfully more advance warning
// than the reactive scheme's own lookahead at low-to-moderate speed (e.g.
// 2x+ margin at kMinSpeed=2.0's own ~2m reactive lookahead), without
// reaching far enough to almost always catch a distant, not-yet-relevant
// corner. Re-validate at the hairpin specifically -- this is a real
// tradeoff against the original 2026-09-01 bump this feature exists to
// prevent, now at a higher kMaxSpeed than when that fix was first tuned.
// RAISED 8.0 -> 15.0 (2026-09-04, user request: "still going too fast
// through the turn... needs to be tuned for earlier slowdown"). Grounded
// directly in the physics this law itself uses: allowedSpeed =
// sqrt(target^2 + 2*a*dist), so the FARTHEST point in the preview window
// bounds the highest speed the car is ever allowed to carry, and at the
// current kMaxSpeed=15.0/kMinSpeed=5.0/kMaxPreviewDeceleration=2.5, a full
// brake from 15 down to 5 needs (15^2-5^2)/(2*2.5) = 40m of straight-line
// distance -- more than 5x the old 8.0m window, meaning the preview law
// COULD NOT physically request early-enough braking for the sharpest
// corners no matter how sharp the upcoming curvature actually was; the
// window itself was the binding constraint, not the deceleration
// assumption. 15.0 doesn't reach the full 40m need (a corner requiring the
// full drop to kMinSpeed still won't get maximum advance warning), but is
// a substantial, physics-motivated step toward it, not a round-number
// guess. Raising THIS far is safer now than the last time this constant
// sat at/above this range (was cut 20.0->10.0->8.0 on 2026-09-01/02
// specifically because a wide window kept picking up distant, not-yet-
// relevant curvature and pinning speed low) -- that failure was traced to
// the preview scan's curvature formula measuring "chord from the car's
// CURRENT position" rather than the path's own true local curvature,
// since fixed (2026-09-03, Menger/three-point curvature, see this loop's
// own comment below) -- a genuinely sharp DISTANT corner should now
// correctly read as sharp only once the car is actually close enough that
// the distance-aware allowedSpeed formula says so, not as an artifact of
// how far the car currently has to turn to reach it. Not yet live-
// validated at this new value -- retest specifically for the "picks up
// distant corners" symptom returning before pushing toward the full 40m.
//
// LOWERED 15.0 -> 10.0 (2026-09-04, same day, user request: "decrease the
// lookahead distance for checking curvature for braking"). This is exactly
// the risk this comment's own previous paragraph flagged retesting for --
// 15.0 gave more advance warning but likely reintroduced some version of
// the original "reacts to curvature that isn't relevant yet" symptom this
// constant has a repeated history of (cut 20->10->8 for that same reason
// on 2026-09-01/02, before being raised again today). 10.0 is a middle
// ground between the two failure directions already confirmed live at
// this constant's extremes -- 8.0 (too short to request meaningful advance
// braking at all) and 15.0/20.0 (too far, picks up not-yet-relevant
// curvature) -- not a fresh guess in either direction. Needs live
// validation like every value tried here today.
constexpr double kBrakePreviewDistance = 10.0;  // meters

// Assumed comfortable (not emergency) braking deceleration for the
// distance-aware preview law below -- how hard the car is willing to slow
// down in order to make an upcoming corner's target speed by the time it
// gets there. Retune from an actual /cmd_ackermann trace approaching the
// hairpin -- too low and the car still won't reach kMaxSpeed on approach
// to any corner; too high and it brakes later than the vehicle can
// actually manage, eating back into the same "arrived too fast" failure
// kBrakePreviewDistance's own history documents.
//
// LOWERED 5.0 -> 2.5 (2026-09-02): the first value treated deceleration as
// a straight-line-only budget, available independently of whatever lateral
// (cornering) grip the turn itself already demands -- a real tire's total
// grip is a single shared budget between braking and cornering (the
// "friction circle"), not two separate ones, so 5.0 let the car plan to
// brake right up to entering a bend while still expecting to hold that
// bend's own lateral demand, asking for more combined grip than the
// vehicle has. Confirmed live: at 5.0, a stuck event showed the car's
// ACTUAL position 2.2m off the stored racing line's nearest point with an
// anomalously low chassis height (0.21m vs the normal ~0.31m resting
// height) -- the signature of a real clip/bump, not a planning-data
// defect (the corridor at that exact spot was a perfectly ordinary ~3.00m
// wide section, not a genuinely tight one). 2.5 was a conservative
// starting correction, not a precisely-derived one. User then manually
// raised it to 5.0 for live testing.
//
// RAISED 2.5 -> 9.5 (2026-09-03, user request: "help me figure out the
// actual deceleration capabilities of our car"). Grounded in this sim's own
// tire mu=1.5 -> ~14.7 m/s^2 (~1.5g) theoretical straight-line max, with 9.5
// picked as "~65% of that, leaving friction-circle margin for cornering."
// That reasoning had a real flaw: it treats the braking/cornering split as
// a FIXED ratio applied uniformly across the whole preview window, but the
// lateral demand isn't uniform -- it's highest exactly AT the corner the
// car is braking FOR, which is precisely where the least braking budget is
// actually left over. A flat 65% allocation is far too optimistic right
// near the apex, so the preview law computed a higher "safe" entry speed
// than the car could actually execute once it got there.
//
// REVERTED 9.5 -> 2.5 (2026-09-03, same day, user report: "we're not
// slowing down enough on the turn" -- directly confirms the above). 2.5 is
// not a fresh guess -- it's the LAST value this file has an actual live
// crash-avoidance confirmation for (see the 5.0->2.5 change above: 5.0
// produced a confirmed live clip/bump, chassis height 0.21m vs the normal
// 0.31m baseline, traced to this exact same over-optimistic-braking-budget
// failure mode). Every value tried above 2.5 since then (5.0, 9.5) has been
// an untested-live increase that, now twice, has produced under-braking
// symptoms once actually driven. Retune UP from here only after a live
// /cmd_ackermann trace confirms a specific higher value doesn't reproduce
// either symptom, not from theoretical tire-model math alone -- the
// STRUCTURAL gap this comment's own history describes (braking distance and
// cornering-speed target only coupled through POSITION, not a genuinely
// shared instantaneous friction budget) is still unfixed, and theoretical
// mu*g-based values will keep overestimating available braking near a
// corner's own apex until that's addressed directly.
// REPLACED with a real friction-circle budget (2026-09-04, user question:
// "why are we forced to use a deceleration value that's not even accurate
// to the actual car characteristics" -- asked right after a stopgap revert
// 8->2.5 for "we're not handling the breaking properly through the turn").
// Every value this constant was ever set to (2.5, 5.0, 8.0, 9.5) was a
// single FLAT number applied to every point in the preview window
// regardless of how much LATERAL (cornering) grip that point's own
// curvature already demands -- the STRUCTURAL gap this comment's own
// history kept pointing at but never actually fixed. A real tire's grip is
// one shared budget between braking and cornering (the friction circle),
// not two independent ones, so the true available braking AT a corner's
// own apex (where lateral demand is highest) is much less than the
// straight-line max, while away from any corner it's much MORE -- a flat
// constant can only ever be tuned for the worst case, which is why 2.5 is
// so much more conservative than this car's real capability (see
// kTireFrictionAccel below, ~14.7 m/s^2) and why every higher flat value
// failed live specifically near a corner.
//
// kTireFrictionAccel = mu*g, this sim's own actual tire model
// (simulation/models/fsd_car/model.sdf's wheel collision blocks, mu=mu2=
// 1.5 -- MUST stay in sync with it) -- the true combined lateral+
// longitudinal grip ceiling, not a guess.
// kPluginMaxDeceleration = this vehicle's own AckermannSteering plugin
// min_acceleration (model.sdf, -8 m/s^2 -- MUST stay in sync with it): a
// separate, tighter, PLUGIN-level cap on how hard a commanded deceleration
// is actually allowed to be, independent of how much tire grip theory
// alone would allow -- the friction-circle result below is clamped to this
// too, since the plugin won't execute more braking than this regardless.
// kFrictionCircleSafetyFactor: real tires don't achieve the textbook mu
// exactly under transient/dynamic load the way this sim's own repeated
// history of overestimating from theoretical tire-model math has already
// shown (see this comment's own history above, the 9.5 attempt) -- 0.6
// leaves a substantial, deliberately conservative margin below the
// friction-circle result, still a first, NOT YET LIVE-VALIDATED value
// (unlike 2.5, which had an actual confirmed-clean run) precisely because
// this whole approach is new; retune from an actual /cmd_ackermann trace
// through the hairpin, the same way every other value in this file has
// been, before trusting it the same way 2.5 was trusted.
//
// LOWERED 0.6 -> 0.45 (2026-09-08, user decision after a live-confirmed
// finding: /planning/debug_racing_line points measured 1.23-1.31m from
// real ground-truth cones at this track's sharp bend near (-27,-24) --
// inside kMinCarClearance=1.35m -- with this track's own real per-side
// margin budget confirmed only ~0.10-0.20m (path_utils.cpp's own
// EnforceMinTurnRadius comment). Root tradeoff, not a code bug: on a
// margin this thin, ordinary multi-stage-pipeline imprecision (spline fit
// -> corridor smoothing -> curvature-minimizing QP) can eat it without any
// single mechanism at fault. User's explicit choice was to keep
// kMinCarClearance/kCorridorSafetyMargin as-is (real physical buffer, not
// negotiable) and instead buy back margin by slowing down harder
// specifically where curvature is high, since THIS constant (shared
// between the cornering-speed law and the forward-preview braking law) is
// the one direct lever that reduces speed at sharp bends without touching
// any planning-side clearance constant. 0.45 is a meaningful, not
// extreme, tightening from the already-conservative 0.6 -- retune further
// only from a live /cmd_ackermann trace through this exact bend, not a
// second guess stacked on an unvalidated first one.
constexpr double kTireFrictionAccel = 1.5 * 9.81;   // m/s^2 (mu * g)
constexpr double kPluginMaxDeceleration = 8.0;      // m/s^2 (model.sdf min_acceleration magnitude)
constexpr double kFrictionCircleSafetyFactor = 0.45;

// kWheelBase/kSteeringLimit REMOVED (2026-09-04) -- were only used to
// convert curvature to a steering angle for the old steeringFraction-based
// speed law, now replaced entirely by speedForCurvature's direct grip-vs-
// curvature formula in Compute() (see its own comment there). The physical
// steering-limit constraint they encoded is still enforced independently,
// upstream, by path_generator.cpp's own kMinTurnRadius when the path is
// built -- it never depended on this file's now-removed local copy.

// Creep speed for an EMPTY path (see the early-return below) -- well under
// kMaxSpeed, cautious but nonzero. Confirmed directly as a real, fatal
// failure mode with the old "stop dead on empty path" behavior: a live
// full-lap test got permanently stuck (identical /estimated_pose and
// /cone_detections=pos=none-for-every-detection, frame after frame, for
// 100+ seconds straight) after every currently-visible cone happened to
// sit just outside lidar_projector.cpp's kMaxValidRange (20m) at the exact
// moment the path went empty. At speed=0 the car's own viewpoint never
// changes, so the SAME cones stay just out of range forever -- a permanent
// deadlock from one bad cycle, exactly the same failure class
// path_generator.cpp's SingleSideOffsetPath already fixed for the
// one-boundary-empty case ("With zero velocity, the car's own viewing
// angle never changes on the next frame either, so this was a permanent
// deadlock from a single bad frame") -- this is that same fix's
// counterpart for the BOTH-boundaries-empty case, which
// SingleSideOffsetPath's own guard (`left.empty() && !right.empty()`)
// deliberately doesn't cover.
//
// yawRate=0 (straight line, not curving toward a guess) is the same
// "car's own forward axis already approximates the local track direction
// closely enough over a short interval" reasoning SingleSideOffsetPath's
// own comment already relies on -- creeping forward a small distance is
// what's needed to bring previously-out-of-range cones back into view,
// not a real steering decision. BUT that reasoning only holds when the
// car's forward axis is ALREADY roughly aligned with the track -- confirmed
// directly as insufficient on its own by a second live stall (same full-lap
// test, further along): the car went empty-path with its heading nearly
// perpendicular to the track's actual direction (a sharp, likely
// over-corrected turn), and *170+ seconds of straight creep never
// recovered it* -- position AND heading both sat frozen, because driving
// straight from a badly-wrong heading just keeps facing the same wrong way
// forever, the exact same "one bad cycle becomes permanent" failure this
// whole mechanism exists to avoid, just for heading instead of position.
// See kSweepYawRate below for the escalation this added. 0.5 m/s is
// comfortably below kMinSpeed (1.0 m/s, the slowest a WORKING path ever
// commands) so a false-empty single cycle costs negligible ground, while a genuine
// multi-cycle gap (like the confirmed 100+ second stall) still recovers
// instead of stalling forever.
constexpr double kCreepSpeed = 0.5;  // m/s

// Minimum path length treated as "usable" -- below this, a NON-empty path
// is handled identically to a fully empty one (2026-09-04, user report:
// "we have somehow regressed the path plan for the hairpin again").
// Confirmed live at the hairpin: perception's own per-detection log showed
// several real cone detections repeatedly excluded ("box narrower than
// kMinBoxWidthForLocalization" -- a real, expected effect of the oblique
// viewing angles a sharp hairpin produces, not a bug in that filter
// itself), leaving only 1-2 usable cones and therefore a 1-2 point
// /planned_path -- NOT empty, so the empty-path exclusions elsewhere in
// this file (both the stuck-watchdog pause and the creep/sweep recovery
// trigger) never engaged, yet 1-2 points can't support a real curvature
// estimate (the forward-preview loop needs a point's own i-1/i+1
// neighbors) or meaningful forward progress either -- functionally
// equivalent to empty, just not literally zero. The car sat with
// essentially zero net displacement (confirmed via repeated live position
// samples) until the stuck-watchdog latched. 10 matches planning.cpp's own
// existing "thin path" diagnostic threshold for the closed-loop pipeline
// (same underlying idea, applied here where it actually gates behavior
// rather than just logging) -- comfortably below this pipeline's normal
// healthy path length (confirmed live at 26-30 points in a typical open-
// pipeline capture) so it only engages on a genuinely degraded cycle, not
// routine variation.
constexpr size_t kMinUsablePathPoints = 10;

// Escalation for kCreepSpeed's second failure mode (badly-wrong heading,
// not just "needs a few more meters of straight travel"): after this many
// CONSECUTIVE empty-path cycles, straight creep alone has had a fair,
// bounded chance (a few seconds at this pipeline's ~30-60Hz planning rate)
// and evidently isn't working, so a persistent turn gets added on top of
// the forward creep -- speed*yawRate traces a circular arc (turn radius =
// speed/kSweepYawRate), sweeping the car's own forward-facing sensors
// through a full 360 deg of heading (2*pi/kSweepYawRate =~ 21s at
// kCreepSpeed) rather than staring down the same wrong direction forever.
// Fixed direction (m_sweepDirection) for the DURATION of one sweep
// attempt -- never re-decided mid-sweep -- so consecutive empty cycles
// compound into one continuous search instead of jittering back and forth
// and covering no new heading at all. Always +1 currently (see its own
// header comment) -- the escalating-reverse-retry logic that used to flip
// it between attempts was removed 2026-09-02 along with the rest of the
// reverse-attempt code.
constexpr int kStraightCreepCycles = 60;
constexpr double kSweepYawRate = 0.3;  // rad/s

// A CONSTANT speed here (kCreepSpeed, unchanged from the sweep's own
// starting point) would trace a FIXED-radius circle (~1.7m at kCreepSpeed)
// forever, not a widening search -- confirmed directly as a real,
// insufficient-on-its-own escalation: a live full-lap test hit a THIRD
// stall (after the straight-creep fix resolved the first two) where
// /estimated_pose sat within a few CENTIMETERS of the same point for
// 300+ seconds while the logged detections stayed empty (no `pos=(` lines
// at all) the entire time -- the sweep was actively turning (heading kept
// changing) but endlessly retracing the same tiny circle, which happened
// to contain no cone anywhere on its circumference. Growing speed over the
// duration of the sweep -- yawRate held fixed -- widens that same circle
// into a genuine outward (Archimedean-like) spiral, so a sweep that keeps
// finding nothing eventually reaches ANY reachable cone regardless of how
// far outside the initial tight circle it sits, rather than searching the
// same small disk forever. Capped at kMaxSweepSpeed (well under kMaxSpeed
// -- see its own comment -- since this is still a blind search, not a
// confident, path-following command) so the spiral eventually levels off
// at a large-but-bounded radius instead of accelerating without limit.
// Growth rate sized for roughly one full tight loop (~21s) before the
// radius really starts opening up, giving the common, cheaper case (the
// answer was nearby, just missed by the initial heading) a real chance
// before committing to a wide search.
constexpr double kSweepSpeedGrowthPerCycle = 0.002;  // m/s added per swept cycle
constexpr double kMaxSweepSpeed = 2.0;               // m/s

// Even the widening spiral (kSweepSpeedGrowthPerCycle above) assumes the
// car can actually MOVE along whatever it commands -- confirmed directly
// as a real, un-covered failure mode: a live full-lap test found
// /cmd_ackermann continuously commanding forward speed (up to
// kMaxSweepSpeed, angular.z pinned at kSweepYawRate) while true
// /estimated_pose sat frozen at the same position for 100+ seconds, and
// the nearest ground-truth cone was 0.96m from the car -- physically
// wedged against it. No forward-only command (creep, sweep, or an ever-
// wider spiral) can ever produce real displacement from CONTACT.
//
// This is NOT scoped to the empty-path sweep alone, unlike an earlier
// version of this fix -- confirmed directly as insufficient by a SECOND
// live collision, at a different point in the same corner: /cmd_ackermann
// showed a completely ordinary, continuously-varying NORMAL pure-pursuit
// command (curvature-scaled yaw rate, speed up near kMaxSpeed) -- not the
// sweep's fixed kSweepYawRate -- while true position again sat frozen.
// planning's own boundary/path generation has no obstacle-awareness at
// all (see track_boundaries.hpp/path_generator.hpp), so a geometrically
// "reasonable" path can still walk the car directly into contact with a
// real cone in a tight enough corner. Every prior escalation in this file
// assumed "commanded speed didn't work" only ever meant "wrong direction"
// or "no path" -- neither covers "confidently driving into an obstacle" --
// so this stuck-check now runs FIRST, unconditionally, ahead of BOTH the
// empty-path and normal-path branches, and overrides either one the same
// way once triggered.
//
// RAISED 90 -> 270 (2026-09-05, user report: "midpoints calculation just
// disappeared all of a sudden, causing the planned path to just stop so
// the car stopped... unacceptable"). Confirmed live: at the moment the
// watchdog latched (logged at (-20.64,-14.75), matching the frozen car's
// own position exactly), a fresh capture taken only moments later showed
// /planning/debug_midpoints and /planned_path had ALREADY recovered on
// their own (7 healthy midpoints, a 17-point published path) -- the
// underlying data gap (perception's own near-field detection limitation
// at this hairpin, a separate, not-yet-fixed issue on its own) was
// genuinely TRANSIENT, and the existing recovery machinery (creep/sweep,
// this file's own thin-path exclusion) was already working -- it just
// didn't get enough real time to finish before this watchdog gave up and
// latched permanently, a one-way decision nothing downstream can undo.
// This constant's own comment above already had a stale assumption baked
// in ("~30Hz planning cycle... 90 cycles =~ 3s") -- this pipeline's real
// rate, confirmed directly via /timing/planning message counts earlier
// this same session, is closer to ~15Hz (Compute() only runs once per
// /planned_path message, not on a fixed timer -- see control.cpp's own
// subscription structure), so 90 cycles was already only ~6s in practice,
// not the ~3s the old comment claimed, and even that turned out to not be
// enough for this specific gap to resolve. 270 (~18s at the real ~15Hz
// rate) gives substantially more real time for a genuine, already-
// confirmed-working recovery to finish before giving up -- the
// independent LONG-TERM check below (kLongTermStuckWindow/
// kLongTermStuckThreshold) is unaffected and still catches a car that
// never actually recovers at all, so this isn't removing protection
// against a genuine physical block, only giving legitimate transient
// recovery enough time to actually happen first.
constexpr int kStuckCyclesBeforeLatch = 270;
// Below this net displacement (from the anchor -- see m_anchorX/Y's
// comment in the header for why it rolls forward on progress rather than
// staying fixed) after kStuckCyclesBeforeLatch, treated as "not actually
// moving" -- comfortably above ordinary EKF pose jitter, comfortably below
// the meaningful distance even kMinSpeed's normal-driving floor (let alone
// a working sweep) should cover in 3 seconds.
constexpr double kStuckDistanceThreshold = 0.3;  // meters

// Independent, LONGER-timescale companion to the short-term check above --
// see m_haveLongTermAnchor's own header comment for the confirmed live gap
// this closes: the short-term anchor's own "roll forward on any 0.3m
// crossing" design, correct for not penalizing real driving, is
// vulnerable to small noise/wobble (wheels spinning against a genuine
// physical block) randomly walking past 0.3m before kStuckCyclesBeforeLatch
// cycles ever elapse, resetting the counter forever. This check instead
// only samples position every kLongTermStuckWindow cycles (not
// continuously), so it can't be perpetually reset by noise -- only by
// ACTUAL displacement since the last snapshot. kLongTermStuckWindow and
// kLongTermStuckThreshold (comfortably more than one snapshot period's
// worth of noise, comfortably less than kMinSpeed's own normal-driving
// distance over that period) are both deliberately looser than the
// short-term check's own -- this is a slower-to-fire backstop for exactly
// the case the fast check structurally can't catch, not a replacement for
// it.
//
// RAISED 300 -> 900 (2026-09-05) alongside kStuckCyclesBeforeLatch's own
// same-day increase (90 -> 270, see its own comment for the confirmed live
// reason) -- 300 was only ~1.1x the new short-term threshold, which
// defeats this check's own stated purpose of being a MEANINGFULLY longer,
// looser backstop than the fast check; keeping roughly the original
// ~3.3x ratio (300/90) instead of letting the two converge preserves that
// each check still covers a genuinely different timescale/failure class,
// not two near-duplicate checks racing each other.
constexpr int kLongTermStuckWindow = 900;
constexpr double kLongTermStuckThreshold = 1.0;  // meters

// Startup grace period, in cycles, before EITHER stuck-watchdog check
// (short-term or long-term) is even evaluated -- confirmed live (2026-09-02)
// as a real, reproducible false-trigger: right after a fresh process start,
// the anchor above initializes on the FIRST valid pose, which arrives well
// before perception/planning have actually warmed up (YOLO's first real
// inference, the EKF's first few corrections, and the landmark pipeline's
// own readiness gate all take real wall-clock seconds) -- during that
// window the car is legitimately given nothing useful to follow yet, which
// looks identical to "physically blocked" to a watchdog that starts timing
// from pose-validity alone. Confirmed directly: the log showed "[STUCK] no
// forward progress for 91 cycles" fired at the car's own SPAWN position,
// well within kStuckCyclesBeforeLatch of process start -- and because the
// resulting m_permanentlyStuck latch never resets for the rest of this
// process's lifetime (see its own header comment), that single false
// trigger permanently disabled the car for the entire run, even after the
// upstream planning data the car was actually waiting on became healthy
// moments later. 300 cycles (~10s at 30Hz) matches kLongTermStuckWindow's
// own reasoning for a comfortably-longer-than-any-real-warm-up timescale --
// generous relative to a genuinely stuck car's own near-immediate real
// displacement once actually driving, so this doesn't mask a real physical
// block, only the brief window before the car has had a fair chance to
// start moving at all.
constexpr int kStartupGraceCycles = 300;
}  // namespace

// Algorithm (classic pure pursuit):
//   1. Pick the lookahead waypoint: the first path point at or beyond the
//      current lookahead distance (proportional to last cycle's speed,
//      see kLookaheadTimeConstant's own comment) from the car's own
//      origin (falls back to the farthest available point if none are
//      that far -- a sparse/short detected path shouldn't mean no command
//      at all).
//   2. curvature = 2*y / (x^2 + y^2) for a target at body-frame (x, y) --
//      the standard pure pursuit result for the circular arc through the
//      origin, heading along +X, that passes through the target. This is
//      purely geometric (depends only on the target's position, not speed),
//      so it's computed before speed.
//   3. Convert curvature to the steering angle it implies via the bicycle
//      model (atan(wheel_base * curvature)), and scale speed linearly
//      between kMaxSpeed (zero angle) and kMinSpeed (at/beyond
//      kSteeringLimit) -- see kMinSpeed's own comment for this scheme's
//      history.
//   4. yaw_rate = speed * curvature (curvature = yaw_rate / speed by
//      definition) -- using the SCALED speed from step 3, so the reported
//      yaw_rate stays consistent with the speed actually commanded.
DriveCommand PurePursuitController::Compute(const ControlInputs &inputs)
{
    // Counted unconditionally, ahead of every branch/early-return below, so
    // the startup grace period (see kStartupGraceCycles) elapses in real
    // time regardless of which branch executes each cycle.
    ++m_totalCycles;

    // Hardcoded straight-forward override for the first kStartupForwardCycles
    // cycles (2026-09-02, user report): at the very start of a lap, the
    // start/finish gate's orange cones have been observed misclassified as
    // yellow by perception, which the planning pipeline then treats as a
    // real boundary cone -- producing a badly-wrong early path that yanks
    // the car into a hard, incorrect left turn before it's gone anywhere.
    // Rather than try to fix the misclassification itself (a perception/
    // model problem, not a control one), unconditionally drive straight for
    // the first 2 seconds regardless of what the path says -- long enough
    // to clear the gate's immediate vicinity before trusting steering
    // input at all. Checked before EVERYTHING else, including the stuck-
    // watchdog below, since this is a known-safe fixed maneuver that should
    // never be preempted by it. kMinSpeed
    // (not kCreepSpeed) chosen deliberately: 2s at kCreepSpeed=0.5m/s only
    // covers 1m, likely not enough to actually clear the gate cones; 2s at
    // kMinSpeed=2.0m/s covers 4m, comfortably past them.
    constexpr int kStartupForwardCycles = 60;  // ~2s at this pipeline's ~30Hz cycle rate
    if (m_totalCycles <= kStartupForwardCycles)
    {
        return DriveCommand{kMinSpeed, 0.0};
    }

    // Checked before everything else, including the anchor/displacement
    // logic below -- see m_permanentlyStuck's own header comment for the
    // confirmed real bug this closes (without this early return, the
    // log-throttle reset a few lines down let the car resume full normal
    // driving for ~kStuckCyclesBeforeLatch cycles between each [STUCK]
    // pulse, forever, instead of actually holding position).
    if (m_permanentlyStuck)
    {
        return DriveCommand{0.0, 0.0};
    }

    // Skip both stuck checks entirely during the startup grace period (see
    // kStartupGraceCycles's own comment) -- m_haveAnchor/m_haveLongTermAnchor
    // simply stay false throughout, so the first cycle AFTER grace ends
    // initializes both anchors fresh from the then-current position, rather
    // than timing out against a stale anchor set back at process start.
    //
    // ALSO skip entirely while inputs.path is empty (2026-09-04, user
    // report: "hairpin section caused path planner to not produce valid
    // path" -- confirmed live at the hairpin, /cone_detections genuinely
    // had ZERO cones for a stretch, /planned_path correctly went empty,
    // and the car latched permanently stuck anyway). Root cause: this
    // block's own anchor/displacement tracking ran completely unconditional
    // on path availability -- gated only on poseValid -- despite this
    // function's OWN comment at the empty-path branch below explicitly
    // claiming these are two separate, non-interfering concerns
    // ("Recovering from a PHYSICAL block is the stuck-check above, not
    // this escalation -- that's a different problem"). In practice they
    // were never actually decoupled: kCreepSpeed=0.5 m/s is deliberately
    // slow (see its own comment), and a real acceleration ramp from a near-
    // stop at the exact moment a hairpin's own tight-cornering speed
    // floor transitions into an empty-path gap can plausibly fail to clear
    // kStuckDistanceThreshold (0.3m) within kStuckCyclesBeforeLatch (90
    // cycles/~3s) even though creep is legitimately, correctly recovering
    // -- the watchdog then misreads deliberate slow recovery as a physical
    // wedge and permanently latches, which the empty-path escalation (a
    // completely different, and in this case actually WORKING, recovery
    // mechanism) can never override once triggered (m_permanentlyStuck is
    // checked first, unconditionally, at the top of this function).
    // Pausing (not resetting) the anchor/cycle-counter here -- rather than
    // rolling it forward, which would be a free pass, or restarting it
    // from a fresh position, which could hide a genuine block that started
    // just before the gap -- means an empty-path stretch neither helps nor
    // hurts: the SAME progress-tracking window picks back up exactly where
    // it left off once real path data resumes, so a genuine physical block
    // that happens to coincide with a data gap is still caught once
    // tracking resumes, just not falsely blamed on the gap itself.
    if (inputs.poseValid && m_totalCycles > kStartupGraceCycles
        && inputs.path.size() >= kMinUsablePathPoints)
    {
        if (!m_haveAnchor)
        {
            m_anchorX = inputs.worldX;
            m_anchorY = inputs.worldY;
            m_haveAnchor = true;
            m_cyclesSinceAnchor = 0;
        }
        else
        {
            ++m_cyclesSinceAnchor;
            const double dx = inputs.worldX - m_anchorX;
            const double dy = inputs.worldY - m_anchorY;
            const double displacement = std::sqrt(dx * dx + dy * dy);
            if (displacement >= kStuckDistanceThreshold)
            {
                // Real progress -- roll the measurement window forward
                // rather than accumulating against an increasingly stale
                // start point.
                m_anchorX = inputs.worldX;
                m_anchorY = inputs.worldY;
                m_cyclesSinceAnchor = 0;
            }
            else if (m_cyclesSinceAnchor > kStuckCyclesBeforeLatch)
            {
                // SAFETY OVERRIDE (2026-08-23): see this function's top
                // comment -- reverse is disallowed, so this latches a hard
                // stop instead of attempting any recovery maneuver. Loudly
                // logged to stderr (unbuffered,
                // unlike stdout here -- see perception.cpp's own per-
                // detection prints for why an unflushed stdout buffer can
                // sit unseen for minutes) specifically so an external
                // watcher can detect this and reset the sim -- a stuck car
                // that cannot reverse has no other way out.
                std::fprintf(stderr,
                    "[STUCK] no forward progress for %d cycles at approx (%.2f,%.2f) -- "
                    "reverse recovery is DISABLED (FS rules); holding position permanently, needs external reset\n",
                    m_cyclesSinceAnchor, inputs.worldX, inputs.worldY);
                // Latch permanently stopped -- see m_permanentlyStuck's own
                // header comment. The old `m_cyclesSinceAnchor = 0` reset
                // that used to live here was meant only to throttle
                // repeated LOG spam, but it also silently let the state
                // machine resume normal driving for the next ~
                // kStuckCyclesBeforeLatch cycles before re-triggering --
                // now handled by the early-return latch check at the top
                // of this function instead, so this trigger only needs to
                // fire once.
                m_permanentlyStuck = true;
                return DriveCommand{0.0, 0.0};
            }
        }
    }
    // else: either no pose yet (startup, stuck-detection just doesn't run
    // until a pose arrives, well before this could matter in practice), or
    // inputs.path is too thin to drive on (see this condition's own comment
    // above and kMinUsablePathPoints's own declaration comment for why
    // that's now excluded the same way an empty path always was) -- either
    // way, falls through to the thin/empty-path branch below when
    // applicable.

    // Independent long-term check -- see kLongTermStuckWindow/
    // kLongTermStuckThreshold's own comment for why the short-term check
    // above can miss a real stuck condition (noise/wobble perpetually
    // resetting its rolling anchor). Snapshots position only once every
    // kLongTermStuckWindow cycles, so only genuine displacement since the
    // last snapshot -- not per-cycle noise -- can ever reset it.
    //
    // DELIBERATELY NOT excluded during a thin path, unlike the short-term
    // check above (2026-09-05, confirmed live: the car sat frozen at the
    // hairpin for 10+ minutes with ZERO stuck-watchdog triggers -- the
    // short-term check's own thin-path exclusion, while correctly fixing
    // the false-positive it was built for, had the side effect of pausing
    // BOTH checks together, since they used to share one gating condition).
    // If a thin/degraded path condition never actually clears at some
    // location (persistent perception starvation, not a brief recoverable
    // gap), creep/sweep can end up not making genuine progress either --
    // and with the short-term check paused the whole time, nothing was
    // left to ever catch that. This check's own ~10s window, sampling only
    // net displacement (not per-cycle path health), still needs a pose and
    // still respects the startup grace period, but runs regardless of path
    // length -- it doesn't care WHY the car isn't moving, only THAT it
    // isn't, over a long enough window that legitimate slow creep/sweep
    // recovery (which does cover real ground, just slowly) won't
    // false-trigger it the way the short-term check's shorter window could.
    if (inputs.poseValid && m_totalCycles > kStartupGraceCycles)
    {
        if (!m_haveLongTermAnchor)
        {
            m_longTermAnchorX = inputs.worldX;
            m_longTermAnchorY = inputs.worldY;
            m_haveLongTermAnchor = true;
            m_cyclesSinceLongTermAnchor = 0;
        }
        else
        {
            ++m_cyclesSinceLongTermAnchor;
            if (m_cyclesSinceLongTermAnchor >= kLongTermStuckWindow)
            {
                const double ldx = inputs.worldX - m_longTermAnchorX;
                const double ldy = inputs.worldY - m_longTermAnchorY;
                const double longTermDisplacement = std::sqrt(ldx * ldx + ldy * ldy);
                if (longTermDisplacement < kLongTermStuckThreshold)
                {
                    std::fprintf(stderr,
                        "[STUCK-LONGTERM] under %.2fm net displacement over %d cycles at approx "
                        "(%.2f,%.2f) -- reverse recovery is DISABLED (FS rules); holding position "
                        "permanently, needs external reset\n",
                        kLongTermStuckThreshold, m_cyclesSinceLongTermAnchor, inputs.worldX, inputs.worldY);
                    m_permanentlyStuck = true;
                    return DriveCommand{0.0, 0.0};
                }
                // Real net progress over the window -- snapshot forward.
                m_longTermAnchorX = inputs.worldX;
                m_longTermAnchorY = inputs.worldY;
                m_cyclesSinceLongTermAnchor = 0;
            }
        }
    }

    if (inputs.path.size() < kMinUsablePathPoints)
    {
        // Too few points to drive on this cycle (empty, or non-empty but
        // too thin -- see kMinUsablePathPoints's own comment) -- creep, and
        // if this persists, sweep in a widening spiral, rather than stop
        // dead. See kCreepSpeed's comment for why a hard stop here is a
        // confirmed permanent-deadlock bug, not a safe default;
        // kSweepYawRate's for why straight creep alone isn't always
        // enough; and kSweepSpeedGrowthPerCycle's for why a fixed-speed
        // sweep isn't either. (Recovering from a PHYSICAL block is the
        // stuck-check above, not this escalation -- that's a different
        // problem: "not enough usable path data", not "detections exist but
        // something's in the way".)
        ++m_consecutiveEmptyCycles;
        if (m_consecutiveEmptyCycles <= kStraightCreepCycles)
        {
            return DriveCommand{kCreepSpeed, 0.0};
        }
        const int sweepCycles = m_consecutiveEmptyCycles - kStraightCreepCycles;
        const double sweepSpeed = std::min(kMaxSweepSpeed,
                                            kCreepSpeed + kSweepSpeedGrowthPerCycle * sweepCycles);
        return DriveCommand{sweepSpeed, m_sweepDirection * kSweepYawRate};
    }
    m_consecutiveEmptyCycles = 0;
    m_sweepDirection = 1.0;  // fresh baseline for whatever the NEXT stuck episode is

    // DIAGNOSTIC TOGGLES (2026-09-01): were used to bisect a confirmed-real
    // commanded-speed oscillation by stripping each recent addition out one
    // at a time. Left all-false for several live runs afterward, which
    // means every one of those runs -- including the recurring hairpin
    // stuck investigation at (-21.7,-11.2) -- was accidentally testing the
    // STRIPPED-DOWN scheme (fixed lookahead, no forward-preview braking, no
    // smoothing), not the tuned one. Re-enabled (2026-09-02) since forward-
    // preview braking specifically exists to handle this exact class of
    // tight corner (see its own comment below) and was silently disabled
    // this whole time. The oscillation bisection itself was never
    // conclusively finished -- if it recurs, flip these back individually
    // rather than assuming re-enabling all three is safe by default.
    constexpr bool kEnableAdaptiveLookahead = false;
    constexpr bool kEnableForwardPreviewBraking = true;
    constexpr bool kEnableSpeedSmoothing = true;
    constexpr double kFixedLookaheadDistance = 3.0;  // meters, used only when adaptive lookahead is off

    // CLOSED-LOOP LOOKAHEAD CEILING (2026-09-05, user report: "we're
    // oscillating around the planned path" -- the first time this session
    // the car has sustained real closed-loop speed (kMaxSpeed=18.0) long
    // enough to expose this). Root cause: the FINAL clamp on lookaheadDistance
    // below (std::clamp(..., kMinLookahead, kMaxLookahead)) pins the real,
    // in-use lookahead to EXACTLY kMaxLookahead whenever kMinLookahead==
    // kMaxLookahead (currently true, both 2.0) -- regardless of
    // baseLookaheadDistance's own value, adaptive or fixed. kMaxLookahead
    // was pinned to 2.0 by explicit user request (see its own history) to
    // fix hairpin corner-cutting on the OPEN pipeline at ~5 m/s, where 2.0m
    // is ~0.4s of reaction time -- reasonable. At kMaxSpeed=18.0 on the
    // CLOSED loop, that same 2.0m is only ~0.11s -- far too short a horizon
    // for the geometric pure-pursuit target to change smoothly cycle to
    // cycle, the exact "steering feels lagged/oscillating" mechanism this
    // file's own kLookaheadTimeConstant comment already documents. Applying
    // ONE ceiling to both regimes was never re-examined once the car
    // actually started reaching closed-loop speed. Split it the same way
    // kFirstLapMaxSpeed/kMaxSpeed already split the speed ceiling itself:
    // the open pipeline keeps EXACTLY 2.0m (the user's own request,
    // untouched), the closed loop gets a larger ceiling. 6.0m is a first,
    // not yet live-validated guess (6/18=0.33s, ~3x the effective 2.0m
    // today) -- not the naive 1:1 kLookaheadTimeConstant*kMaxSpeed=18.0m,
    // since the corridor here measures as narrow as ~1.1-1.4m combined in
    // places and an 18m straight-line lookahead risks corner-cutting on any
    // bend the curvature-based cap below doesn't fully catch in time.
    // curvatureBasedLookaheadCap's own scan window and break distance are
    // widened to match (see below) so it can still see and shrink for a
    // bend anywhere in this new, larger reach -- it only ever SHRINKS the
    // lookahead, never grows it, so this raise can't reintroduce the
    // corner-cutting the 2.0m pin was originally set to prevent.
    constexpr double kClosedLoopMaxLookahead = 6.0;  // meters
    const bool isClosedLoopPath = inputs.path.size() >= kClosedLoopPathSizeThreshold;
    const double effectiveMaxLookahead = isClosedLoopPath ? kClosedLoopMaxLookahead : kMaxLookahead;

    // Lookahead distance for THIS cycle, from LAST cycle's speed -- see
    // kLookaheadTimeConstant's own comment for why proportional-to-speed,
    // and m_lastSpeed's own comment in the header for why one cycle
    // delayed.
    const double baseLookaheadDistance = kEnableAdaptiveLookahead
        ? std::clamp(kLookaheadTimeConstant * m_lastSpeed, kMinLookahead, effectiveMaxLookahead)
        : kFixedLookaheadDistance;

    // Curvature-based lookahead CAP, on top of the speed-based distance
    // above (2026-09-04, user report: "we've got to make the debug_target
    // closer to the car for curves, or else we will hit the cones"). Pure
    // pursuit's own geometry is the direct mechanism here: the target is
    // reached by a circular arc from the car's current pose, and a
    // lookahead distance that's long relative to the TRUE local turn radius
    // makes that arc's chord cut inside the actual path through a bend --
    // exactly a corner-cut into whatever's on the inside of the turn. The
    // base distance above only accounts for speed (or is simply fixed,
    // kEnableAdaptiveLookahead is currently off), with no awareness of the
    // upcoming path's own shape at all -- a straight approach and a tight
    // bend get the identical lookahead at the same speed today.
    // Scans the SAME near-field the reactive law's own target search will
    // use (ordered nearest-first, so this can safely break once past
    // kMaxLookahead rather than scanning the whole published path) with the
    // same local (Menger) curvature calculation the forward-preview
    // braking loop below already uses -- a property of the path's shape,
    // not of the car's current heading -- and caps the lookahead to
    // kLookaheadRadiusFraction of the TIGHTEST local turn radius found
    // there. 0.6 is a first, not yet live-validated value: small enough
    // that the chord deviation from the true arc stays modest relative to
    // the turn radius, large enough that this doesn't collapse to
    // kMinLookahead on every gentle curve the racing line's own widening
    // already handles fine. Only ever SHRINKS the lookahead relative to the
    // speed-based distance (std::min below), never grows it -- a straight
    // section still gets the full speed-based distance exactly as before.
    constexpr double kLookaheadRadiusFraction = 0.6;
    double curvatureBasedLookaheadCap = effectiveMaxLookahead;
    for (size_t i = 1; i + 1 < inputs.path.size(); ++i)
    {
        const auto &wp = inputs.path[i];
        const double distSq = wp.x * wp.x + wp.y * wp.y;
        if (distSq > effectiveMaxLookahead * effectiveMaxLookahead)
        {
            break;
        }
        const auto &prev = inputs.path[i - 1];
        const auto &next = inputs.path[i + 1];
        const double abx = wp.x - prev.x;
        const double aby = wp.y - prev.y;
        const double acx = next.x - prev.x;
        const double acy = next.y - prev.y;
        const double lenAB = std::hypot(abx, aby);
        const double lenBC = std::hypot(next.x - wp.x, next.y - wp.y);
        const double lenCA = std::hypot(prev.x - next.x, prev.y - next.y);
        const double denom = lenAB * lenBC * lenCA;
        const double localCurvature = denom > 1e-6 ? (2.0 * std::abs(abx * acy - aby * acx) / denom) : 0.0;
        if (localCurvature > 1e-6)
        {
            const double localRadius = 1.0 / localCurvature;
            curvatureBasedLookaheadCap = std::min(curvatureBasedLookaheadCap, kLookaheadRadiusFraction * localRadius);
        }
    }
    const double lookaheadDistance =
        std::clamp(std::min(baseLookaheadDistance, curvatureBasedLookaheadCap), kMinLookahead, effectiveMaxLookahead);

    // inputs.path is already sorted nearest-ahead-first (see planning.cpp).
    double targetX = 0.0, targetY = 0.0;
    bool found = false;
    for (const auto &wp : inputs.path)
    {
        if (wp.x * wp.x + wp.y * wp.y >= lookaheadDistance * lookaheadDistance)
        {
            targetX = wp.x;
            targetY = wp.y;
            found = true;
            break;
        }
    }
    if (!found)
    {
        const auto &last = inputs.path.back();
        targetX = last.x;
        targetY = last.y;
    }

    const double lookaheadSq = targetX * targetX + targetY * targetY;
    const double curvature = lookaheadSq > 1e-6 ? (2.0 * targetY / lookaheadSq) : 0.0;

    // See kFirstLapMaxSpeed's own comment -- caps the speed law's ceiling
    // during the open/first-lap pipeline, before the validated closed loop
    // takes over.
    const double effectiveMaxSpeed =
        inputs.path.size() < kClosedLoopPathSizeThreshold ? kFirstLapMaxSpeed : kMaxSpeed;

    // Grip-based cornering speed limit, replacing the old steeringAngle/
    // steeringFraction-based LINEAR scheme entirely (2026-09-04, user
    // report: "still going too fast around the corners causing us to skid
    // out past the planned path and crash into cone" -- reported right
    // after the braking-distance budget just above was made friction-
    // circle-aware, which exposed that the CORNERING speed law itself was
    // never grip-based at all: steeringFraction is a linear function of
    // steering ANGLE, which has no direct physical relationship to how much
    // lateral acceleration a given speed+curvature combination actually
    // demands of the tires (that relationship is a_lat = v^2 * curvature,
    // not linear in angle). A corner could be commanded at a speed well
    // past what its own curvature can hold laterally while still reading as
    // a "moderate" steeringFraction, which is exactly a skid-out.
    // v = sqrt(a_lat_max / curvature) is the textbook max speed for a given
    // curvature at a given lateral grip budget -- reusing the SAME
    // kTireFrictionAccel/kFrictionCircleSafetyFactor budget as the braking
    // model above (one shared assumed-grip model, not two unrelated
    // schemes). curvature ~0 has no lateral demand at all (effectiveMaxSpeed
    // instead of a divide-by-near-zero blowup); kMinSpeed remains a hard
    // floor -- see its own comment -- for corners tight enough that even
    // this budget can't be held at any practical speed. The corridor-based
    // racing line already widens real corners well past raw geometric
    // centerline curvature specifically to avoid needing this floor often;
    // if it engages constantly post-optimization, that's itself a sign the
    // racing line isn't achieving the widening it's supposed to, not a
    // reason to loosen this formula. NOT YET LIVE-VALIDATED, same as the
    // braking budget above -- retune kFrictionCircleSafetyFactor (shared
    // between both) from an actual /cmd_ackermann trace through the hairpin
    // before trusting this the way the old scheme eventually was.
    const auto speedForCurvature = [effectiveMaxSpeed](double _curvature) -> double
    {
        const double absCurvature = std::abs(_curvature);
        if (absCurvature < 1e-6)
        {
            return effectiveMaxSpeed;
        }
        const double gripLimitedSpeed =
            std::sqrt(kFrictionCircleSafetyFactor * kTireFrictionAccel / absCurvature);
        return std::clamp(gripLimitedSpeed, kMinSpeed, effectiveMaxSpeed);
    };

    // Forward-preview braking (2026-09-01): the scheme above only reacts to
    // the SINGLE lookahead target's own curvature, which is the current
    // speed's worth of distance ahead (~kLookaheadTimeConstant seconds) --
    // fine at 5.0 m/s, but confirmed live insufficient once pushed to 8.0:
    // the car showed a bump-like chassis-height artifact entering this
    // track's sharp hairpin specifically, arriving too fast to have already
    // scrubbed down to kMinSpeed by the time the reactive target's own
    // curvature caught up with the turn actually being that tight. Scanning
    // every published waypoint out to kBrakePreviewDistance (not just the
    // one reactive target) gives the car advance warning of an upcoming
    // tight corner while it's still comfortably far enough away to actually
    // brake for it.
    //
    // DISTANCE-AWARE REWRITE (2026-09-02, user report: "monitor the speed,
    // see if we're actually hitting the max" -- confirmed live it almost
    // never did, ceiling ~12.7 m/s of kMaxSpeed=15.0 over several minutes of
    // continuous driving, average ~6.5 m/s). Root cause: the ORIGINAL
    // version took the sharpest curvature found ANYWHERE in the window and
    // applied its full speed penalty AS IF that corner were right at the
    // car's nose, with zero regard for how far away it actually was -- so a
    // real corner 9.9m out (right at the edge of the old 10.0m window)
    // pinned the speed exactly as hard as one 0.1m out. On this track's
    // real corner spacing that's very rarely NOT true somewhere in a 10m
    // window, which is exactly why kBrakePreviewDistance had to be shrunk
    // from 20.0 to 10.0 on 2026-09-02 in the first place -- a workaround
    // for the mechanism being distance-blind, not a fix to it.
    //
    // Replaced with an actual braking-distance model: for each waypoint,
    // ask "what is the fastest the car could be going RIGHT NOW and still
    // be able to decelerate (at this point's own friction-circle-derived
    // availableDeceleration, see below) down to that waypoint's own
    // curvature-implied target speed by the time it gets there" -- from
    // v0^2 = v^2 + 2*a*d (rearranged from the standard
    // constant-deceleration kinematics v^2 = v0^2 - 2*a*d), so a sharp
    // corner far away only requires the car to ALREADY be decelerating by
    // now, not to already be at the corner's own slow speed. The tightest
    // (smallest) such bound across the whole window is the answer -- still
    // only ever pulls speed DOWN relative to the reactive law (never up),
    // preserving the original safety property, but no longer conflates
    // "a hazard exists somewhere in the window" with "the hazard is here
    // now".
    double previewSpeed = effectiveMaxSpeed;
    if (kEnableForwardPreviewBraking)
    {
        // BUG FIX (2026-09-03): this used to `continue` past any point
        // farther than kBrakePreviewDistance and keep scanning the REST of
        // inputs.path -- fine when inputs.path was always a short forward-
        // only window (every point genuinely on the car's own upcoming
        // route), but planning.cpp now publishes the FULL closed loop on
        // /planned_path (2026-09-03, user request), which necessarily
        // includes the loop's own closing seam (its last points are
        // geometrically right back next to its first, since it's a closed
        // loop) and any other spot where the track happens to pass close to
        // itself. A `continue`-based scan doesn't distinguish "genuinely
        // the car's own near-term path" from "some other, unrelated point
        // that just happens to be geographically nearby" -- confirmed live:
        // a point at the tail of the array, body-frame (-0.60,0.22) (the
        // loop's own seam, not an upcoming corner), produced a 58.7-degree
        // implied steering angle and pinned speed at kMinSpeed continuously
        // once the closed-loop pipeline activated. `break` on the first
        // out-of-range point instead: inputs.path is ordered starting at
        // the car's current position and walking forward continuously, so
        // once a point is farther than the preview distance, everything
        // after it in the array is either farther still (irrelevant to
        // this preview) or a later, unrelated point -- never something
        // that needs to be skipped-past-and-continued.
        // Points closer than kPreviewMinDistance are SKIPPED (not treated
        // as zero curvature, not broken-out-of -- just not trusted for a
        // braking decision) -- their own braking-distance term (`dist`
        // below) is tiny regardless, so they barely constrain previewSpeed
        // either way. kMinLookahead (2.0m) is the same threshold the
        // reactive law's own target search already treats as the closest
        // sensible lookahead, reused here for consistency.
        //
        // CURVATURE FIX (2026-09-03, user report: "we should be calculating
        // the curvature of the arc at the local path, not relative to the
        // current car's position"). The old `2*wp.y/distSq` formula is the
        // correct pure-pursuit result for "what arc must the car drive
        // RIGHT NOW, from its current position and heading, to pass through
        // this point" -- exactly right for the REACTIVE law just below
        // (reactiveSpeed/speedForCurvature), which really is asking that
        // question about the single lookahead target. But this preview loop
        // is asking a DIFFERENT question for each waypoint: "how sharp is
        // the TRACK ITSELF at this future point" -- and those aren't the
        // same thing. Early in a bend, a waypoint just past the entry can
        // still have a small y relative to its distance from the car's
        // current heading (the car hasn't turned into the bend yet), so the
        // old formula underestimates how tight the corner actually is until
        // the car is already close -- silently defeating the whole point of
        // advance/preview braking for exactly the corners it exists to
        // catch. Replaced with the actual local (Menger) curvature of the
        // path at each point, from that point's own neighbors in the
        // already-ordered inputs.path array -- purely a property of the
        // path's shape there, independent of where the car currently is or
        // which way it's currently pointed.
        constexpr double kPreviewMinDistance = kMinLookahead;
        for (size_t i = 1; i + 1 < inputs.path.size(); ++i)
        {
            const auto &wp = inputs.path[i];
            const double distSq = wp.x * wp.x + wp.y * wp.y;
            if (distSq > kBrakePreviewDistance * kBrakePreviewDistance)
            {
                break;
            }
            if (distSq < kPreviewMinDistance * kPreviewMinDistance)
            {
                continue;
            }
            const auto &prev = inputs.path[i - 1];
            const auto &next = inputs.path[i + 1];
            const double abx = wp.x - prev.x;
            const double aby = wp.y - prev.y;
            const double acx = next.x - prev.x;
            const double acy = next.y - prev.y;
            const double lenAB = std::hypot(abx, aby);
            const double lenBC = std::hypot(next.x - wp.x, next.y - wp.y);
            const double lenCA = std::hypot(prev.x - next.x, prev.y - next.y);
            const double denom = lenAB * lenBC * lenCA;
            const double wpCurvature =
                denom > 1e-6 ? (2.0 * std::abs(abx * acy - aby * acx) / denom) : 0.0;
            const double wpTargetSpeed = speedForCurvature(wpCurvature);
            // Friction-circle braking budget: how much LONGITUDINAL decel is
            // left over once wpTargetSpeed's own lateral demand at this
            // point's curvature is subtracted from the shared tire grip
            // circle, clamped to the plugin's own hard command cap -- see
            // this file's kTireFrictionAccel/kPluginMaxDeceleration/
            // kFrictionCircleSafetyFactor comment above for the full
            // reasoning (this replaces the old flat kMaxPreviewDeceleration
            // constant entirely).
            const double aLatAtTarget = wpTargetSpeed * wpTargetSpeed * wpCurvature;
            const double aLonFromTireBudget = std::sqrt(std::max(
                0.0, kTireFrictionAccel * kTireFrictionAccel - aLatAtTarget * aLatAtTarget));
            const double availableDeceleration =
                kFrictionCircleSafetyFactor * std::min(aLonFromTireBudget, kPluginMaxDeceleration);
            const double dist = std::sqrt(distSq);
            const double allowedSpeed =
                std::sqrt(wpTargetSpeed * wpTargetSpeed + 2.0 * availableDeceleration * dist);
            previewSpeed = std::min(previewSpeed, allowedSpeed);
        }
    }

    const double reactiveSpeed = speedForCurvature(curvature);
    const double rawSpeed = std::min(reactiveSpeed, previewSpeed);

    // Smoothing (2026-09-01): confirmed live as necessary once
    // kBrakePreviewDistance's forward-preview scan was added -- the raw
    // min() above is a max-over-window statistic recomputed from scratch
    // every cycle against the reactive pipeline's own freshly-recomputed
    // path, so a single momentarily-noisy waypoint (ordinary landmark
    // jitter, not a real corner) can yank rawSpeed down hard for one cycle
    // and let it bounce back the next -- confirmed directly via
    // /control/debug_target: the published target's own distance from the
    // car swinging 2.3-5.1m cycle to cycle, a direct, visible symptom
    // since lookahead distance is proportional to m_lastSpeed. A light EMA
    // damps that single-cycle noise while still tracking a REAL, sustained
    // speed change (entering/exiting an actual corner) within a handful of
    // cycles -- kSpeedSmoothingAlpha=0.3 gives a settling time constant of
    // roughly 3 cycles (~0.3-0.4s at this pipeline's ~9-10Hz rate), fast
    // enough to still brake for a genuine tight corner in time, slow
    // enough to reject a one-cycle blip. m_lastSpeed starts at 0.0 (see
    // its own header comment) but has several cycles to converge during
    // control.cpp's own 3s startup hold before any command actually
    // matters, so the cold-start bias this could otherwise cause is a
    // non-issue in practice.
    constexpr double kSpeedSmoothingAlpha = 0.3;
    const double speed = kEnableSpeedSmoothing
        ? kSpeedSmoothingAlpha * rawSpeed + (1.0 - kSpeedSmoothingAlpha) * m_lastSpeed
        : rawSpeed;
    const double rawYawRate = speed * curvature;

    // EMA smoothing for yawRate (2026-09-04, user report: "a lot of jitter
    // in the planned path that's causing oscillations") -- see
    // m_lastYawRate's own header comment for the root cause this addresses
    // (pose-estimation noise baked into the body-frame path every cycle,
    // which can flip the reactive target and swing raw curvature even when
    // the underlying world-frame line hasn't moved).
    //
    // LOWERED 0.3 -> 0.15 (2026-09-05, user report: "the planned path
    // updated too quickly at the hairpin, causing the control algorithm to
    // allow the car to keep veering out wide and hit a cone"). This is
    // exactly the "retune independently from kSpeedSmoothingAlpha if
    // steering ends up needing a different settling time" case this
    // constant's own history already flagged as untested. The open
    // pipeline's own racing line is recomputed completely FRESH every
    // cycle with no cross-cycle blending at all (see planning.cpp's own
    // "No cross-cycle blending" comment -- an earlier position-keyed cache
    // was removed after it caused a DIFFERENT, worse bug, grid-cell
    // collisions scrambling point order), so at a difficult, sparse-data
    // section like the hairpin, the published line's own shape can
    // genuinely shift from cycle to cycle, not just its pose-noise
    // projection into body frame. At alpha=0.3 (~3-cycle settling time),
    // the commanded steering was tracking those cycle-to-cycle plan
    // changes closely enough to follow a still-settling, not-yet-accurate
    // shape out wide before the plan itself caught up to hugging the real
    // apex. 0.15 (~6-cycle settling time) trades a bit more steering lag
    // for genuine corners (still fast enough at this pipeline's own ~15Hz
    // rate -- roughly 0.4s to settle, well within a hairpin's own multi-
    // second transit time) for meaningfully more damping against a plan
    // that hasn't stabilized yet. kSpeedSmoothingAlpha is left at 0.3,
    // unchanged -- the user's own report was specifically about steering/
    // veering, not speed, and there's no live evidence yet that speed
    // needs the same change.
    //
    // Applied unconditionally, NOT gated on kEnableSpeedSmoothing -- that
    // flag is one of this file's own diagnostic bisection toggles for a
    // DIFFERENT, already-resolved oscillation investigation (see its own
    // comment above); coupling this new, independent addition to it would
    // make future bisection harder to reason about, not easier.
    constexpr double kYawRateSmoothingAlpha = 0.15;
    const double yawRate =
        kYawRateSmoothingAlpha * rawYawRate + (1.0 - kYawRateSmoothingAlpha) * m_lastYawRate;

    m_lastSpeed = speed;  // for NEXT cycle's lookahead distance
    m_lastYawRate = yawRate;
    return DriveCommand{speed, yawRate, targetX, targetY};
}
