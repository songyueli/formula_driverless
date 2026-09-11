#include "centerline_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "path_utils.hpp"

namespace fsd
{
namespace
{
// Index-window radius for ClosedLoopMidpointExtractor's cursor-advancing
// yellow search -- same scale as corridor.cpp's kChainSearchWindow, same
// role: wide enough to absorb ordinary local density differences between
// the two chains (blue and yellow cone spacing isn't perfectly matched
// index-for-index), narrow enough that it can't jump across a hairpin
// fold-back to a wrong-but-nearby candidate.
constexpr int kClosedChainSearchWindow = 8;
// Hop-distance cap for ClosedLoopMidpointExtractor's own per-color chain
// ordering -- see OrderWaypointsByTraversal's own _maxHopDistance comment
// for why this needs to be larger than the default kMaxPairDistance
// (8.0m) at full-track scale. 10.5m (a first attempt, tuned against ONE
// live capture) turned out NOT to be a stable value: confirmed directly
// (2026-08-31) against a SECOND live capture that the real gap size
// varies run to run -- it's not genuine wide cone spacing, it's
// perception occasionally missing a cone's detection/localization
// somewhere along the drive, and WHICH cone gets missed isn't
// deterministic. That second capture needed ~15m before the blue chain
// closed (104/106 landmarks; 10.5m only reached 13/106). Raised to 15.0m
// as a more robust margin past both measured requirements so far, still
// well short of a "no cap at all" value that would reintroduce real
// hairpin-fold-back-hop risk (see OrderWaypointsByTraversal's own
// comment) -- if a future run needs more than this, that's a sign the
// underlying fragility (a hard distance cap trying to distinguish "missed
// cone" from "wrong hairpin leg" using only distance) needs a structural
// fix, not another bump.
constexpr double kClosedChainMaxHop = 15.0;  // meters

// Longitudinal-alignment window for BOTH extractors' own blue-to-yellow
// pairing (2026-09-06/08) -- see ClosedLoopMidpointExtractor's own comment
// at its use for the full derivation (a real blue/yellow cone-spacing
// mismatch, confirmed ~1.85m vs ~2.15m at the start/finish straight, makes
// raw-nearest-Euclidean pairing flip-flop between the yellow cone ahead
// and the one behind as the blue index advances -- a genuine zigzag, not
// noise). Promoted to file scope (2026-09-08) so TwoPointMidpointExtractor
// can share the exact same criterion instead of drifting from a second,
// hand-copied constant -- see TwoPointMidpointExtractor's own comment for
// why its own pairing needed the identical fix.
constexpr double kPairLongitudinalWindow = 3.0;  // meters

// Gate-anchor midpoints from orange cones alone (2026-09-04) -- see both
// extractors' own header comments for why this exists: blue/yellow
// coverage has a real, multi-meter hole running THROUGH a start/finish
// gate, and neither extractor previously had any orange-awareness to fill
// it. Same mutual-nearest-neighbor requirement as the blue-yellow pairing
// below, applied to orange against itself -- on this track that's just the
// one gate (2 cones, 1 pair), but this generalizes to multiple gates
// without change. kMaxPairDistance reused as the same "plausibly related"
// distance bound every other pairing in this file already uses, not a new
// orange-specific magic number.
std::vector<PathPoint> OrangeGateMidpoints(const std::vector<WorldCone> &orange)
{
    std::vector<PathPoint> gateMidpoints;
    for (size_t i = 0; i < orange.size(); ++i)
    {
        const WorldCone &a = orange[i];
        double bestDistSq = kMaxPairDistance * kMaxPairDistance;
        size_t bestJ = orange.size();
        for (size_t j = 0; j < orange.size(); ++j)
        {
            if (i == j)
            {
                continue;
            }
            const double dx = orange[j].x - a.x;
            const double dy = orange[j].y - a.y;
            const double distSq = dx * dx + dy * dy;
            if (distSq < bestDistSq)
            {
                bestDistSq = distSq;
                bestJ = j;
            }
        }
        if (bestJ == orange.size() || bestJ <= i)
        {
            continue;  // no partner in range, or already emitted from the other direction
        }
        // Mutual check, same reasoning as blue/yellow: a's own nearest
        // orange must be bestJ AND bestJ's own nearest orange must be a,
        // or this isn't really a gate pair (e.g. a third stray orange
        // cone sitting near two others that are each other's true match).
        double reciprocalBestDistSq = kMaxPairDistance * kMaxPairDistance;
        size_t reciprocalBestI = orange.size();
        for (size_t k = 0; k < orange.size(); ++k)
        {
            if (k == bestJ)
            {
                continue;
            }
            const double dx = orange[k].x - orange[bestJ].x;
            const double dy = orange[k].y - orange[bestJ].y;
            const double distSq = dx * dx + dy * dy;
            if (distSq < reciprocalBestDistSq)
            {
                reciprocalBestDistSq = distSq;
                reciprocalBestI = k;
            }
        }
        if (reciprocalBestI != i)
        {
            continue;
        }
        gateMidpoints.push_back(
            PathPoint{(a.x + orange[bestJ].x) / 2.0, (a.y + orange[bestJ].y) / 2.0});
    }
    return gateMidpoints;
}
}  // namespace
std::vector<PathPoint> TwoPointMidpointExtractor(const std::vector<WorldCone> &blue,
                                                  const std::vector<WorldCone> &yellow,
                                                  const std::vector<WorldCone> &orange,
                                                  const Pose2D &vehiclePose)
{
    // REWRITTEN (2026-09-05, user report: "the planning pipeline doesn't
    // properly handle this scattered arrangement of cones" at the hairpin
    // -- confirmed live, not assumed: captured the exact windowed blue/
    // yellow landmarks at the failure point and replayed this function's
    // OLD mutual-nearest-neighbor logic against them by hand. Of 7 blue
    // cones, 2 were dropped outright: each one's own nearest yellow cone
    // was ALSO some OTHER (closer) blue cone's nearest yellow, and the
    // reciprocal check -- correctly, by its own rule -- rejected the
    // farther blue's claim as a cross-pair rather than force a bad pairing.
    // But dropping it entirely, with no fallback, left a hole: the
    // resulting midpoints jumped from (-23.71,-10.84) straight to
    // (-25.29,-18.98), an 8.29m gap -- just OVER kMaxPairDistance (8.0m),
    // the hop cap OrderWaypointsByTraversal's own greedy walk uses. The
    // walk correctly refused to bridge a gap that large, stopped there,
    // and dropped every real midpoint beyond it -- the exact truncation
    // this whole investigation has been chasing. This was never a
    // perception problem (both dropped cones were legitimately detected
    // and localized, sitting right there in the landmark map) and never a
    // "too aggressive" threshold to loosen -- it's a real structural gap
    // in a competitive, single-shot nearest-neighbor rule that has no
    // notion of "already claimed, try the next-best option instead of
    // giving up entirely" once local cone spacing along the two chains
    // stops lining up 1:1 (unremarkable at a hairpin, where the inside
    // boundary's tighter radius packs its own cones closer together than
    // the outside one).
    //
    // Replaced with the SAME per-color-CHAIN approach
    // ClosedLoopMidpointExtractor (below) already uses successfully: order
    // blue and yellow SEPARATELY into their own travel-ordered chains
    // first, then walk the (now-ordered) blue chain in sequence, pairing
    // each point with whichever yellow-chain point is nearest within a
    // cursor-anchored search window (escalating to a full-chain scan if
    // the windowed result looks suspiciously far -- same sanity-recheck
    // pattern already proven at this exact class of bug elsewhere in this
    // file and in corridor.cpp). Critically, this has NO reciprocal/mutual
    // requirement at all: walking one blue point at a time in chain order
    // means there's no "two blues competing for one yellow" scenario to
    // begin with -- each blue independently gets the genuinely nearest
    // yellow available, full stop, which is exactly what a real, single
    // continuous boundary should produce. The chains' own ordering step
    // (not the pairing step) is what still needs generous heading-reversal
    // tolerance for a real hairpin's own sharp, coarsely-cone-spaced turn
    // -- see the 150-degree override below, same reasoning this file's own
    // history already established for the old approach's final reorder,
    // just applied one stage earlier now.
    auto toPoints = [](const std::vector<WorldCone> &_cones)
    {
        std::vector<PathPoint> pts;
        pts.reserve(_cones.size());
        for (const auto &c : _cones)
        {
            pts.push_back(PathPoint{c.x, c.y});
        }
        return pts;
    };

    // CHAIN HOP CAP -- two failed single-threshold attempts before this one
    // (2026-09-08, same day, user principle: the published path should
    // never be cut short). Attempt 1 (kMaxPairDistance=8.0): correctly
    // reached the hairpin's inside boundary in 3 hops, then stopped because
    // the genuinely next cone sat 11.2m away (a temporary perception miss)
    // -- confirmed live as the direct cause of getting stuck/wedged there,
    // with the car unable to route around a nearby cone on a too-short
    // path. Attempt 2 (kClosedChainMaxHop=15.0, same 150-degree tolerance
    // as the short-hop case): reached far enough, but a live capture right
    // after the very next run showed the published midpoint sequence
    // jumping between DIFFERENT LEGS of the hairpin fold-back -- a rollover.
    // The real problem was never the DISTANCE cap alone -- it's that a
    // single heading tolerance can't safely license both a short, sharp
    // hairpin-apex turn (needs ~150 degrees) AND a long gap-bridge hop
    // (which should need much LESS heading change than that, since a
    // genuine same-leg continuation is still following the same gentle
    // curve, just with one cone missing -- it's the wrong-leg candidates
    // that typically demand the big swing). Fixed at the mechanism level
    // (see OrderWaypointsByTraversal's own header comment for the general
    // reasoning): _longHopDistance=kMaxPairDistance keeps the full 150-
    // degree tolerance for any hop inside the normal, validated range, but
    // demands _longHopMaxHeadingReversalDeg=70.0 for anything beyond it --
    // generous enough for a real curving continuation over a longer span,
    // tight enough to firmly reject a fold-back leg swap. Not yet re-
    // validated live against this exact hairpin capture (the wrong-leg
    // jump's own raw hop distance/angle wasn't preserved from that run) --
    // watch for a recurrence of either failure mode and retune from there,
    // not by reverting to a single blind threshold again.
    // CROSS-COLOR DIVERGENCE CHECK (2026-09-08, same day, user report:
    // "the planned path... too wide" right after the hairpin). Confirmed
    // via direct replay against a real capture: yellowChain's own hop 0
    // grabbed a cone still sitting in the window from the hairpin's own
    // just-exited inner/return leg (hop 0 is deliberately heading-
    // unchecked -- see OrderWaypointsByTraversal's own comment for why),
    // and every later hop walked deeper along that wrong leg while
    // blueChain correctly continued forward -- the corridor then measured
    // width against two chains tracing different, non-parallel parts of
    // the track. BuildValidatedChainPair (path_utils.hpp/.cpp) detects
    // exactly this (the two chains' own net directions ending up more than
    // 90 degrees apart, which a real track's parallel boundaries should
    // never do) and retries whichever chain disagrees more with the
    // vehicle's own heading, excluding its own bad first point.
    const PathPoint cursorStart{vehiclePose.x, vehiclePose.y};
    const double headingX = std::cos(vehiclePose.yaw);
    const double headingY = std::sin(vehiclePose.yaw);
    fsd::ChainPair validatedChains =
        BuildValidatedChainPair(toPoints(blue), toPoints(yellow), cursorStart, kClosedChainMaxHop,
                                 /*_haveInitialHeading=*/true, headingX, headingY,
                                 /*_maxHeadingReversalDeg=*/150.0, /*_longHopDistance=*/kMaxPairDistance,
                                 /*_longHopMaxHeadingReversalDeg=*/70.0);
    const std::vector<PathPoint> &blueChain = validatedChains.blue;
    const std::vector<PathPoint> &yellowChain = validatedChains.yellow;
    if (blueChain.empty() || yellowChain.empty())
    {
        return {};
    }

    // LONGITUDINAL-ALIGNMENT PAIRING (2026-09-08, user report: path
    // disappearing mid-corner at the hairpin on lap 1 -- root-caused via
    // static review, not yet a fresh live capture: this loop was still
    // using the OLD raw-nearest-Euclidean selection ClosedLoopMidpointExtractor
    // itself was fixed away from on 2026-09-06 (see that function's own
    // comment for the full derivation -- a real blue/yellow cone-spacing
    // mismatch makes "nearest by raw distance" flip-flop between the
    // yellow cone ahead and the one behind as the blue index advances,
    // producing a genuine zigzag). That fix was never ported back to this,
    // its twin function -- and a hairpin is exactly where the underlying
    // mismatch is worst: the inside boundary's tighter turn radius packs
    // its own cones closer together than the outside boundary's (see this
    // function's own REWRITTEN comment above), which is a BIGGER spacing
    // mismatch than the roughly-straight section the bug was originally
    // confirmed at. Same fix: prefer the yellow candidate with the
    // smallest longitudinal offset along the blue chain's own local
    // tangent (most directly across), not the smallest raw distance.
    std::vector<PathPoint> waypoints;
    waypoints.reserve(blueChain.size());
    const int yellowSize = static_cast<int>(yellowChain.size());
    const int blueSize = static_cast<int>(blueChain.size());
    size_t yellowCursor = 0;
    for (int bi = 0; bi < blueSize; ++bi)
    {
        const PathPoint &b = blueChain[static_cast<size_t>(bi)];
        const PathPoint &bPrev = blueChain[static_cast<size_t>(bi > 0 ? bi - 1 : bi)];
        const PathPoint &bNext = blueChain[static_cast<size_t>(bi + 1 < blueSize ? bi + 1 : bi)];
        double btx = bNext.x - bPrev.x;
        double bty = bNext.y - bPrev.y;
        const double btlen = std::sqrt(btx * btx + bty * bty);
        if (btlen > 1e-9)
        {
            btx /= btlen;
            bty /= btlen;
        }

        const int lo = std::max(0, static_cast<int>(yellowCursor) - kClosedChainSearchWindow);
        const int hi = std::min(yellowSize - 1, static_cast<int>(yellowCursor) + kClosedChainSearchWindow);
        int bestIdx = -1;
        double bestLongAbs = std::numeric_limits<double>::max();
        double bestDistSq = std::numeric_limits<double>::max();
        for (int i = lo; i <= hi; ++i)
        {
            const double dx = yellowChain[static_cast<size_t>(i)].x - b.x;
            const double dy = yellowChain[static_cast<size_t>(i)].y - b.y;
            const double longAbs = std::abs(dx * btx + dy * bty);
            if (longAbs < kPairLongitudinalWindow && longAbs < bestLongAbs)
            {
                bestLongAbs = longAbs;
                bestDistSq = dx * dx + dy * dy;
                bestIdx = i;
            }
        }
        if (bestIdx < 0)
        {
            // Nothing in the window is genuinely "beside" this blue point
            // -- fall back to raw nearest rather than reporting no pair
            // when a usable one exists just not well-aligned.
            for (int i = lo; i <= hi; ++i)
            {
                const double dx = yellowChain[static_cast<size_t>(i)].x - b.x;
                const double dy = yellowChain[static_cast<size_t>(i)].y - b.y;
                const double distSq = dx * dx + dy * dy;
                if (distSq < bestDistSq)
                {
                    bestDistSq = distSq;
                    bestIdx = i;
                }
            }
        }
        if (bestIdx < 0)
        {
            continue;  // yellowChain ran out on this side -- skip, don't force a bad pair
        }
        // Same sanity re-check as ClosedLoopMidpointExtractor's own
        // (identical structural risk: yellowCursor can fall behind the
        // blue chain's true nearest yellow, since local density along the
        // two chains isn't perfectly correlated index-for-index) -- see
        // that function's own comment for the full reasoning.
        //
        // BUG FIX (2026-09-08, live-confirmed via direct capture at the
        // hairpin, /planning/debug_midpoints and /planning/debug_spline):
        // this rescan used to select by longitudinal alignment with NO
        // effective upper bound (globalBestLongAbs seeded from bestLongAbs,
        // which stays at std::numeric_limits<double>::max() whenever the
        // primary windowed search found nothing plausible) -- so it would
        // accept literally any yellow point in the whole chain as "best",
        // however misaligned. Confirmed live: a captured midpoint at
        // (-23.55,-16.22) sat wildly out of sequence between neighbors at
        // y=-9.66 and y=-13.86, and FitAndSampleSpline's own curve through
        // it swung within 0.156m of an unrelated real cone as a direct
        // consequence.
        // A first fix (bounding the same longitudinal criterion at
        // kPairLongitudinalWindow) was itself confirmed live to be the
        // WRONG criterion for this tier specifically: bPrev/bNext's local
        // tangent estimate is exactly what's unreliable right at a sharp
        // apex (large direction change over very few chain points), so
        // even GENUINELY correct pairs there can read as implausibly
        // misaligned by that measure -- confirmed directly by ground-truth
        // replay: every real blue/yellow cross-pair at this exact hairpin
        // (six of them, checked against trackdrive.sdf) measures EXACTLY
        // 3.000m apart by raw Euclidean distance, a track-wide constant
        // that doesn't depend on any tangent estimate at all. Bounding by
        // kPairLongitudinalWindow rejected legitimate pairs here too,
        // starving the extractor down to too few midpoints and stalling
        // the car in the creep/sweep search instead of crashing -- safer
        // than the original bug, but still not making it through.
        // FIXED: this tier now mirrors the ALREADY-SAFE raw-nearest
        // fallback one tier up (used when nothing in the WINDOW passes the
        // longitudinal filter) -- extended to the whole chain instead of
        // just the window, and still bounded by kMaxPairDistance (the same
        // "plausibly a real pair" bound already used everywhere else in
        // this file, six times the real 3.0m spacing here, so it can't
        // accidentally reach a wrong-leg cone at this hairpin's own real
        // leg separation). No longitudinal/tangent dependence at all, so
        // it can't be defeated by a bad tangent estimate the way the
        // alignment-based criterion was.
        if (bestDistSq > kMaxPairDistance * kMaxPairDistance)
        {
            int globalBestIdx = -1;
            double globalBestDistSq = kMaxPairDistance * kMaxPairDistance;
            for (int i = 0; i < yellowSize; ++i)
            {
                const double dx = yellowChain[static_cast<size_t>(i)].x - b.x;
                const double dy = yellowChain[static_cast<size_t>(i)].y - b.y;
                const double distSq = dx * dx + dy * dy;
                if (distSq < globalBestDistSq)
                {
                    globalBestDistSq = distSq;
                    globalBestIdx = i;
                }
            }
            if (globalBestIdx < 0)
            {
                continue;  // nothing within kMaxPairDistance anywhere on this side -- skip, don't force a bad pair
            }
            bestIdx = globalBestIdx;
            bestDistSq = globalBestDistSq;
        }
        yellowCursor = static_cast<size_t>(bestIdx);
        const PathPoint &y = yellowChain[static_cast<size_t>(bestIdx)];
        waypoints.push_back(PathPoint{(b.x + y.x) / 2.0, (b.y + y.y) / 2.0});
    }

    // Gate anchor(s) -- see OrangeGateMidpoints's own comment. Appended
    // before the final ordering pass so it places them correctly in the
    // travel sequence rather than needing special-casing.
    const std::vector<PathPoint> gateMidpoints = OrangeGateMidpoints(orange);
    waypoints.insert(waypoints.end(), gateMidpoints.begin(), gateMidpoints.end());

    // Final pass: waypoints are ALREADY in travel order by construction
    // (one per blueChain entry, in blueChain's own already-correct order)
    // except for the just-appended orange anchors, which this call folds
    // into their correct position -- a safety net matching the old
    // implementation's own final step, not the primary ordering mechanism
    // anymore. Same 150-degree tolerance as the chain-ordering calls above
    // for consistency, though this pass should rarely need to reject
    // anything now that the pairing itself can no longer create the
    // multi-meter gaps that used to force it to.
    return OrderWaypointsByTraversal(std::move(waypoints), PathPoint{vehiclePose.x, vehiclePose.y},
                                      kMaxPairDistance, /*_haveInitialHeading=*/true,
                                      std::cos(vehiclePose.yaw), std::sin(vehiclePose.yaw),
                                      /*_maxHeadingReversalDeg=*/150.0);
}

std::vector<PathPoint> ClosedLoopMidpointExtractor(const std::vector<WorldCone> &blue,
                                                    const std::vector<WorldCone> &yellow,
                                                    const std::vector<WorldCone> &orange,
                                                    const Pose2D &vehiclePose)
{
    auto toPoints = [](const std::vector<WorldCone> &_cones)
    {
        std::vector<PathPoint> pts;
        pts.reserve(_cones.size());
        for (const auto &c : _cones)
        {
            pts.push_back(PathPoint{c.x, c.y});
        }
        return pts;
    };

    // "why is the midpoint calc cooked" (2026-09-05, user report -- looking
    // at live /planning/debug_midpoints near the start/finish straight).
    // Confirmed directly: blue_001/yellow_001 and blue_002/yellow_002 (both
    // present, well-observed, normally ~2m-spaced landmarks -- verified
    // live against /estimated_landmarks, nothing missing from the map) were
    // COMPLETELY ABSENT from the published midpoint array, leaving an
    // 8.34m gap right through the gate -- the exact spot both this
    // session's closed-loop stalls have now happened at, independent of
    // speed (15.0 and 5.0 both hit it). Root cause: these two
    // OrderWaypointsByTraversal calls build the blue/yellow chains from
    // RAW, cone-spaced landmarks (several meters apart), the same category
    // TwoPointMidpointExtractor's own midpoints are (see its own
    // _maxHeadingReversalDeg=150.0 override and comment) -- but were never
    // given the same override, so they still use the tighter 90-degree
    // default validated only for DENSELY spline-resampled points. Seeded
    // heading (vehiclePose.yaw, from the exact instant this lap's recompute
    // triggered) stays the reference for hop 1 specifically (hop 0 is
    // exempt, see OrderWaypointsByTraversal's own comment) -- if the
    // vehicle's actual heading at that instant doesn't align well with
    // this straight's true direction (plausible: lap detection can trigger
    // a beat before/after the car is perfectly square with the track), hop
    // 1 gets rejected under the 90-degree default, and the walk permanently
    // skips that cone rather than ever revisiting it. Same fix as
    // TwoPointMidpointExtractor's own: widen to 150.0.
    // CROSS-COLOR DIVERGENCE CHECK (2026-09-08) -- same mechanism and same
    // fix as TwoPointMidpointExtractor's own call (see its comment for the
    // full confirmed-live reasoning): this chain build has the identical
    // heading-unchecked-hop-0 structure and is therefore just as
    // vulnerable to grabbing a wrong-leg cone right after a fold-back.
    const PathPoint cursorStart{vehiclePose.x, vehiclePose.y};
    const double headingX = std::cos(vehiclePose.yaw);
    const double headingY = std::sin(vehiclePose.yaw);
    fsd::ChainPair validatedChains =
        BuildValidatedChainPair(toPoints(blue), toPoints(yellow), cursorStart, kClosedChainMaxHop,
                                 /*_haveInitialHeading=*/true, headingX, headingY,
                                 /*_maxHeadingReversalDeg=*/150.0);
    const std::vector<PathPoint> &blueChain = validatedChains.blue;
    const std::vector<PathPoint> &yellowChain = validatedChains.yellow;
    if (blueChain.empty() || yellowChain.empty())
    {
        return {};
    }

    // LONGITUDINAL-ALIGNMENT PAIRING (2026-09-06, user report: "path plan
    // went too wide" -- confirmed live right at the start/finish straight,
    // /planning/debug_corridor_left and _right symmetrically ballooning
    // from ~0.9m to ~1.95m combined width over about 8m of travel, with
    // BOTH sides growing in lockstep. Traced to the SPLINE's own local
    // tangent swinging +-25 degrees off the straight's true heading right
    // there (confirmed via /planning/debug_spline) -- and that swing traces
    // further back to the RAW MIDPOINT sequence itself already zigzagging
    // (80 -> 102 -> 78 -> 58 degrees heading, over cones this track's own
    // ground truth confirms sit on a near-perfectly straight line). Root
    // cause: this track's blue cones here are spaced ~1.85m apart while the
    // paired yellow cones are spaced ~2.15m apart (confirmed directly
    // against /estimated_landmarks) -- a real, physical mismatch, not
    // sensor noise. The OLD selection below picked whichever yellowChain
    // point was RAW-EUCLIDEAN nearest to each blue point -- fine when the
    // two chains' spacing lines up, but with a genuine spacing mismatch,
    // "nearest by raw distance" flip-flops between the yellow cone ahead
    // and the one behind as the blue index advances, since a blue point
    // can sit closer to a DIAGONALLY-nearby yellow cone than to the one
    // genuinely "across" from it. That flip-flop is exactly what produces
    // the zigzag. Fixed the same way corridor.cpp's own
    // NearestChainLateralDistance already selects a "beside" point: prefer
    // the candidate with the SMALLEST longitudinal offset along the blue
    // chain's own LOCAL tangent (i.e. most directly across), not the
    // smallest raw distance -- this is invariant to a spacing mismatch
    // between the two chains, since it asks "which yellow point is most
    // nearly beside this blue point" instead of "which is closest overall".
    // (kPairLongitudinalWindow now lives at file scope -- see its own
    // comment there -- shared with TwoPointMidpointExtractor's identical fix.)
    std::vector<PathPoint> midpoints;
    midpoints.reserve(blueChain.size());
    const int yellowSize = static_cast<int>(yellowChain.size());
    const int blueSize = static_cast<int>(blueChain.size());
    size_t yellowCursor = 0;
    for (int bi = 0; bi < blueSize; ++bi)
    {
        const PathPoint &b = blueChain[static_cast<size_t>(bi)];
        const PathPoint &bPrev = blueChain[static_cast<size_t>(bi > 0 ? bi - 1 : bi)];
        const PathPoint &bNext = blueChain[static_cast<size_t>(bi + 1 < blueSize ? bi + 1 : bi)];
        double btx = bNext.x - bPrev.x;
        double bty = bNext.y - bPrev.y;
        const double btlen = std::sqrt(btx * btx + bty * bty);
        if (btlen > 1e-9)
        {
            btx /= btlen;
            bty /= btlen;
        }

        const int lo = std::max(0, static_cast<int>(yellowCursor) - kClosedChainSearchWindow);
        const int hi = std::min(yellowSize - 1, static_cast<int>(yellowCursor) + kClosedChainSearchWindow);
        int bestIdx = -1;
        double bestLongAbs = std::numeric_limits<double>::max();
        double bestDistSq = std::numeric_limits<double>::max();
        for (int i = lo; i <= hi; ++i)
        {
            const double dx = yellowChain[static_cast<size_t>(i)].x - b.x;
            const double dy = yellowChain[static_cast<size_t>(i)].y - b.y;
            const double longAbs = std::abs(dx * btx + dy * bty);
            if (longAbs < kPairLongitudinalWindow && longAbs < bestLongAbs)
            {
                bestLongAbs = longAbs;
                bestDistSq = dx * dx + dy * dy;
                bestIdx = i;
            }
        }
        if (bestIdx < 0)
        {
            // Nothing in the window is genuinely "beside" this blue point
            // (sparse yellow chain here) -- fall back to raw nearest,
            // same as the old behavior, rather than reporting no pair when
            // a usable one exists just not well-aligned.
            for (int i = lo; i <= hi; ++i)
            {
                const double dx = yellowChain[static_cast<size_t>(i)].x - b.x;
                const double dy = yellowChain[static_cast<size_t>(i)].y - b.y;
                const double distSq = dx * dx + dy * dy;
                if (distSq < bestDistSq)
                {
                    bestDistSq = distSq;
                    bestIdx = i;
                }
            }
        }
        if (bestIdx < 0)
        {
            continue;  // yellowChain ran out on this side -- skip, don't force a bad pair
        }

        // SANITY RE-CHECK (2026-09-04, user report: "the midpoints veer off
        // the track" -- confirmed live via a captured snapshot: one
        // midpoint sat 10-13m from BOTH its array-adjacent neighbors, with
        // no legitimate (<8m) blue-yellow pair anywhere close to it,
        // producing a genuine 179-degree heading reversal in the
        // downstream spline). Root cause: this is the exact same
        // structural gap as corridor.cpp's NearestChainLateralDistance,
        // fixed there earlier the same day but never applied here --
        // yellowCursor can fall behind the blue chain's own true nearest
        // yellow (blue/yellow local density along the two chains isn't
        // perfectly correlated index-for-index), and once it has, this
        // windowed search keeps "successfully" finding SOME candidate
        // within its own +-8 window without that candidate being the
        // genuinely closest yellow anywhere on the chain. Only pay the
        // extra full-chain scan when the windowed result already looks
        // suspicious (farther than kMaxPairDistance, the same "plausibly a
        // real pair" bound every other pairing in this file already uses)
        // -- the common, already-correct case pays nothing extra.
        //
        // BUG FIX (2026-09-08, same defect confirmed live in this
        // function's twin, TwoPointMidpointExtractor -- see its own
        // matching comment for the full derivation, live evidence, and the
        // reasoning behind this specific fix): this rescan used to select
        // by longitudinal alignment with no effective upper bound
        // (globalBestLongAbs seeded from bestLongAbs, which stays at
        // std::numeric_limits<double>::max() whenever the primary windowed
        // search found nothing plausible), so it would accept literally any
        // yellow point in the whole chain as "best", however misaligned. A
        // first fix (bounding that same criterion at kPairLongitudinalWindow)
        // was itself confirmed live to be the wrong criterion for this tier:
        // bPrev/bNext's local tangent is exactly what's unreliable right at
        // a sharp apex, so even genuinely correct pairs there can read as
        // implausibly misaligned by that measure -- confirmed by ground-
        // truth replay showing every real cross-pair at this track's own
        // hairpin measures a constant 3.000m by raw Euclidean distance,
        // independent of any tangent estimate. This tier now mirrors the
        // ALREADY-SAFE raw-nearest fallback one tier up (used when nothing
        // in the WINDOW passes the longitudinal filter), extended to the
        // whole chain instead of just the window, and still bounded by
        // kMaxPairDistance so it can't reach a wrong-leg cone at this
        // hairpin's own real leg separation.
        if (bestDistSq > kMaxPairDistance * kMaxPairDistance)
        {
            int globalBestIdx = -1;
            double globalBestDistSq = kMaxPairDistance * kMaxPairDistance;
            for (int i = 0; i < yellowSize; ++i)
            {
                const double dx = yellowChain[static_cast<size_t>(i)].x - b.x;
                const double dy = yellowChain[static_cast<size_t>(i)].y - b.y;
                const double distSq = dx * dx + dy * dy;
                if (distSq < globalBestDistSq)
                {
                    globalBestDistSq = distSq;
                    globalBestIdx = i;
                }
            }
            if (globalBestIdx < 0)
            {
                continue;  // nothing within kMaxPairDistance anywhere on this side -- skip, don't force a bad pair
            }
            bestIdx = globalBestIdx;
            bestDistSq = globalBestDistSq;
        }

        yellowCursor = static_cast<size_t>(bestIdx);
        const PathPoint &y = yellowChain[static_cast<size_t>(bestIdx)];
        midpoints.push_back(PathPoint{(b.x + y.x) / 2.0, (b.y + y.y) / 2.0});
    }

    // OrderWaypointsByTraversal has no concept of vehicle heading -- it
    // just walks toward whichever remaining point is nearest, so the very
    // FIRST hop from the vehicle's own position can commit to either
    // rotational direction around the loop with equal ease (unlike the
    // windowed/reactive pipeline, which never faces this because
    // LandmarkMap::QueryWindow already excludes everything behind the car
    // before ordering even starts -- QueryAll() has no such filter, since
    // a closed loop genuinely needs every cone, front and back, to close).
    // Confirmed live (2026-08-31) as a real, not theoretical, failure:
    // this walk committed backward, so EVERY point in the published
    // bounded slice had negative body-frame x, and RemoveBehindCarPoints
    // stripped all of them -- an empty /planned_path, car stuck, with no
    // upstream stage reporting anything wrong (midpoints/spline/corridor/
    // optimizer all produced perfectly valid data, just walking the wrong
    // way). Checked once here, not per-hop: once the greedy walk commits
    // to a rotational direction at its first hop, every later hop
    // continues the same way (each visited point is removed from
    // consideration, so "nearest remaining" naturally keeps progressing
    // around the loop, not oscillating) -- confirmed against the same
    // live capture that exposed this bug. If the chain's own start heads
    // away from the vehicle's forward heading, reverse the whole thing.
    if (midpoints.size() >= 2)
    {
        const double cosYaw = std::cos(vehiclePose.yaw);
        const double sinYaw = std::sin(vehiclePose.yaw);
        const double forward =
            (midpoints.front().x - vehiclePose.x) * cosYaw + (midpoints.front().y - vehiclePose.y) * sinYaw;
        if (forward < 0.0)
        {
            // Reverse the WALK direction without relocating index 0 away
            // from the vehicle: a plain std::reverse would move the
            // vehicle-nearest point (currently at the front, by
            // construction of OrderWaypointsByTraversal's own cursor)
            // to the BACK instead, which would break the bounded-slice
            // truncation below (it assumes index 0 is near the vehicle).
            // Reverse, then rotate that same point back to the front --
            // on a closed loop this is exactly "start at the same point,
            // walk the ring the other way", not a different set of
            // points or a different starting position.
            std::reverse(midpoints.begin(), midpoints.end());
            std::rotate(midpoints.begin(), midpoints.end() - 1, midpoints.end());
        }
    }

    // Clean up a local backtrack/zigzag right at the start -- confirmed
    // live (2026-09-01) as a real defect, structurally confined to the
    // first few hops: index 0 is anchored to the nearest cone to the
    // VEHICLE's raw world position (not a real track point), so it isn't
    // guaranteed to be a point that leads smoothly onward -- unlike every
    // LATER hop, which is anchored to a real, already-visited cone (a
    // trustworthy reference). Confirmed directly via a live capture: index
    // 1 stepped away from index 0, then index 2 reversed hard back PAST
    // index 0's own position (a ~1.8x detour ratio -- going through index
    // 1 was ~1.8x longer than going directly from 0 to 2), producing a
    // visible triangular spike in the downstream spline/corridor (this is
    // where debug_corridor_left/right's own local turn angles spiked to
    // 90-145 degrees). A point p is a genuine local detour if the path
    // through it is meaningfully longer than skipping it entirely -- the
    // classic triangle-inequality signature of a point that doesn't
    // belong in the chain's own natural order.
    //
    // Deliberately scoped to a SMALL window right after index 0, not
    // applied to the whole array: elsewhere in the loop, every hop is
    // anchored to a real cone and a genuinely sharp turn (e.g. an actual
    // hairpin apex) can legitimately show a similar detour ratio without
    // being a defect -- this codebase's own racing-line optimizer already
    // widens real hairpins, so removing a real apex point here would be
    // actively harmful. Restricting the check to where the defect is
    // actually possible avoids that risk entirely.
    constexpr size_t kSeamCleanupWindow = 5;
    constexpr double kSeamDetourRatio = 1.5;
    for (size_t pass = 0; pass < 2 && midpoints.size() > kSeamCleanupWindow + 2; ++pass)
    {
        bool removed = false;
        for (size_t i = 1; i < std::min(kSeamCleanupWindow, midpoints.size() - 1); ++i)
        {
            const PathPoint &prev = midpoints[i - 1];
            const PathPoint &cur = midpoints[i];
            const PathPoint &next = midpoints[i + 1];
            const double viaDist = std::hypot(cur.x - prev.x, cur.y - prev.y) +
                                    std::hypot(next.x - cur.x, next.y - cur.y);
            const double directDist = std::hypot(next.x - prev.x, next.y - prev.y);
            if (directDist > 1e-6 && viaDist / directDist > kSeamDetourRatio)
            {
                midpoints.erase(midpoints.begin() + static_cast<std::ptrdiff_t>(i));
                removed = true;
                break;  // indices shifted -- restart this pass's scan
            }
        }
        if (!removed)
        {
            break;
        }
    }

    // GLOBAL GROSS-OUTLIER REMOVAL (2026-09-08, user report: "you can see
    // the problem with the debug midpoints right now" -- confirmed live,
    // an undeniable one: a live capture showed isolated points teleporting
    // tens of meters away and back within 1-2 array steps, e.g.
    // (-22.84,-12.65) -> (9.50,-19.66) -> (-40.30,-7.13), a detour ratio of
    // ~4.6x. This is a DIFFERENT failure from the chain-divergence bug
    // fixed the same day (BuildValidatedChainPair, path_utils.hpp): that
    // one is a whole CHAIN heading the wrong way from a bad hop 0; this is
    // an isolated bad PAIRING somewhere in the middle of an otherwise-
    // correct chain (most plausibly the yellowCursor losing sync mid-chain
    // and the existing per-pair sanity re-check -- see this function's own
    // pairing loop above -- not catching it because the wrong candidate
    // still happened to be within kMaxPairDistance, just not the genuinely
    // closest one). The seam-cleanup pass just above catches this exact
    // triangle-inequality signature already, but is deliberately restricted
    // to the first few points near index 0 (see its own comment for why:
    // a real hairpin apex elsewhere can legitimately show a MILD detour
    // ratio without being a defect, and this codebase's own racing-line
    // optimizer needs those real apex points, not a cleaned-up centerline).
    // This pass runs the SAME check across the WHOLE array (closed loop,
    // so wrapping at both ends).
    //
    // RATIO ALONE WASN'T ENOUGH (2026-09-08, same day, user report: "the
    // planned path is still broken" -- confirmed live, right after this
    // pass first shipped: the 30+ meter teleports were gone, but a real
    // residual zigzag remained in the same hairpin-adjacent stretch, e.g.
    // (-30.64,-8.89) -> (-28.54,-18.10) -> (-24.60,-10.06), a 9.45m single
    // hop with a detour ratio of 2.99 -- just under the original 3.0
    // threshold, and directly upstream of the next stuck event at
    // (-25.27,-12.72), inside that same span). A pure ratio threshold
    // can't safely go much below 3.0 globally: a genuinely sharp real turn
    // over a SHORT absolute distance can have just as high a ratio as a
    // buggy long-distance zigzag, so tightening the ratio bar alone risks
    // cutting real apex points elsewhere on the track. Fixed by adding a
    // second, absolute-distance condition instead: a hop this large in
    // real terms (kGrossDetourAbsoluteDist, well beyond this track's own
    // measured normal cone-derived midpoint spacing of a few meters) is
    // never a legitimate single hop regardless of ratio, so the two
    // conditions together only fire on hops that are BOTH suspiciously
    // long AND suspiciously indirect -- a real short, sharp apex turn
    // (small absolute distance, ratio irrelevant) stays untouched, while a
    // genuinely bad long-distance pairing (this bug's own actual
    // signature) gets caught even at a ratio as mild as 2.0.
    constexpr double kGrossDetourRatio = 2.0;
    constexpr double kGrossDetourAbsoluteDist = 5.0;  // meters
    const size_t maxGrossPasses = std::min(midpoints.size(), static_cast<size_t>(20));
    for (size_t pass = 0; pass < maxGrossPasses && midpoints.size() > 4; ++pass)
    {
        bool removed = false;
        const size_t n = midpoints.size();
        for (size_t i = 0; i < n; ++i)
        {
            const PathPoint &prev = midpoints[(i + n - 1) % n];
            const PathPoint &cur = midpoints[i];
            const PathPoint &next = midpoints[(i + 1) % n];
            const double distPrev = std::hypot(cur.x - prev.x, cur.y - prev.y);
            const double distNext = std::hypot(next.x - cur.x, next.y - cur.y);
            const double viaDist = distPrev + distNext;
            const double directDist = std::hypot(next.x - prev.x, next.y - prev.y);
            const bool suspiciouslyLong = std::max(distPrev, distNext) > kGrossDetourAbsoluteDist;
            if (directDist > 1e-6 && suspiciouslyLong && viaDist / directDist > kGrossDetourRatio)
            {
                midpoints.erase(midpoints.begin() + static_cast<std::ptrdiff_t>(i));
                removed = true;
                break;  // indices shifted -- restart this pass's scan
            }
        }
        if (!removed)
        {
            break;
        }
    }

    // Gate anchor(s) -- see OrangeGateMidpoints's own comment. Inserted
    // LAST, after every index-0-relative pass above (the direction-check
    // and seam-cleanup logic both assume index 0 is the vehicle-nearest
    // point; inserting earlier could shift that or get cleaned up as a
    // spurious "detour" itself). Unlike the open pipeline's own version
    // (which just appends before a fresh OrderWaypointsByTraversal call),
    // this array is ALREADY in final travel order with no re-sort left to
    // run -- each gate point is inserted via "cheapest insertion" (find
    // the adjacent pair {midpoints[i], midpoints[i+1]} -- wrapping at the
    // end, since this is a closed loop -- whose replacement by
    // {midpoints[i], gate, midpoints[i+1]} adds the least extra distance),
    // the standard way to place a new point into an already-ordered chain
    // without knowing its "true" index ahead of time.
    const std::vector<PathPoint> gateMidpoints = OrangeGateMidpoints(orange);
    for (const auto &gate : gateMidpoints)
    {
        if (midpoints.size() < 2)
        {
            midpoints.push_back(gate);
            continue;
        }
        size_t bestI = 0;
        double bestExtra = std::numeric_limits<double>::max();
        for (size_t i = 0; i < midpoints.size(); ++i)
        {
            const PathPoint &p = midpoints[i];
            const PathPoint &q = midpoints[(i + 1) % midpoints.size()];
            const double viaDist =
                std::hypot(gate.x - p.x, gate.y - p.y) + std::hypot(q.x - gate.x, q.y - gate.y);
            const double directDist = std::hypot(q.x - p.x, q.y - p.y);
            const double extra = viaDist - directDist;
            if (extra < bestExtra)
            {
                bestExtra = extra;
                bestI = i;
            }
        }
        midpoints.insert(midpoints.begin() + static_cast<std::ptrdiff_t>(bestI) + 1, gate);
    }

    return midpoints;
}
}  // namespace fsd
