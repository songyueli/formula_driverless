#include "corridor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "path_utils.hpp"

namespace fsd
{
namespace
{
// How far along the tangent a chain candidate can sit and still count as
// "beside" this sample, not "ahead of/behind it" -- roughly a couple of
// this track's own measured cone spacings (~2m, see kDuplicatePruneRadius's
// comment in ekf.cpp), generous enough to always find a same-side
// candidate at normal sample spacing without pulling in one so far along
// the chain it isn't really telling you the width HERE.
constexpr double kLongitudinalWindow = 3.0;  // meters
// Real cone base radius (2026-09-05) -- same figure used elsewhere in this
// codebase (control/src/pure_pursuit_controller.cpp's kMinCarClearance
// derivation, localization/src/localization.cpp's LandmarkStddev) for "the
// largest real cone's own physical footprint". Used by the safety-floor
// fix below as a minimal physical buffer, so a capped bound can never land
// exactly at (zero clearance from) a real cone's own center point.
constexpr double kConeRadius = 0.1425;  // meters
// Chain-index window searched around the running cursor, not a global
// search -- see ComputeCorridor's own comment for why.
constexpr int kChainSearchWindow = 8;

// Known, ground-truth-confirmed total track width (2026-09-05) -- same
// figure as path_generator.cpp's own kAssumedHalfTrackWidth (1.5m each
// side), duplicated here rather than shared, same pattern as this file's
// other genuinely-small cross-file constants (kClosedChainMaxHop). Used by
// the width cross-check below as a correction anchor, not a hard bound
// like the corridor min/max half-widths.
constexpr double kAssumedTrackWidth = 3.0;  // meters
// How far blueDist+yellowDist may deviate from kAssumedTrackWidth before
// being treated as suspect rather than ordinary per-sample noise (a
// spline sample is rarely perfectly centered on the true cross-section,
// especially through a curve -- confirmed live at a genuine, trustworthy
// sample as a 1.47/1.53 split, not a perfect 1.50/1.50 one).
constexpr double kWidthSanityTolerance = 0.4;  // meters

// Hop-distance cap for BuildChain's own per-color chain ordering when
// building the CLOSED-LOOP corridor (ComputeCorridor's closed=true path,
// which orders chains over the ENTIRE discovered map, not a small nearby
// window) -- same value and same reasoning as centerline_extractor.cpp's
// own kClosedChainMaxHop (kept duplicated here rather than shared, same
// pattern as this project's other genuinely-small cross-file constants):
// this track's own blue/yellow boundaries have real gaps just over 8.0m
// somewhere (a missed detection, not a genuinely sparse track), which the
// default kMaxPairDistance=8.0 hop cap would truncate a full-track chain
// at. BUG FIX (2026-09-01): BuildChain was passing kMaxPairDistance
// unconditionally, even for closed=true -- confirmed live as a real,
// recurring failure: a chain broken here leaves NearestChainLateralDistance
// with no boundary data for every corridor sample past the break, which
// ComputeCorridor's own "no data" branch floors to minHalfWidth -- found
// live as a sustained 1.00m corridor (exactly 2*0.5m, the floor) across 13
// consecutive samples on a stretch this track's own cone layout measures
// at ~2.8m real width, directly upstream of a stuck event.
constexpr double kClosedChainMaxHop = 15.0;  // meters
// Moving-average half-width (in samples) for smoothing the raw per-sample
// half-width -- see ComputeCorridor's own comment for why it's piecewise
// and needs this.
constexpr int kSmoothingWindow = 3;

// Lateral (perpendicular, left-positive) distance from _sample to the
// nearest chain point "beside" it (within kLongitudinalWindow along
// _tangent), searching only a local index window around *_cursor and
// advancing *_cursor to whatever index was actually used -- this is what
// keeps consecutive samples from jumping across a hairpin fold-back to a
// same-color landmark on the chain's OTHER, not-yet-reached leg: the
// window can only ever advance a few indices per sample, the same
// discipline OrderWaypointsByTraversal's own hop cap enforces for the
// midpoint chain. Returns -1.0 (sentinel) if the chain has no candidate at
// all this cycle (empty chain, or window ran off either end) -- callers
// treat that as "no boundary data", not zero width.
double NearestChainLateralDistance(const std::vector<PathPoint> &_chain, size_t &_cursor,
                                    const PathPoint &_sample, double _tangentX, double _tangentY,
                                    bool _closed)
{
    if (_chain.empty())
    {
        return -1.0;
    }
    const int chainSize = static_cast<int>(_chain.size());
    // Closed: wrap the search window modulo chain size, so the index right
    // after the last one is treated as beside the one right before the
    // first -- see ComputeCorridor's own _closed comment. Open (default):
    // unchanged, clamp at both ends exactly as before.
    std::vector<int> indices;
    indices.reserve(static_cast<size_t>(2 * kChainSearchWindow + 1));
    if (_closed)
    {
        for (int off = -kChainSearchWindow; off <= kChainSearchWindow; ++off)
        {
            indices.push_back(((static_cast<int>(_cursor) + off) % chainSize + chainSize) % chainSize);
        }
    }
    else
    {
        const int lo = std::max(0, static_cast<int>(_cursor) - kChainSearchWindow);
        const int hi = std::min(chainSize - 1, static_cast<int>(_cursor) + kChainSearchWindow);
        for (int i = lo; i <= hi; ++i)
        {
            indices.push_back(i);
        }
    }

    // Prefer the closest-along-tangent candidate within the longitudinal
    // window -- "directly beside" the sample, not diagonally ahead of it.
    double bestLateral = -1.0;
    double bestLongAbs = std::numeric_limits<double>::max();
    int bestIdx = -1;
    for (int i : indices)
    {
        const double dx = _chain[static_cast<size_t>(i)].x - _sample.x;
        const double dy = _chain[static_cast<size_t>(i)].y - _sample.y;
        const double longitudinal = dx * _tangentX + dy * _tangentY;
        if (std::abs(longitudinal) < kLongitudinalWindow && std::abs(longitudinal) < bestLongAbs)
        {
            bestLongAbs = std::abs(longitudinal);
            bestLateral = std::abs(-dx * _tangentY + dy * _tangentX);
            bestIdx = i;
        }
    }
    if (bestIdx < 0)
    {
        // Nothing in the window passed the longitudinal filter this cycle
        // (sparse chain) -- fall back to the single closest-by-full-
        // distance candidate rather than reporting "no data" when there IS
        // a chain nearby, just none of it well-aligned.
        double bestDistSq = std::numeric_limits<double>::max();
        for (int i : indices)
        {
            const double dx = _chain[static_cast<size_t>(i)].x - _sample.x;
            const double dy = _chain[static_cast<size_t>(i)].y - _sample.y;
            const double distSq = dx * dx + dy * dy;
            if (distSq < bestDistSq)
            {
                bestDistSq = distSq;
                bestLateral = std::abs(-dx * _tangentY + dy * _tangentX);
                bestIdx = i;
            }
        }
    }
    // SANITY RE-CHECK (2026-09-04, user report: "the debug left corridor
    // and right corridor... near the hairpin?? it like goes wide" --
    // confirmed live: leftBound pinned at exactly kCorridorMaxHalfWidth
    // (2.0m, the ceiling clamp) for 10 consecutive samples through the
    // hairpin, while the REAL nearest blue cone at that exact spot
    // (cross-checked against ground truth) was only 1.60m away -- the
    // windowed search above was missing it and finding some farther chain
    // point instead). Same structural gap as the closedLoopCursor sanity
    // check added earlier the same day in planning.cpp: *_cursor can fall
    // behind the sample's true nearest chain point (a hairpin's own cone
    // spacing-vs-chain-index relationship doesn't track 1:1 with the
    // spline's own index advance), and once it has, the WINDOW (anchored
    // to the stale cursor) can keep "successfully" finding SOME candidate
    // that passes the longitudinal filter without that candidate being the
    // genuinely closest one -- silently returning a too-large lateral
    // distance instead of ever re-syncing. Only pay the extra full-chain
    // scan when the windowed result already looks suspicious (larger than
    // this file's own corridor ceiling could ever legitimately need,
    // maxHalfWidth is corridor.hpp's own constant, not available here --
    // reuse kCorridorMaxHalfWidth's caller-side value isn't threaded
    // through, so a fixed, generously-large sanity bound is used instead)
    // -- the common, already-correct case pays nothing extra.
    constexpr double kChainDistanceSanityBound = 3.0;  // meters
    // LOWER sanity bound added (2026-09-05, user report: debug_spline's own
    // left/right corridor width visibly collapsing to the kCorridorMinHalfWidth
    // floor on BOTH sides for a long, sustained stretch right through a
    // hairpin, with the car then wedging on a blue cone after the racing
    // line was forced needlessly narrow there). Confirmed via ground truth:
    // the REAL cone-to-cone width at that exact stretch measures ~3.00m,
    // identical to everywhere else on the track (blue_073/yellow_073,
    // blue_074/yellow_074, blue_075/yellow_075 all ~3.00m apart) -- this
    // was never a genuinely narrow section, the corridor's own DISTANCE
    // MEASUREMENT was wrong. Root cause: the sanity re-check above only
    // ever guarded against a windowed result that's too LARGE (missing the
    // true nearest candidate entirely). It had no defense against the
    // opposite failure, which is exactly what a hairpin's own fold-back
    // produces: the two legs of the same-color chain pass physically close
    // to each other there, so either the windowed search or its own
    // leg-blind "closest by full distance" fallback (used whenever nothing
    // in the window passes the longitudinal-alignment filter -- see above)
    // can latch onto a point on the WRONG leg that happens to be
    // Euclidean-close, producing a lateral distance far SMALLER than the
    // real boundary's own true distance, rather than far larger. A width
    // that small (kMinPlausibleLateralDistance below) is implausible for a
    // real track section -- this codebase's own narrowest already-confirmed
    // sections stay well above it -- so it's a reliable signal to re-verify.
    // Needs a DIFFERENT full-chain strategy from the too-large case just
    // below, not the same one reused: "find something smaller than the
    // current lateral" (the too-large branch's own fix) can never recover
    // from an already-too-small value, since it would only ever replace the
    // answer with something even smaller. The too-small branch instead
    // mirrors the PRIMARY windowed search's own selection rule -- best
    // longitudinal ("beside") alignment, not smallest lateral distance --
    // run across the whole chain instead of the local window; that's what
    // correctly identifies "same side as the sample" regardless of which
    // leg happens to sit closer in raw Euclidean terms, unlike the
    // Euclidean-only fallback above.
    constexpr double kMinPlausibleLateralDistance = 0.8;  // meters
    if (bestIdx >= 0 && bestLateral > kChainDistanceSanityBound)
    {
        // Too LARGE: the windowed search missed a genuinely closer point --
        // find the smallest lateral distance among the (still longitudinally-
        // filtered, so still "beside" the sample, not diagonally ahead of
        // it) candidates across the WHOLE chain.
        double globalBestLateral = bestLateral;
        int globalBestIdx = bestIdx;
        for (int i = 0; i < chainSize; ++i)
        {
            const double dx = _chain[static_cast<size_t>(i)].x - _sample.x;
            const double dy = _chain[static_cast<size_t>(i)].y - _sample.y;
            const double longitudinal = dx * _tangentX + dy * _tangentY;
            if (std::abs(longitudinal) >= kLongitudinalWindow)
            {
                continue;
            }
            const double lateral = std::abs(-dx * _tangentY + dy * _tangentX);
            if (lateral < globalBestLateral)
            {
                globalBestLateral = lateral;
                globalBestIdx = i;
            }
        }
        bestLateral = globalBestLateral;
        bestIdx = globalBestIdx;
    }
    else if (bestIdx >= 0 && bestLateral < kMinPlausibleLateralDistance)
    {
        // Too SMALL: a wrong-leg fold-back candidate, not a genuinely
        // narrow section (see this block's own comment above) -- a plain
        // "find something smaller" search, like the too-large branch above
        // uses, can never fix this (it would only ever replace the answer
        // with something EVEN smaller). Instead, mirror the PRIMARY
        // windowed search's own selection rule -- best longitudinal
        // ("beside") alignment, not smallest lateral distance -- but run it
        // across the WHOLE chain instead of the local window. That's the
        // same criterion already used, just not fold-back-limited to a
        // small index range around a cursor that may itself have already
        // drifted onto the wrong leg, and it's what correctly identifies
        // "same side as the sample" REGARDLESS of which leg happens to sit
        // closer in raw Euclidean terms.
        double globalBestLongAbs = std::numeric_limits<double>::max();
        double globalBestLateral = bestLateral;
        int globalBestIdx = bestIdx;
        for (int i = 0; i < chainSize; ++i)
        {
            const double dx = _chain[static_cast<size_t>(i)].x - _sample.x;
            const double dy = _chain[static_cast<size_t>(i)].y - _sample.y;
            const double longitudinal = dx * _tangentX + dy * _tangentY;
            const double longAbs = std::abs(longitudinal);
            if (longAbs < kLongitudinalWindow && longAbs < globalBestLongAbs)
            {
                globalBestLongAbs = longAbs;
                globalBestLateral = std::abs(-dx * _tangentY + dy * _tangentX);
                globalBestIdx = i;
            }
        }
        bestLateral = globalBestLateral;
        bestIdx = globalBestIdx;
    }

    if (bestIdx >= 0)
    {
        _cursor = static_cast<size_t>(bestIdx);
    }
    return bestLateral;
}
}  // namespace

std::vector<CorridorSample> ComputeCorridor(const std::vector<PathPoint> &splineSamples,
                                             const std::vector<WorldCone> &blue,
                                             const std::vector<WorldCone> &yellow,
                                             const std::vector<WorldCone> &orange,
                                             double safetyMargin, double minHalfWidth,
                                             double maxHalfWidth, bool closed)
{
    std::vector<CorridorSample> result;
    if (splineSamples.empty())
    {
        return result;
    }
    result.reserve(splineSamples.size());

    // Initial heading for BuildValidatedChainPair's own fold-back
    // protection (see its own comment in path_utils.hpp) -- the local
    // tangent between the first two spline
    // samples, the best directional estimate available here (no vehicle
    // pose to draw on directly). Falls back to no initial heading only if
    // there's genuinely just one sample to work with (degenerate input).
    bool haveInitialHeading = false;
    double initialHeadingX = 0.0, initialHeadingY = 0.0;
    if (splineSamples.size() >= 2)
    {
        const double dx = splineSamples[1].x - splineSamples[0].x;
        const double dy = splineSamples[1].y - splineSamples[0].y;
        const double len = std::sqrt(dx * dx + dy * dy);
        if (len > 1e-6)
        {
            haveInitialHeading = true;
            initialHeadingX = dx / len;
            initialHeadingY = dy / len;
        }
    }
    // closed=true builds these chains over the ENTIRE discovered map (see
    // kClosedChainMaxHop's own comment for why that needs a larger hop
    // cap than the windowed open pipeline's default kMaxPairDistance).
    const double chainMaxHop = closed ? kClosedChainMaxHop : kMaxPairDistance;
    // See BuildValidatedChainPair's own comment for why closed=true needs
    // the widened 150.0 threshold (raw, cone-spaced chain, same category as
    // centerline_extractor.cpp's own fix) while the open/windowed path
    // keeps the default.
    const double chainMaxHeadingReversalDeg = closed ? 150.0 : 90.0;
    // CROSS-COLOR DIVERGENCE CHECK (2026-09-08, user report: "the planned
    // path... too wide" -- confirmed live as THIS function's own symptom:
    // /planning/debug_corridor_left and _right symmetrically ballooning to
    // ~2x this track's normal width right after the hairpin. Root-caused
    // via direct replay against a real capture: this function's own
    // yellowChain took a wrong first hop back into the hairpin's own
    // just-exited leg while blueChain correctly continued forward, so
    // every per-sample lateral-distance measurement below was comparing
    // against two chains tracing different, non-parallel parts of the
    // track -- see BuildValidatedChainPair's own header comment
    // (path_utils.hpp) for the general mechanism, identical to
    // centerline_extractor.cpp's two extractors' own version of this same
    // bug (same heading-unchecked-hop-0 structure in the shared
    // OrderWaypointsByTraversal this function's own BuildChain wraps).
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
    fsd::ChainPair validatedChains =
        BuildValidatedChainPair(toPoints(blue), toPoints(yellow), splineSamples.front(), chainMaxHop,
                                 haveInitialHeading, initialHeadingX, initialHeadingY, chainMaxHeadingReversalDeg);
    const std::vector<PathPoint> &blueChain = validatedChains.blue;
    const std::vector<PathPoint> &yellowChain = validatedChains.yellow;
    size_t blueCursor = 0;
    size_t yellowCursor = 0;
    const int sampleCount = static_cast<int>(splineSamples.size());

    for (size_t i = 0; i < splineSamples.size(); ++i)
    {
        // Local tangent via central difference. Closed: wraps modulo
        // sampleCount, so index 0's "prev" is the last sample and the last
        // sample's "next" is index 0 -- see this function's own _closed
        // comment for why there's no real start/end to clamp at. Open
        // (default): unchanged, clamp at both ends via forward/backward
        // difference exactly as before.
        const PathPoint &prev = closed ? splineSamples[static_cast<size_t>(
                                              (static_cast<int>(i) - 1 + sampleCount) % sampleCount)]
                                        : splineSamples[i == 0 ? i : i - 1];
        const PathPoint &next = closed
            ? splineSamples[static_cast<size_t>((static_cast<int>(i) + 1) % sampleCount)]
            : splineSamples[i + 1 < splineSamples.size() ? i + 1 : i];
        double tx = next.x - prev.x;
        double ty = next.y - prev.y;
        const double tlen = std::sqrt(tx * tx + ty * ty);
        if (tlen > 1e-9)
        {
            tx /= tlen;
            ty /= tlen;
        }
        else
        {
            tx = 1.0;
            ty = 0.0;
        }

        double blueDist =
            NearestChainLateralDistance(blueChain, blueCursor, splineSamples[i], tx, ty, closed);
        double yellowDist =
            NearestChainLateralDistance(yellowChain, yellowCursor, splineSamples[i], tx, ty, closed);

        // WIDTH CROSS-CHECK (2026-09-05, user report: car flipped upside
        // down, rollover confirmed at a real cone). Root-caused via direct
        // ground-truth comparison against /estimated_landmarks: the BLUE
        // landmark near that corner had drifted ~0.5-1.6m inward (toward
        // the spline path) after two closed-loop laps of thinning
        // observation counts, while the paired YELLOW landmark stayed
        // accurate (~0.2m error, ordinary EKF noise). blueDist there read
        // ~0.92m against a true ~1.47m -- NearestChainLateralDistance's
        // own fold-back sanity checks don't catch this (it's not a
        // wrong-leg chain hop, the nearest blue point really is the
        // nearest blue LANDMARK, that landmark's own estimated position is
        // just wrong) -- so the corridor faithfully built a narrow left
        // bound from genuinely bad input data, the racing line hugged it,
        // and the car clipped the real (undrifted) cone position at
        // closed-loop speed. Cross-checking against the known, repeatedly
        // ground-truth-confirmed total track width catches exactly this:
        // a single drifted landmark shows up as blueDist+yellowDist
        // deviating from kAssumedTrackWidth even though each side's own
        // sanity checks pass individually. Correct by trusting whichever
        // side's raw reading is LARGER (this drift mechanism only ever
        // shrinks a reading, pulling toward the path, never grows one) and
        // reconstructing the other from the known total instead of its own
        // now-suspect value -- reconstructs 3.0 - 1.53 = 1.47m in the
        // confirmed live case, matching ground truth exactly. Only runs
        // when both sides have real data (a missing side already gets its
        // own, separate minHalfWidth handling below) and only corrects the
        // suspect side, never invents extra room past what the trusted
        // side's own direct measurement already allows.
        if (blueDist >= 0.0 && yellowDist >= 0.0 &&
            std::abs((blueDist + yellowDist) - kAssumedTrackWidth) > kWidthSanityTolerance)
        {
            if (blueDist > yellowDist)
            {
                yellowDist = std::max(0.0, kAssumedTrackWidth - blueDist);
            }
            else
            {
                blueDist = std::max(0.0, kAssumedTrackWidth - yellowDist);
            }
        }

        // Independent per side -- see CorridorSample's own comment for why
        // this replaced a single min(blueDist, yellowDist) bound. The floor
        // (minHalfWidth) only applies when THAT side's chain has no data at
        // all this sample; a real, valid distance measurement is trusted
        // even when it's below the floor (that's the actual track being
        // tight there, not sensor noise -- see this file's own
        // ComputeCorridor declaration comment in corridor.hpp).
        double leftBound = (blueDist < 0.0) ? minHalfWidth : std::max(0.0, blueDist - safetyMargin);
        double rightBound = (yellowDist < 0.0) ? minHalfWidth : std::max(0.0, yellowDist - safetyMargin);

        // Tighten (never widen) against nearby orange gate cones -- see
        // this function's declaration in corridor.hpp for why blue/yellow
        // alone can leave the corridor wider than the actual gate passage.
        // Plain per-cone scan, no chain/cursor: orange is sparse enough
        // (trackdrive.sdf has exactly 2) that this is cheap and there's no
        // fold-back ambiguity to guard against.
        //
        // BUG FIX (2026-09-03, user report: "racing line and spline...
        // warped" -- confirmed live: /planning/debug_corridor_left vs
        // _right collapsed to an EXACT 0.00m combined width across 13+
        // consecutive samples right on the start/finish straight, near a
        // real orange cone at approx (57.3,-4.25)). Root cause: this used
        // to apply the SAME UNSIGNED lateral distance from ONE orange cone
        // to tighten BOTH leftBound and rightBound via min() -- but a gate
        // cone sits on only ONE side of the track, same as any blue/yellow
        // boundary cone. If that cone's true lateral offset happens to be
        // small on its own side (a gate genuinely narrower than the open
        // track, exactly the scenario this whole check exists to catch),
        // the old code ALSO force-tightened the OPPOSITE side down to that
        // same small value even though the track was perfectly normal
        // width there -- collapsing the corridor to near-zero on BOTH
        // sides from a single cone on only one of them. Fixed by using the
        // SIGNED lateral offset (left-positive, same convention as
        // blueDist/yellowDist and ClampToCorridor above/elsewhere in this
        // file) to determine which side this specific cone is actually on,
        // and only tightening that one side -- matching exactly how blue
        // (always left) and yellow (always right) are already handled
        // above, just per-cone instead of per-color since orange doesn't
        // carry that distinction on its own.
        // rawOrangeLeft/RightDist track the nearest orange cone's own RAW
        // (pre-margin) lateral distance per side, alongside the margin-
        // adjusted leftBound/rightBound tightening below -- needed by the
        // safety-floor fix further down, which must cap against true
        // available space (including any real orange gate) rather than
        // reverting to a blue/yellow-only view that could ignore a real,
        // nearby gate cone. Sentinel -1.0 means "no orange cone was close
        // enough on this side to matter", same convention as blueDist/
        // yellowDist's own "no data" sentinel.
        double rawOrangeLeftDist = -1.0;
        double rawOrangeRightDist = -1.0;
        for (const WorldCone &o : orange)
        {
            const double odx = o.x - splineSamples[i].x;
            const double ody = o.y - splineSamples[i].y;
            const double oLongitudinal = odx * tx + ody * ty;
            if (std::abs(oLongitudinal) < kLongitudinalWindow)
            {
                const double oLateral = -odx * ty + ody * tx;  // left-positive
                const double oAbsLateral = std::abs(oLateral);
                const double oBound = std::max(0.0, oAbsLateral - safetyMargin);
                if (oLateral >= 0.0)
                {
                    leftBound = std::min(leftBound, oBound);
                    rawOrangeLeftDist =
                        (rawOrangeLeftDist < 0.0) ? oAbsLateral : std::min(rawOrangeLeftDist, oAbsLateral);
                }
                else
                {
                    rightBound = std::min(rightBound, oBound);
                    rawOrangeRightDist =
                        (rawOrangeRightDist < 0.0) ? oAbsLateral : std::min(rawOrangeRightDist, oAbsLateral);
                }
            }
        }

        leftBound = std::clamp(leftBound, 0.0, maxHalfWidth);
        rightBound = std::clamp(rightBound, 0.0, maxHalfWidth);

        // SAFETY FLOOR (2026-09-04, user report: "the way the planned path
        // gets constructed as soon as we get near the orange cones has
        // regressed" -- confirmed live: leftBound legitimately hit 0 from
        // the orange-tightening fix above (working as intended), but
        // rightBound INDEPENDENTLY also collapsed to exactly 0 a few
        // samples later -- a genuine, separate bug in
        // NearestChainLateralDistance's yellow-chain tracking right near
        // the gate, unrelated to orange at all (this loop's own oBound for
        // the far orange cone was measured ~1.85-2.25m there, nowhere near
        // tight enough to explain it). Root-causing that chain-tracking
        // bug specifically was not resolved this pass -- this is a direct,
        // narrower safety net instead: a car has nonzero width, so a
        // corridor whose COMBINED width drops below what two independent
        // per-side floors would already allow is not safe data to hand to
        // the racing-line optimizer regardless of WHICH upstream mechanism
        // produced it (this one, or a different one found later) --
        // twice now, a literal-zero corridor here has led directly to
        // EnforceMinClearance/TurnRadius pushing racing-line points
        // erratically and the car getting stuck. Restores BOTH sides to
        // minHalfWidth together (not just clamping the collapsed one back
        // up alone) specifically so the two sides stay geometrically
        // consistent with each other post-fix, rather than pairing a
        // now-floored side against whatever the other side's own
        // (possibly also-suspect) raw value happened to be.
        // FIXED (2026-09-05, user goal: 10 laps, zero stuck events --
        // confirmed live, THREE separate times in one night, the car
        // ending up ~0.96-0.97m from a real cone at a tight/sustained turn
        // and physically wedging, reverse recovery DISABLED per FS rules).
        // Root cause: this floor restored BOTH bounds to minHalfWidth
        // UNCONDITIONALLY whenever their margin-adjusted sum was small --
        // correct when that smallness is a chain-tracking artifact (the
        // 2026-09-04 bug this floor was originally added for, where the
        // TRUE track width there was normal), but WRONG when the track is
        // genuinely narrow there (a real hairpin apex): blindly forcing
        // minHalfWidth in the second case can claim more space than
        // actually exists, up to and including placing the corridor edge
        // PAST a real cone -- exactly what a racing line hugging that edge
        // then does too, which is exactly the wedge signature just
        // confirmed live.
        // Fix: cap each restored bound at that side's own RAW (pre-margin)
        // distance to the nearest REAL constraint -- the blue/yellow chain
        // distance (blueDist/yellowDist, already computed above via
        // NearestChainLateralDistance, which already has this file's own
        // fold-back protection baked in) AND the nearest orange gate cone
        // on that side (rawOrangeLeft/RightDist, tracked in the loop just
        // above) -- rather than blindly trusting minHalfWidth. Both must be
        // considered: capping by blue/yellow alone would ignore a real,
        // nearby gate cone and could recreate the 2026-09-03 "path routes
        // through the gate cones" regression this file's own orange-
        // tightening logic above exists to prevent. This preserves the
        // ORIGINAL floor's exact behavior for genuine "no data at all"
        // sides (both distances are the -1.0 sentinel -- minHalfWidth is
        // still the only sane default there) while now also protecting the
        // genuinely-narrow case: whenever ANY real measurement exists, this
        // bound can never claim more room than the tightest of them
        // actually allows. Deliberately reuses distances this function
        // already trusts elsewhere (not a fresh global cone scan), so this
        // doesn't reintroduce the exact fold-back-ambiguity risk this
        // file's own chain/cursor machinery exists to avoid -- a raw
        // nearest-cone scan at a hairpin can't tell "the real boundary
        // cone" from "a physically-close cone on the OTHER leg of the
        // fold", which is precisely the class of bug already fixed
        // elsewhere in this file via BuildChain/NearestChainLateralDistance
        // rather than by re-solving it here.
        // PER-SIDE FIX (2026-09-05, same rollover incident as the width
        // cross-check above): this used to fire on the COMBINED sum and
        // then overwrite BOTH bounds down to min(minHalfWidth, cap) --
        // correct for the side that was actually deficient, but it also
        // dragged the OTHER, perfectly healthy side down to minHalfWidth
        // even when that side's own margin-adjusted bound was already
        // larger. Confirmed live: rightBound had already computed a real,
        // trustworthy ~0.58m (from an accurate yellow landmark) when
        // leftBound collapsed to 0 (from the drifted blue landmark above)
        // -- the old code still forced rightBound down to exactly 0.5,
        // throwing away ~0.08m of real, verified-safe room for no reason.
        // Now applied independently per side, and only ever RAISES a
        // bound that's actually below minHalfWidth -- a side already at or
        // above it is left untouched, never clamped down to exactly the
        // floor value.
        if (leftBound < minHalfWidth)
        {
            double leftCap = minHalfWidth;
            if (blueDist >= 0.0)
            {
                leftCap = std::min(leftCap, blueDist - kConeRadius);
            }
            if (rawOrangeLeftDist >= 0.0)
            {
                leftCap = std::min(leftCap, rawOrangeLeftDist - kConeRadius);
            }
            leftBound = std::max(leftBound, std::max(0.0, leftCap));
        }
        if (rightBound < minHalfWidth)
        {
            double rightCap = minHalfWidth;
            if (yellowDist >= 0.0)
            {
                rightCap = std::min(rightCap, yellowDist - kConeRadius);
            }
            if (rawOrangeRightDist >= 0.0)
            {
                rightCap = std::min(rightCap, rawOrangeRightDist - kConeRadius);
            }
            rightBound = std::max(rightBound, std::max(0.0, rightCap));
        }

        result.push_back(CorridorSample{splineSamples[i], tx, ty, leftBound, rightBound});
    }

    // Smooth each raw per-sample bound independently -- see this function's
    // own declaration in corridor.hpp for why it's piecewise by
    // construction. Closed: wraps modulo result.size(), same reasoning as
    // the tangent and chain-search wraparound above -- otherwise the
    // smoothing window would clamp at index 0/size-1 even though those
    // aren't real boundaries on a closed loop, leaving a visible seam in
    // the smoothed bound right at wherever the sample array happens to
    // start.
    std::vector<double> smoothedLeft(result.size());
    std::vector<double> smoothedRight(result.size());
    const int resultCount = static_cast<int>(result.size());
    for (size_t i = 0; i < result.size(); ++i)
    {
        double sumL = 0.0, sumR = 0.0;
        int count = 0;
        if (closed)
        {
            for (int off = -kSmoothingWindow; off <= kSmoothingWindow; ++off)
            {
                const int j = ((static_cast<int>(i) + off) % resultCount + resultCount) % resultCount;
                sumL += result[static_cast<size_t>(j)].leftBound;
                sumR += result[static_cast<size_t>(j)].rightBound;
                ++count;
            }
        }
        else
        {
            const size_t lo = (i >= static_cast<size_t>(kSmoothingWindow)) ? i - static_cast<size_t>(kSmoothingWindow) : 0;
            const size_t hi = std::min(result.size() - 1, i + static_cast<size_t>(kSmoothingWindow));
            for (size_t j = lo; j <= hi; ++j)
            {
                sumL += result[j].leftBound;
                sumR += result[j].rightBound;
                ++count;
            }
        }
        smoothedLeft[i] = sumL / static_cast<double>(count);
        smoothedRight[i] = sumR / static_cast<double>(count);
    }
    for (size_t i = 0; i < result.size(); ++i)
    {
        result[i].leftBound = smoothedLeft[i];
        result[i].rightBound = smoothedRight[i];
    }

    return result;
}
}  // namespace fsd
