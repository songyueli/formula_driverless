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
}  // namespace
std::vector<PathPoint> TwoPointMidpointExtractor(const std::vector<WorldCone> &blue,
                                                  const std::vector<WorldCone> &yellow,
                                                  const Pose2D &vehiclePose)
{
    // Same mutual-nearest-neighbor requirement as
    // path_generator.cpp's NearestPairMidpointPath, and for the same
    // reason: at a sharp corner, the inside boundary's tighter cone
    // spacing means a blue landmark's own nearest yellow can easily be one
    // a DIFFERENT blue landmark is a strictly better match for -- a
    // genuine cross-pair, not noise. Requiring the match to be symmetric
    // costs an extra O(blue) reverse scan per candidate but means a bad
    // pair is skipped rather than accepted.
    std::vector<PathPoint> waypoints;
    waypoints.reserve(blue.size());
    for (const auto &l : blue)
    {
        const WorldCone *nearest = nullptr;
        double bestDistSq = kMaxPairDistance * kMaxPairDistance;
        for (const auto &r : yellow)
        {
            const double dx = r.x - l.x;
            const double dy = r.y - l.y;
            const double distSq = dx * dx + dy * dy;
            if (distSq < bestDistSq)
            {
                bestDistSq = distSq;
                nearest = &r;
            }
        }
        if (!nearest)
        {
            continue;
        }

        const WorldCone *reciprocal = nullptr;
        double reciprocalBestDistSq = kMaxPairDistance * kMaxPairDistance;
        for (const auto &l2 : blue)
        {
            const double dx = l2.x - nearest->x;
            const double dy = l2.y - nearest->y;
            const double distSq = dx * dx + dy * dy;
            if (distSq < reciprocalBestDistSq)
            {
                reciprocalBestDistSq = distSq;
                reciprocal = &l2;
            }
        }
        if (reciprocal != &l)
        {
            continue;  // not a mutual match -- likely a cross-pair, skip rather than accept
        }

        waypoints.push_back(PathPoint{(l.x + nearest->x) / 2.0, (l.y + nearest->y) / 2.0});
    }

    // Order into a travel-order chain starting from the vehicle's own
    // WORLD position (not the body-frame origin -- these are world-frame
    // midpoints), same reasoning as path_generator.cpp's own ordering step
    // for why a plain x-sort breaks at a hairpin.
    return OrderWaypointsByTraversal(std::move(waypoints), PathPoint{vehiclePose.x, vehiclePose.y},
                                      kMaxPairDistance, /*_haveInitialHeading=*/true,
                                      std::cos(vehiclePose.yaw), std::sin(vehiclePose.yaw));
}

std::vector<PathPoint> ClosedLoopMidpointExtractor(const std::vector<WorldCone> &blue,
                                                    const std::vector<WorldCone> &yellow,
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

    const PathPoint cursorStart{vehiclePose.x, vehiclePose.y};
    const double headingX = std::cos(vehiclePose.yaw);
    const double headingY = std::sin(vehiclePose.yaw);
    const std::vector<PathPoint> blueChain =
        OrderWaypointsByTraversal(toPoints(blue), cursorStart, kClosedChainMaxHop,
                                   /*_haveInitialHeading=*/true, headingX, headingY);
    const std::vector<PathPoint> yellowChain =
        OrderWaypointsByTraversal(toPoints(yellow), cursorStart, kClosedChainMaxHop,
                                   /*_haveInitialHeading=*/true, headingX, headingY);
    if (blueChain.empty() || yellowChain.empty())
    {
        return {};
    }

    std::vector<PathPoint> midpoints;
    midpoints.reserve(blueChain.size());
    const int yellowSize = static_cast<int>(yellowChain.size());
    size_t yellowCursor = 0;
    for (const auto &b : blueChain)
    {
        const int lo = std::max(0, static_cast<int>(yellowCursor) - kClosedChainSearchWindow);
        const int hi = std::min(yellowSize - 1, static_cast<int>(yellowCursor) + kClosedChainSearchWindow);
        int bestIdx = -1;
        double bestDistSq = std::numeric_limits<double>::max();
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
        if (bestIdx < 0)
        {
            continue;  // yellowChain ran out on this side -- skip, don't force a bad pair
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
    return midpoints;
}
}  // namespace fsd
