#pragma once

#include <vector>

// Shared input/output types for every controller implementation -- kept
// separate from any one controller's own header so a NEW controller
// doesn't need to include (or depend on) an existing one's header just to
// get these.

// Everything a controller needs to produce one drive command, for one
// cycle. A struct (not positional Compute() parameters) specifically so
// that a future controller needing MORE input (e.g. an MPC controller
// needing current speed for its dynamics model) extends this struct
// instead of changing every controller's Compute() signature and every
// call site -- see pure_pursuit_controller.hpp's header comment for the
// rest of that reasoning.
//
// Deliberately does NOT carry a currentSpeed/velocity field: nothing reads
// one (pure pursuit is a purely geometric controller for the STEERING
// decision -- see pure_pursuit_controller.cpp); worldX/worldY below is a
// narrower addition for a different purpose (see its own comment), not the
// vehicle-dynamics state a future MPC controller would still need to add
// separately.
struct ControlInputs
{
    struct Waypoint
    {
        double x;  // body frame, meters, +X = forward
        double y;  // body frame, meters, +Y = left
    };

    // planning.cpp's /planned_path, already ordered nearest-ahead-first.
    // Empty means "no path this cycle" (e.g. no cones currently detected).
    std::vector<Waypoint> path;

    // World-frame position from /estimated_pose, added specifically so
    // PurePursuitController's empty-path recovery (see its own kSweepYawRate/
    // kReverse* comments) can detect PHYSICAL immobility -- commanding
    // speed but not actually displacing, confirmed directly as a real
    // failure mode: a live test found /cmd_ackermann continuously
    // commanding forward speed (up to kMaxSweepSpeed) while true position
    // sat frozen for 100+ seconds, because the car was physically wedged
    // against a cone and this control stack had no reverse capability at
    // all to back out of contact. NOT used for the normal steering
    // decision (that stays purely geometric/body-frame, unaffected by
    // localization quality) -- only for this one recovery check, so a
    // temporarily-degraded pose estimate can only ever affect HOW FAST
    // recovery kicks in, never normal driving.
    double worldX = 0.0;
    double worldY = 0.0;
    bool poseValid = false;
};

// A controller's output, in the same body-velocity convention
// /cmd_ackermann already uses (see control.cpp's own topic-doc comment) --
// deliberately NOT a gz::msgs::Twist directly, so a controller class
// itself has no gz-transport/protobuf dependency; control.cpp (the only
// place that talks to gz-transport) converts this to a Twist message.
struct DriveCommand
{
    double speed;     // m/s, commanded forward speed
    double yawRate;   // rad/s, commanded yaw rate (NOT a steering angle --
                       // see control.cpp's topic-doc comment for why)
};
