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
// Deliberately does NOT carry a currentSpeed/velocity field yet: nothing
// reads one today (pure pursuit is a purely geometric controller -- see
// pure_pursuit_controller.cpp), and control.cpp doesn't subscribe to any
// vehicle-state topic at all right now. Adding an unused field here would
// be speculative plumbing with nothing to populate it correctly; add it
// (and the subscription that feeds it) together, when a controller that
// actually needs it gets written.
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
