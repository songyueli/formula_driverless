#pragma once

#include <vector>

#include "corridor.hpp"

// Stage 4 (final) of the landmark-based racing-line pipeline: the actual
// least-curvature racing line, constrained to stay within stage 3's
// corridor.
namespace fsd
{
// REWRITTEN 2026-09-04 (user report: "it doesn't appear to be properly
// stretching the curve to the left and right debug corridors, which is the
// whole point of those corridors"). The previous approach -- a bounded
// iterative neighbor-pull (pull each point toward the midpoint of its
// +/-neighborRadius neighbors, project the candidate's offset from its OWN
// fixed corridor sample into longitudinal/lateral and clamp both) -- went
// through several live-tuning cycles the same day trying to fix it:
// neighborRadius=1 converged too slowly to use meaningful corridor width
// (measured ~5.7% average utilization -- essentially the centerline);
// widening the radius fixed that at one location (the hairpin) but broke
// two others (unwarranted drift on a straight section; points ending up
// outside their own corridor's ACTUAL bound once they'd drifted far enough
// along-track that their fixed-index clamp target went stale); clamping
// the drift then produced a THIRD failure -- a visibly jagged, non-smooth
// line, confirmed via an OFFLINE replay against captured live corridor
// data to be a genuine sawtooth (122+ heading-direction sign flips across
// the loop, vs ~10 for the timid radius=1 baseline), not a perception
// issue. Each fix in that cycle patched one symptom of the same underlying
// mismatch: a heuristic "pull toward neighbors" proxy, external corridor
// clamping bolted on afterward, is the wrong tool for "shape a smooth
// curvature-minimizing line that provably never exits a per-sample bound."
//
// This version solves the ACTUAL problem directly: minimum-curvature
// racing line as a convex QP, solved via Successive Over-Relaxation (SOR).
// Each corridor sample i is represented by a single SCALAR lateral offset
// y_i from its own FIXED point, along that sample's own FIXED left-normal
// direction -- P_i = sample_i.point + y_i * normal_i. This is the key
// structural fix: a point has NO longitudinal degree of freedom at all, so
// it can never drift away from its own assigned corridor sample and get
// checked against a stale/wrong bound -- the failure mode above is
// eliminated by construction, not capped after the fact. The objective is
// the standard discrete curvature penalty, sum_i |P_{i-1} - 2 P_i +
// P_{i+1}|^2, minimized subject to y_i in [-rightBound_i, leftBound_i] --
// a convex QP with box constraints. SOR (a coordinate-wise closed-form
// update per point, each sweep, scaled by _omega for faster convergence
// than plain Gauss-Seidel) converges MONOTONICALLY for this problem, which
// is what actually prevents the oscillation/jaggedness the old approach
// showed -- confirmed via the same offline replay: heading-direction sign
// flips dropped to the same ~10-20 range as the timid radius=1 baseline
// (not the 122+ of the old wide-radius attempt), while corridor-width
// utilization climbed well past that baseline's 5.7% and kept improving
// with more sweeps, all with ZERO corridor-bound violations anywhere
// across the full captured closed loop (re-checked against each output
// point's own TRUE nearest corridor sample, not just its assigned index --
// the exact check that exposed the old approach's staleness bug).
//
// Why this produces a REAL racing-line shape, not just smoothing:
// minimizing curvature over a bounded corridor is the textbook formulation
// of "the racing line" itself -- it naturally pulls the path toward
// whichever side lets it stay straightest, which is the corridor's outside
// edge through the entry/exit of a bend and its inside edge near the
// apex (confirmed at the hairpin in the same offline replay: a clean,
// monotonic sweep from ~95% of the available width on one side, easing
// through center near the true apex, out to the other side).
//
// _sweeps/_omega: SOR sweep count and over-relaxation factor. 2000
// sweeps at omega=1.9 is what the offline validation above was run at
// (converged: heading-change metrics stop moving well before that count,
// utilization keeps climbing very slowly past it -- diminishing returns,
// not still finding a meaningfully different answer). This runs ONCE per
// lap completion (not per-cycle), so the cost of extra sweeps is
// essentially free -- prefer erring toward more sweeps over a smaller,
// faster-but-less-converged count. omega above ~1.9 was not swept in the
// offline validation; retest before pushing higher.
// _closed: false (default) preserves the original contract -- index 0 and
// the last index stay fixed at their own corridor sample's centerline
// (y=0, never updated), same as the old code's fixed anchors. true is for
// the full-track closed loop (see lap_detector.hpp): every index is
// solved, with neighbors looked up modulo corridor.size().
std::vector<PathPoint> OptimizeRacingLine(const std::vector<CorridorSample> &corridor, int sweeps,
                                           double omega, bool closed = false);

// Re-projects an already-computed point back inside _sample's corridor
// bound, using the exact same longitudinal/lateral decomposition
// OptimizeRacingLine's own inner loop uses (see its .cpp) -- exposed
// separately so callers that modify a racing-line point AFTER
// optimization (e.g. planning.cpp's cross-cycle jitter-smoothing cache)
// can restore the corridor guarantee that modification may have broken.
// A blend between this cycle's freshly-clamped point and a PAST cycle's
// point is not itself guaranteed to stay in bounds if the corridor
// narrowed between those two cycles (more/better-localized landmarks
// tightening the measured track width) -- re-clamping here is what
// prevents that from ever driving the published line into a cone.
PathPoint ClampToCorridor(const PathPoint &point, const CorridorSample &sample);
}  // namespace fsd
