#pragma once

#include <chrono>
#include <cstdint>
#include <functional>

namespace fsd {

// RAII scope timer -- measures wall-clock duration of whatever scope it's
// constructed in and hands the result (whole microseconds) to a callback on
// destruction. Deliberately just wraps "however much work happens to be in
// the block" rather than anything specific to what that work currently is --
// perception/localization/planning/control each wrap their own per-cycle
// callback body with one of these, so if any of those bodies later grows
// (e.g. planning becoming a real path-planning algorithm, control becoming
// MPC), the timing keeps working without needing to change.
class ScopedTimer
{
public:
    explicit ScopedTimer(std::function<void(int64_t)> _onDone)
        : m_onDone(std::move(_onDone))
        , m_start(std::chrono::steady_clock::now())
    {
    }

    ~ScopedTimer()
    {
        const auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - m_start);
        m_onDone(elapsedUs.count());
    }

    ScopedTimer(const ScopedTimer &) = delete;
    ScopedTimer &operator=(const ScopedTimer &) = delete;

private:
    std::function<void(int64_t)> m_onDone;
    std::chrono::steady_clock::time_point m_start;
};

} // namespace fsd
