#pragma once

// The host time guest deadlines are paced against: steady_clock, or a
// multiple of it for unattended runs faster than real time. Everything that
// paces the guest reads now() and waits until real(), so all of it speeds
// up together.
//
// Stepped (step()), the clock is a stepper's alone: it stands still until
// the stepper moves it on, so the guest sees the same times however long
// the host takes, and a recording made on a busy machine is the one made on
// an idle one. The guest's thread marks its waits on the clock with Wait,
// and the stepper moves the clock on only once the guest waits, having
// looked at the clock since it last moved (settled()).

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

namespace ps2x::host_clock
{
    using Clock = std::chrono::steady_clock;

    namespace detail
    {
        inline double speed = 1.0;
        inline Clock::time_point origin = Clock::now();

        inline std::atomic<bool> stepped{false};
        inline std::atomic<Clock::rep> at{0}; // stepped: now(), from the epoch
        inline std::atomic<uint64_t> steps{0};
        // The guest's wait: whether it waits, the steps it had seen when it
        // looked, and the time it waits for.
        inline std::atomic<bool> waiting{false};
        inline std::atomic<uint64_t> seen{0};
        inline std::atomic<Clock::rep> until{0};
        // How often a wait past the stepped clock looks again.
        constexpr auto kLook = std::chrono::microseconds(100);
    }

    // Before the runtime starts; time keeps running from now.
    inline void setSpeed(double speed)
    {
        detail::origin = Clock::now();
        detail::speed = speed;
    }

    inline double speed() { return detail::speed; }

    // Before the runtime starts: the clock stands at now until stepTo().
    inline void step()
    {
        detail::at.store(Clock::now().time_since_epoch().count());
        detail::stepped.store(true);
    }

    inline bool stepped() { return detail::stepped.load(std::memory_order_relaxed); }

    inline Clock::time_point now()
    {
        if (stepped())
            return Clock::time_point(Clock::duration(detail::at.load()));
        const Clock::time_point real = Clock::now();
        if (detail::speed == 1.0)
            return real;
        return detail::origin + std::chrono::duration_cast<Clock::duration>((real - detail::origin) * detail::speed);
    }

    // The real time at which now() reaches t, to wait until. Rounded up, so
    // now() on waking is never short of t. Stepped, a t not reached yet has
    // no real time: a moment on, to look again. A waiter loops until now()
    // reaches t, never acting on an early wake.
    inline Clock::time_point real(Clock::time_point t)
    {
        if (t == Clock::time_point::max())
            return t;
        if (stepped())
            return Clock::now() + (now() < t ? std::chrono::duration_cast<Clock::duration>(detail::kLook)
                                             : Clock::duration{});
        if (detail::speed == 1.0)
            return t;
        return detail::origin + std::chrono::ceil<Clock::duration>((t - detail::origin) / detail::speed);
    }

    // The guest's thread waiting on the clock until `t` (or for something
    // else, with max()). Made before it looks (reads now(), or what the
    // stepper moves) to decide whether to, so that look sees the clock as of
    // the steps marked; block() once it has decided to, just before it
    // blocks. Settled only from there: a look that goes on to work never
    // is. One at a time, by the guest's thread alone; a no-op unstepped.
    class Wait
    {
    public:
        explicit Wait(Clock::time_point t = Clock::time_point::max())
        {
            if (!stepped())
                return;
            m_seen = detail::steps.load();
            detail::until.store(t.time_since_epoch().count());
        }
        void block()
        {
            if (!stepped())
                return;
            detail::seen.store(m_seen);
            detail::waiting.store(true);
        }
        ~Wait()
        {
            if (stepped())
                detail::waiting.store(false);
        }
        Wait(const Wait &) = delete;
        Wait &operator=(const Wait &) = delete;

    private:
        uint64_t m_seen = 0;
    };

    // Stepped: whether the guest waits on the clock, having looked at it
    // since it last moved, and if so the time it waits for.
    inline bool settled(Clock::time_point &waitsFor)
    {
        if (!detail::waiting.load() || detail::seen.load() != detail::steps.load())
            return false;
        waitsFor = Clock::time_point(Clock::duration(detail::until.load()));
        // Still the same wait: it may have ended and another begun between.
        return detail::waiting.load() && detail::seen.load() == detail::steps.load();
    }

    // Stepped: the clock on to t, for the guest to go on to it.
    inline void stepTo(Clock::time_point t)
    {
        detail::at.store(t.time_since_epoch().count());
        detail::steps.fetch_add(1u);
    }
}
