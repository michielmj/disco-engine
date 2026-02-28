#define DEBUG_ENABLED

#ifndef DISCO_PREDECESSOREVENTQUEUE_H
#define DISCO_PREDECESSOREVENTQUEUE_H

#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

// Forward declare PyObject to avoid including Python.h in a public header.
struct _object;
using PyObject = _object;

namespace disco {

using Headers = std::map<std::string, std::string>;

struct PredecessorEvent {
    double epoch{};
    PyObject* data{nullptr};
    Headers headers{};

    PredecessorEvent() noexcept;

    // Construct from BORROWED PyObject* (will INCREF).
    PredecessorEvent(double epoch_,
                     PyObject* data_,
                     Headers headers_);

    PredecessorEvent(const PredecessorEvent& other);
    PredecessorEvent& operator=(const PredecessorEvent& other);

    PredecessorEvent(PredecessorEvent&& other) noexcept;
    PredecessorEvent& operator=(PredecessorEvent&& other) noexcept;

    ~PredecessorEvent();

    // Transfer ownership of the PyObject* out of this event (prevents DECREF in destructor).
    PyObject* release_data() noexcept;

    bool operator<(const PredecessorEvent& other) const {
        // priority_queue puts "largest" on top; invert for min-heap semantics
        return epoch > other.epoch;
    }
};

class PredecessorEventQueue {
public:
    PredecessorEventQueue();

    bool push(double epoch, PyObject* data, Headers& headers);

    std::vector<PredecessorEvent> pop();

    std::vector<PredecessorEvent> popAll();

    bool promise(uint64_t seqnr, double epoch, uint64_t num_events);

    bool tryNextEpoch();

    double getEpoch() const;

    double getNextEpoch() const;

    bool waitingForPromise() const;

    bool waitingForEvents() const;

    bool empty() const;

    std::string dump() const;

private:
    bool _tryNextEpoch();          // requires _mtx held
    bool _increment(double epoch); // requires _mtx held

    double epochAfter(uint64_t seqnr);   // requires _mtx held
    double epochBefore(uint64_t seqnr);  // requires _mtx held

private:
    mutable std::mutex _mtx;

    std::map<double, uint64_t> _event_count;
    std::map<uint64_t, uint64_t> _num_events;
    std::map<uint64_t, double> _epochs;

    std::priority_queue<PredecessorEvent, std::deque<PredecessorEvent>> _events;

    double _epoch;
    double _next_epoch;
    uint64_t _seqnr;
};

}  // namespace disco

#endif  // DISCO_PREDECESSOREVENTQUEUE_H
