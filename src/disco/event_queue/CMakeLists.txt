# src/disco/event_queue/CMakeLists.txt
add_pybind11_extension(
    TARGET      event_queue_core
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/_core.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/EventQueue.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/PredecessorEventQueue.cpp
    DESTINATION disco/event_queue
)