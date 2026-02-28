macro(add_pybind11_extension)
    cmake_parse_arguments(EXT "" "TARGET;DESTINATION" "SOURCES" ${ARGN})

    pybind11_add_module(${EXT_TARGET} MODULE ${EXT_SOURCES})
    target_compile_features(${EXT_TARGET} PRIVATE cxx_std_17)

    if(APPLE)
        target_compile_options(${EXT_TARGET} PRIVATE -Wno-deprecated-declarations)
    endif()

    set_target_properties(${EXT_TARGET} PROPERTIES
        OUTPUT_NAME "_core"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    install(TARGETS ${EXT_TARGET} LIBRARY DESTINATION ${EXT_DESTINATION})  # <-- LIBRARY added
endmacro()


macro(add_cython_extension)
    cmake_parse_arguments(EXT "" "TARGET;SOURCE;DESTINATION" "" ${ARGN})

    set(_c_out "${CMAKE_CURRENT_BINARY_DIR}/${EXT_TARGET}.c")
    add_custom_command(
        OUTPUT  ${_c_out}
        COMMAND ${CYTHON_EXECUTABLE} --3str -o ${_c_out} ${EXT_SOURCE}
        DEPENDS ${EXT_SOURCE}
        COMMENT "Cythonizing ${EXT_SOURCE}"
    )

    Python_add_library(${EXT_TARGET} MODULE WITH_SOABI ${_c_out})
    set_target_properties(${EXT_TARGET} PROPERTIES
        OUTPUT_NAME              "_core"
        LINKER_LANGUAGE          C
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )

    target_include_directories(${EXT_TARGET} PRIVATE ${NUMPY_INCLUDE_DIR})
    if(DEFINED ENV{DISCO_NATIVE_MARCH})
        target_compile_options(${EXT_TARGET} PRIVATE
            $<$<NOT:$<C_COMPILER_ID:MSVC>>:-O3 -march=native>
            $<$<C_COMPILER_ID:MSVC>:/O2>
        )
    else()
        target_compile_options(${EXT_TARGET} PRIVATE
            $<$<NOT:$<C_COMPILER_ID:MSVC>>:-O3>
            $<$<C_COMPILER_ID:MSVC>:/O2>
        )
    endif()

    if(OpenMP_C_FOUND)
        target_link_libraries(${EXT_TARGET} PRIVATE OpenMP::OpenMP_C)
    endif()

    install(TARGETS ${EXT_TARGET} LIBRARY DESTINATION ${EXT_DESTINATION})  # <-- LIBRARY added
endmacro()