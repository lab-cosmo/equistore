cmake_minimum_required(VERSION 3.10)
project(equistore-tests)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_NO_SYSTEM_FROM_IMPORTED ON)

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'release' as none was specified.")
        set(CMAKE_BUILD_TYPE "release"
            CACHE STRING
            "Choose the type of build, options are: debug or release"
        FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug)
    endif()
endif()

set(EQUISTORE_SERIALIZATION ON)
add_subdirectory(../../ equistore)

add_subdirectory(external)

find_program(VALGRIND valgrind)
if (VALGRIND)
    message(STATUS "Running tests using valgrind")
    set(TEST_COMMAND
        "${VALGRIND}" "--tool=memcheck" "--dsymutil=yes" "--error-exitcode=125"
        "--leak-check=full" "--show-leak-kinds=definite,indirect,possible" "--track-origins=yes"
        "--gen-suppressions=all"
    )
else()
    set(TEST_COMMAND "")
endif()


if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "-Weverything")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-weak-vtables")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-float-equal")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-prototypes")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shadow-uncaptured-local")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-padded")
endif()


enable_testing()

file(GLOB ALL_TESTS *.cpp)
foreach(_file_ ${ALL_TESTS})
    get_filename_component(_name_ ${_file_} NAME_WE)
    add_executable(${_name_} ${_file_})
    target_link_libraries(${_name_} equistore catch)
    target_compile_definitions(${_name_} PRIVATE "-DDATA_NPZ=\"${CMAKE_CURRENT_SOURCE_DIR}/../data.npz\"")

    add_test(
        NAME ${_name_}
        COMMAND ${TEST_COMMAND} $<TARGET_FILE:${_name_}>
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endforeach()
