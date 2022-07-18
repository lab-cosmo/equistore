function(unpack_library _name_)
    set(_archive_ ${CMAKE_CURRENT_SOURCE_DIR}/${_name_}.tar.gz)
    file(SHA256 ${_archive_} _shasum_)
    if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${_name_}/${_shasum_}")
        message(STATUS "Unpacking ${_name_} sources")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${_name_}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xf ${_archive_}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E touch ${_name_}/${_shasum_}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif()

    if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${_name_}/.git")
        # .git directories can get huge and increase the size of this repository
        message(FATAL_ERROR "${_name_} archive contains a .git directory, please remove it")
    endif()

    # make sure cmake re-runs whenever the file changes
    get_directory_property(_previous_list_ CMAKE_CONFIGURE_DEPENDS)
    set_directory_properties(PROPERTIES CMAKE_CONFIGURE_DEPENDS "${_previous_list_};${_name_}.tar.gz")
endfunction()



add_library(catch STATIC catch/catch.cpp)
target_include_directories(catch PUBLIC catch)
