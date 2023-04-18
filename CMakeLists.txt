cmake_minimum_required(VERSION 3.16)
project(dprive_ml)
set(CMAKE_FIND_DEBUG_MODE TRUE)
find_package(OpenFHE REQUIRED)

set(CMAKE_CXX_STANDARD 14)
if (CMAKE_BUILD_TYPE)
    set(RELEASE_TYPES
            Debug
            Release
            )
    list(FIND RELEASE_TYPES ${CMAKE_BUILD_TYPE} INDEX_FOUND)
    if (${INDEX_FOUND} EQUAL -1)
        message(
                FATAL_ERROR
                "CMAKE_BUILD_TYPE must be one of Debug or Release."
        )
    endif ()
else ()
    # if no build type is chosen, default to Release mode
    set(CMAKE_BUILD_TYPE Release CACHE STRING
            "Choose the type of build, options are: None, Debug, Release."
            FORCE)
endif ()
if (NOT MATHBACKEND)
    set(MATHBACKEND 4)
endif ()
set(CMAKE_CXX_FLAGS "-DMATHBACKEND=${MATHBACKEND}")

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})

if (BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries(${OpenFHE_STATIC_LIBRARIES})
else ()
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    link_libraries(${OpenFHE_SHARED_LIBRARIES})
endif ()

message("CMake Flags: ${CMAKE_CXX_FLAGS}")
if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
    message("CMake Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
else()
    message("CMake Debug flags: ${CMAKE_CXX_FLAGS_DEBUG}")
endif()

add_executable(lr_nag lr_nag.cpp enc_matrix.cpp enc_matrix.h data_io.cpp data_io.h lr_types.h pt_matrix.cpp pt_matrix.h utils.cpp utils.h lr_train_funcs.cpp lr_train_funcs.h parameters.h)
add_executable(cheb_analysis cheb_analysis.cpp enc_matrix.cpp enc_matrix.h data_io.cpp data_io.h lr_types.h pt_matrix.cpp pt_matrix.h utils.cpp utils.h lr_train_funcs.cpp lr_train_funcs.h)

# ADD src
add_subdirectory(train_data)
