cmake_minimum_required(VERSION 3.16)

project(dprive_ml)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenFHE REQUIRED)

set( CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS} )

include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)
link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})

if (BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries(${OpenFHE_STATIC_LIBRARIES})
else ()
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    link_libraries(${OpenFHE_SHARED_LIBRARIES})
endif ()

add_executable(lr_nag lr_nag.cpp enc_matrix.cpp enc_matrix.h data_io.cpp data_io.h lr_types.h pt_matrix.cpp pt_matrix.h utils.cpp utils.h lr_train_funcs.cpp lr_train_funcs.h parameters.h)
add_executable(cheb_analysis cheb_analysis.cpp enc_matrix.cpp enc_matrix.h data_io.cpp data_io.h lr_types.h pt_matrix.cpp pt_matrix.h utils.cpp utils.h lr_train_funcs.cpp lr_train_funcs.h)

# ADD src
add_subdirectory(train_data)
