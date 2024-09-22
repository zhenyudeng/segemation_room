cmake_minimum_required(VERSION 3.0.0)
#交叉编译opencv3-3.3.0
project(room_seg)
set(BUILD_IN_RK3308  OFF)
include(cmake/gcc-flags.cmake)
if(BUILD_IN_RK3308)
     set(CMAKE_BUILD_TYPE Release)

     include(cmake/rk_aarch64-linux-gnu.cmake)
     set(OpenCV_LIBRARY_DIRS ./thridparty/opencv/lib)
     set(OpenCV_INCLUDE_DIRS ./thridparty/opencv/include)

     set(OpenCV_LIBRARIES
         libopencv_core.so
         libopencv_highgui.so
         libopencv_dnn.so
         libopencv_imgcodecs.so
         libopencv_imgproc.so
         libopencv_flann.so
         libopencv_videoio.so
         libz.so
     )
     link_directories(${OpenCV_LIBRARY_DIRS})

     include_directories( ${OpenCV_INCLUDE_DIRS} )
else()
     set(CMAKE_BUILD_TYPE Debug)
     add_definitions(-std=c++11 -g)
     find_package(OpenCV REQUIRED)
     include_directories(${OpenCV_INCLUDE_DIRS})
endif()

include_directories(
    ./
    room_segmentation)


MESSAGE(STATUS "OpenCV_INCLUDE_DIRS=" ${OpenCV_INCLUDE_DIRS})
MESSAGE(STATUS "OpenCV_LIBRARIES=" ${OpenCV_LIBRARIES})


add_executable(${PROJECT_NAME}
    ipa_room_segmentation/abstract_voronoi_segmentation
    ipa_room_segmentation/contains.cpp
    ipa_room_segmentation/voronoi_segmentation.cpp
    ipa_room_segmentation/wavefront_region_growing.cpp
    ipa_room_segmentation/room_class.cpp
    ipa_room_segmentation/meanshift2d.cpp
    main.cc )
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

