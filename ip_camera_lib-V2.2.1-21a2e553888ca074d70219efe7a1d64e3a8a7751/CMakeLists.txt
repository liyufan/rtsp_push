cmake_minimum_required(VERSION 3.5)
project(ip_camera_lib)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_SUPPRESS_REGENERATION true)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")

find_package(OpenCV REQUIRED)
#find_package(Boost COMPONENTS thread system  REQUIRED)

link_directories(
        /usr/local/cuda/lib64
)

include_directories(include
		nvidia_sdk
		nvidia_sdk/NvCodec
		/usr/local/cuda/include
		)

CUDA_add_library(camera_reader SHARED
        src/camera_reader.cpp
        nvidia_sdk/NvCodec/NvDecoder/NvDecoder.cpp
        nvidia_sdk/Utils/ColorSpace.cu
)

add_library(camera_writer SHARED
		src/camera_writer.cpp )

add_library(frame_packer SHARED
		src/frame_packer.cpp
		nvidia_sdk/NvCodec/NvEncoder/NvEncoder.cpp
		nvidia_sdk/NvCodec/NvEncoder/NvEncoderCuda.cpp )

target_link_libraries(camera_reader
    ${OpenCV_LIBRARIES}
		pthread
		nvcuvid
		cuda
		-L/usr/local/lib/
		avformat
		avutil
		avcodec
)

target_link_libraries(camera_writer
		${OpenCV_LIBRARIES}
		pthread
		nvcuvid
		cuda
		-L/usr/local/lib/
		avformat
		avutil
		avcodec
		)

target_link_libraries(frame_packer
		${OpenCV_LIBRARIES}
		pthread
		nvcuvid
		cuda
		-L/usr/local/lib/
		avformat
		avutil
		avcodec)

add_executable(ip_camera main.cpp )

target_link_libraries(ip_camera
    camera_reader
    pthread
    nvcuvid
    cuda
    -L/usr/local/lib/
    avformat
    avutil
    avcodec
)

add_executable(test_writer src/test_writer.cpp )
target_link_libraries(test_writer
		camera_writer
		pthread
		nvcuvid
		cuda
		-L/usr/local/lib/
		avformat
		avutil
		avcodec
		)

add_executable(test_encoder src/test_encoder.cpp)
target_link_libraries(test_encoder
        -L/usr/local/lib/
        avcodec avformat avutil swscale avfilter avdevice )

add_executable(test_multi_client src/test_multi_client.cpp)
target_link_libraries(test_multi_client
        -L/usr/local/lib/
        avcodec avformat avutil swscale avfilter avdevice )

add_executable(test_cv2h264
		src/test_cv2h264.cpp
		nvidia_sdk/NvCodec/NvEncoder/NvEncoder.cpp
		nvidia_sdk/NvCodec/NvEncoder/NvEncoderCuda.cpp
		)

target_link_libraries(test_cv2h264
		${OpenCV_LIBRARIES}
		pthread
		nvcuvid
		cuda
		-L/usr/local/lib/
		avformat
		avutil
		avcodec
        dl
		)

add_executable(test_frame_packer test/test_frame_packer.cpp)
target_link_libraries(test_frame_packer
		frame_packer
		${OpenCV_LIBRARIES}
		pthread
		nvcuvid
		cuda
		-L/usr/local/lib/
		avformat
		avutil
		avcodec
		dl
		)

#add_library(ip_cam_reader SHARED
#    src/pybridge.cpp
#    src/camera_reader.cpp
#)
#
#
#target_link_libraries(ip_cam_reader
##        camera_reader
#        ${OpenCV_LIBRARIES}
#        pthread
#)

install(TARGETS
#    ip_cam_reader
        camera_reader
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY
        include/ip_camera
		DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
