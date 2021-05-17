cmake_minimum_required(VERSION 3.5)
project(ip_camera_lib)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_SUPPRESS_REGENERATION true)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-9.2)
set(OpenCV_DIR /home/ganyi/opencv-3.4.10/build-cuda9.2)
#set(OpenCV_DIR /usr/local/opencv340)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
#find_package(Boost COMPONENTS thread system  REQUIRED)


link_directories(
        /usr/local/cuda/lib64/
#		/home/ganyi/vscodeproject/rtsp/ip_cam/Video_Codec_SDK_9.0.20/Lib/linux/stubs/x86_64
#		/usr/local/lib
#		/usr/lib
		/usr/local/opt/openssl/lib
		/usr/local/lib
		/home/ganyi/darknet
)

include_directories(
		/home/ganyi/vscodeproject/rtsp/ip_cam/include/
		/home/ganyi/vscodeproject/rtsp/ip_cam/include/ip_camera/
		/home/ganyi/vscodeproject/rtsp/ip_cam/nvidia_sdk
		/home/ganyi/vscodeproject/rtsp/ip_cam/nvidia_sdk/NvCodec
		/home/ganyi/vscodeproject/rtsp/ip_cam/Video_Codec_SDK_9.0.20
		/home/ganyi/vscodeproject/rtsp/ip_cam/Video_Codec_SDK_9.0.20/include/
		/usr/local/cuda/include/
		/usr/local/include/
		/usr/include/
        /home/ganyi/darknet/include
		/home/ganyi/darknet/src
		/home/ganyi/vscodeproject/uWebSockets-0.14.6/src
		)

CUDA_add_library(camera_reader SHARED
        src/camera_reader.cpp
        nvidia_sdk/NvCodec/NvDecoder/NvDecoder.cpp
        nvidia_sdk/Utils/ColorSpace.cu
)

#include_directories()
add_library(camera_writer SHARED
		src/camera_writer.cpp)
#
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
        #		darknet
		uWS
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
		-L/usr/local/lib
		avformat
		avutil
		avcodec
		darknet
		jsoncpp
        uWS
        ssl
        crypto
        z
        uv
        )


#
#add_executable(test_darknet src/test_darknet.cpp)
#target_link_libraries(test_darknet
#		${OpenCV_LIBRARIES}
#		-L/usr/local/lib/
#		pthread
#		cuda
#		darknet
#		jsoncpp
#		uWS
#		ssl
#		crypto
#		z
#		uv
#		)


add_executable(test_websocket src/test_websocket.cpp)

target_link_libraries(test_websocket
		-L/usr/local/lib/
		pthread
		uWS
		ssl
		crypto
		z
		uv)

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

#add_executable(test_cv2h264
#		src/test_cv2h264.cpp
#		nvidia_sdk/NvCodec/NvEncoder/NvEncoder.cpp
#		nvidia_sdk/NvCodec/NvEncoder/NvEncoderCuda.cpp
#		)

#target_link_libraries(test_cv2h264
#		${OpenCV_LIBRARIES}
#		pthread
#		nvcuvid
#		cuda
#		-L/usr/local/lib/
#		avformat
#		avutil
#		avcodec
#        dl
#		)

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
#        camera_reader
#        ${OpenCV_LIBRARIES}
#        pthread
#)

install(TARGETS
#    	ip_cam_reader
		frame_packer
        camera_reader
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY
        include/ip_camera
		DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
