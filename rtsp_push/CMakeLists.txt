cmake_minimum_required(VERSION 3.5)
project(rtsp)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")
SET(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-9.2)
SET(OpenCV_DIR /home/ganyi/opencv-3.4.10/build-cuda9.2)




find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
#find_package(ONNXRUNTIME REQUIRED)
#find_package(Torch REQUIRED)
message("${OpenCV_DIR}")
message("${CUDA_DIR}")
#message("${ONNXRUNTIME_LIBRARIES}")


message("${CUDA_INCLUDE_DIR}")
message("${OpenCV_LIBRARIES}")
message("done")


LINK_DIRECTORIES(
#        /home/ganyi/opencv-3.4.10/build/lib
        /usr/local/cuda/lib64
        /usr/local/lib
        /home/ganyi/install/usr/local/lib
        /home/ganyi/vscodeproject/rtsp/ip_cam/build_cuda9
#        /usr/local/lib
        /home/ganyi/vscodeproject/rtsp/RtspServer/obj
        /usr/local/opt/ssl/lib
        /home/ganyi/onnxruntime/build/Linux/Release
)

INCLUDE_DIRECTORIES(
        ./
        ${CUDA_INCLUDE_DIR}
#        /home/ganyi/opencv-3.4.10/build/include
        /usr/local/cuda/include
        /home/ganyi/install/usr/local/include/onnxruntime
        /home/ganyi/install/usr/local/include/onnxruntime/core/session
        /usr/local/include
        /usr/include
        /home/ganyi/darknet/include
        /home/ganyi/vscodeproject/rtsp/RtspServer/src/3rdpart
        /home/ganyi/vscodeproject/rtsp/RtspServer/src
        /home/ganyi/vscodeproject/rtsp/ip_cam/include/ip_camera
        /home/ganyi/vscodeproject/rtsp/ip_cam/nvidia_sdk
        /home/ganyi/vscodeproject/rtsp/ip_cam/nvidia_sdk/NvCodec
)
#set(FS_SOURCES local_filesystem.h sync_api.h controller.h controller.cc  local_filesystem_posix.cc sync_api_posix.cc)
#add_library(slim_fs_lib ${FS_SOURCES})
#CUDA_add_library(nmslib SHARED
#        nms.cu
#        gpu_nms.h
#        )
#CUDA_add_executable(onnx_test )
#target_link_libraries(nmslib cuda)

FILE(GLOB_RECURSE MY_SOURCE /home/ganyi/vscodeproject/rtsp/RtspServer/src/*.cpp)
#ADD_EXECUTABLE(onnx_test test_onnx.cpp)
#target_link_libraries(onnx_test PRIVATE onnxruntime ${OpenCV_LIBRARIES} nmslib
#        )





#${OpenCV_LIBRARIES})
#target_link_libraries(onnx_test ${OpenCV_LIBRARIES})
#target_link_libraries(onnx_test ${OpenCV_LIBRARIES} onnxruntime pthread )
#target_link_libraries(onnx_test ${OpenCV_LIBRARIES})
#slim_fc_lib)


ADD_EXECUTABLE(rtsp_push rtsp_push_cv.cpp ${MY_SOURCE} )
#
TARGET_LINK_LIBRARIES(rtsp_push

        ${OpenCV_LIBRARIES}
        camera_reader
        frame_packer
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
