//
// Created by nimrod on 19-11-21.
//

#ifndef IP_CAMERA_LIB_FRAME_PACKER_H
#define IP_CAMERA_LIB_FRAME_PACKER_H

#include <cuda.h>
#include <nvEncodeAPI.h>
#include <opencv2/opencv.hpp>

class NvEncoderCuda;

class FramePacker
{
public:
    FramePacker():
    _cuContext(NULL),
    _enc(NULL),
    _eFormat(NV_ENC_BUFFER_FORMAT_IYUV),
    _initialized(false)
    {};

    ~FramePacker();
    void init(int iGpu=0);
    void set_frame_size(int width, int height);
    void init_encoder();

    bool is_initialized() {return _initialized;}

    std::vector<std::vector<uint8_t>> pack(cv::Mat &frame);
    std::vector<std::vector<uint8_t>> end_encode();
    void get_sequence_params(std::vector<uint8_t> &buffer);
private:
    CUcontext _cuContext;

    NV_ENC_BUFFER_FORMAT _eFormat;

    int _width;
    int _height;

    bool _initialized;
    NvEncoderCuda* _enc;
};


#endif //IP_CAMERA_LIB_FRAME_PACKER_H
