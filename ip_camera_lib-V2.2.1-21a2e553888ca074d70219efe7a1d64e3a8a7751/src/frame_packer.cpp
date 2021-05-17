//
// Created by nimrod on 19-11-21.
//

#include "ip_camera/frame_packer.h"
#include <Utils/NvCodecUtils.h>
#include <NvCodec/NvEncoder/NvEncoder.h>
#include <NvCodec/NvEncoder/NvEncoderCuda.h>
#include "Utils/NvEncoderCLIOptions.h"

simplelogger::Logger *logger;

void FramePacker::init(int iGpu)
{
    int nGpu = 0;

    CUdevice cuDevice = 0;

    ck(cuInit(0));
    ck(cuDeviceGetCount(&nGpu));

    if (iGpu < 0 || iGpu >= nGpu)
    {
        std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
        return;
    }
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    std::cout << "GPU in use: " << szDeviceName << std::endl;

    ck(cuDeviceGet(&cuDevice, iGpu));
    ck(cuCtxCreate(&_cuContext, 0, cuDevice));
}

void FramePacker::init_encoder()
{
    if(_enc != NULL)
    {
        _enc->DestroyEncoder();
        delete _enc;
        _enc = NULL;
    }

    NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
    NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
    initializeParams.encodeConfig = &encodeConfig;

    NvEncoderInitParam initParam;

    _enc = new NvEncoderCuda(_cuContext, _width, _height, _eFormat);
    _enc->CreateDefaultEncoderParams(&initializeParams, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_DEFAULT_GUID);
    initParam.SetInitParams(&initializeParams, _eFormat);

    _enc->CreateEncoder(&initializeParams);
    _initialized = true;
}

void FramePacker::set_frame_size(int width, int height)
{
    _width = width;
    _height = height;
}

std::vector<std::vector<uint8_t>> FramePacker::end_encode()
{
    std::vector<std::vector<uint8_t>> vPacket;
    _enc->EndEncode(vPacket);
    std::cout << "enc.EndEncode(vPacket)" << std::endl;
    return vPacket;
}

void FramePacker::get_sequence_params(std::vector<uint8_t> &buffer)
{
    if(_enc)
    {
        _enc->GetSequenceParams(buffer);
    }
}

std::vector<std::vector<uint8_t>> FramePacker::pack(cv::Mat &frame)
{
    std::vector<std::vector<uint8_t>> vPacket;
    int nFrame = 0;

    if (!frame.empty())
    {
        cv::Mat yuv_frame;
        cv::cvtColor(frame, yuv_frame, cv::COLOR_BGR2YUV_IYUV);
        uint8_t* p_frame = yuv_frame.data;

        const NvEncInputFrame* encoderInputFrame = _enc->GetNextInputFrame();
        NvEncoderCuda::CopyToDeviceFrame(_cuContext, p_frame, 0, (CUdeviceptr)encoderInputFrame->inputPtr,
                                         (int)encoderInputFrame->pitch,
                                         _enc->GetEncodeWidth(),
                                         _enc->GetEncodeHeight(),
                                         CU_MEMORYTYPE_HOST,
                                         encoderInputFrame->bufferFormat,
                                         encoderInputFrame->chromaOffsets,
                                         encoderInputFrame->numChromaPlanes);

        NV_ENC_PIC_PARAMS picParams = {};
//        picParams.pictureType = NV_ENC_PIC_TYPE_IDR;

        _enc->EncodeFrame(vPacket, &picParams);
//        std::cout << "enc.EncodeFrame(vPacket)" << std::endl;
    }

    nFrame += (int)vPacket.size();
//    std::cout << "vPacket.size: " << (int)vPacket.size() << std::endl;

    return vPacket;
}

FramePacker::~FramePacker()
{
    if(_enc != NULL)
    {
        _enc->DestroyEncoder();
        delete _enc;
        _enc = NULL;
    }

    if(_cuContext)
    {
        ck(cuCtxDestroy(_cuContext));
    }
}

