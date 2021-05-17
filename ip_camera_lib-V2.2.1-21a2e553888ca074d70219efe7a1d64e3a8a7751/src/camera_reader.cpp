//
// Created by nimrod on 19-4-17.
//

#include "ip_camera/camera_reader.h"
#include <iostream>

#include <unistd.h>
#include <cuda.h>
#include <iostream>
//#include "FramePresenterGL.h"
#include "NvCodec/NvDecoder/NvDecoder.h"
#include "Utils/NvCodecUtils.h"
#include "Utils/FFmpegDemuxer.h"
#include "Utils/ColorSpace.h"
//#include "../Common/AppDecUtils.h"
#include <opencv2/opencv.hpp>

//#include <boost/thread/thread.hpp>
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

static CUdeviceptr pTmpImage = 0;

enum OutputFormat
{
    native = 0, bgrp, rgbp, bgra, rgba, bgra64, rgba64
};

void GetImage(CUdeviceptr dpSrc, uint8_t *pDst, int nWidth, int nHeight)
{
    CUDA_MEMCPY2D m = {0};
    m.WidthInBytes = nWidth;
    m.Height = nHeight;
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = (CUdeviceptr) dpSrc;
    m.srcPitch = m.WidthInBytes;
    m.dstMemoryType = CU_MEMORYTYPE_HOST;
    m.dstDevice = (CUdeviceptr) (m.dstHost = pDst);
    m.dstPitch = m.WidthInBytes;
    cuMemcpy2D(&m);
}

void ConvertSemiplanarToPlanar(uint8_t *pHostFrame, int nWidth, int nHeight, int nBitDepth)
{
    if(nBitDepth == 8)
    {
        // nv12->iyuv
        YuvConverter<uint8_t> converter8(nWidth, nHeight);
        converter8.UVInterleavedToPlanar(pHostFrame);
    }
    else
    {
        // p016->yuv420p16
        YuvConverter<uint16_t> converter16(nWidth, nHeight);
        converter16.UVInterleavedToPlanar((uint16_t *) pHostFrame);
    }
}

void CameraReader::wait_done()
{
    _work_thread->join();
}

bool CameraReader::init_cuda()
{
    int iGpu = 0;
    try
    {
        ck(cuInit(0));
        int nGpu = 0;
        ck(cuDeviceGetCount(&nGpu));
        if(iGpu < 0 || iGpu >= nGpu)
        {
            std::ostringstream err;
            err << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
            throw std::invalid_argument(err.str());
        }
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, iGpu));
        char szDeviceName[80];
        ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
        std::cout << "GPU in use: " << szDeviceName << std::endl;
        ck(cuCtxCreate(&_cuContext, 0, cuDevice));
//        ck(cuCtxCreate(&_cuContext, CU_CTX_SCHED_BLOCKING_SYNC, cuDevice));

        std::cout << "Decode with NvDecoder." << std::endl;


    }
    catch(const std::exception &ex)
    {
        std::cout << ex.what();
        return false;
    }
    return true;
}


bool CameraReader::init(std::string camera_url)
{
    std::cout << camera_url.size() << std::endl;
    _cam_url = camera_url;

//    if(!init_cuda())
//    {
//        std::cout<< camera_url << " open error!" << std::endl;
//        _working = false;
//        return false;
//    }

    _working = true;
    return true;
}


void CameraReader::run()
{
    int iGpu = 0;
    bool bReturn = 1;
    CUdeviceptr pTmpImage = 0;
    OutputFormat eOutputFormat = bgra;

//    std::cout << "CameraReader::run debug: " << __LINE__ << std::endl;

    try
    {
        ck(cuInit(0));
        int nGpu = 0;
        ck(cuDeviceGetCount(&nGpu));
//        std::cout << "CameraReader::run debug: " << __LINE__ << std::endl;
        if(iGpu < 0 || iGpu >= nGpu)
        {
            std::ostringstream err;
            err << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]";
            throw std::invalid_argument(err.str());
        }
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, iGpu));
        char szDeviceName[80];
        ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
        LOG(INFO) << "GPU in use: " << szDeviceName;
//        CUcontext cuContext = NULL;
        ck(cuCtxCreate(&_cuContext, 0, cuDevice));

        FFmpegDemuxer demuxer(_cam_url.c_str());
        if(!demuxer.is_connected())
        {
            std::cerr << "CameraReader::run demuxer.init failed: "<< _cam_url << std::endl;
            ck(cuCtxDestroy(_cuContext));
//            delete _cuContext;
            _working = false;
            return;
        }

        NvDecoder dec(_cuContext, demuxer.GetWidth(), demuxer.GetHeight(), true,
                      FFmpeg2NvCodecId(demuxer.GetVideoCodec()));
        _width = demuxer.GetWidth();
        _height = demuxer.GetHeight();
        int anSize[] = {0, 3, 3, 4, 4, 8, 8};
        int nFrameSize = eOutputFormat == native ? demuxer.GetFrameSize() : _width * _height * anSize[eOutputFormat];
//        std::unique_ptr<uint8_t[]> pImage(new uint8_t[nFrameSize]);
        pImage = (std::unique_ptr<uint8_t[]>)(new uint8_t[nFrameSize]);

        int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
        uint8_t *pVideo = NULL;
        uint8_t **ppFrame;

        cuMemAlloc(&pTmpImage, _width * _height * anSize[eOutputFormat]);
        std::cout<< "cuMemAlloc(&pTmpImage: " << _width * _height * anSize[eOutputFormat] << std::endl;

        do
        {
//            std::cout << "CameraReader::run debug: " << __LINE__ << " "<<nFrameReturned << std::endl;
            if(!demuxer.Demux(&pVideo, &nVideoBytes))
            {
                std::cerr << "CameraReader::run demuxer.Demux failed" << std::endl;
                _working = false;
                break;
            }

//            std::cout << "CameraReader::run debug: " << __LINE__ << " "<<nFrameReturned << std::endl;
            if(!dec.Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned))
            {
                std::cerr << "CameraReader::run dec.Decode failed" << std::endl;
                _working = false;
                break;
            }

//            std::cout << "CameraReader::run debug: " << __LINE__ << " "<<nFrameReturned << std::endl;
            if(!nFrame && nFrameReturned)
                LOG(INFO) << dec.GetVideoInfo();
//            std::cout << "CameraReader::run debug: " << __LINE__ << " "<<nFrameReturned << std::endl;

            for(int i = 0; i < nFrameReturned; i++)
            {
//                std::cout << "CameraReader::run debug: " << __LINE__ << std::endl;
                if(dec.GetBitDepth() == 8)
                {
                    if(dec.GetOutputFormat() == cudaVideoSurfaceFormat_YUV444)
                        YUV444ToColor32<BGRA32>((uint8_t *) ppFrame[i], dec.GetWidth(), (uint8_t *) pTmpImage,
                                                4 * dec.GetWidth(), dec.GetWidth(), dec.GetHeight());
                    else
                        Nv12ToColor32<BGRA32>((uint8_t *) ppFrame[i], dec.GetWidth(), (uint8_t *) pTmpImage,
                                              4 * dec.GetWidth(), dec.GetWidth(), dec.GetHeight());
                    {
                        GetImage(pTmpImage, reinterpret_cast<uint8_t *>(pImage.get()), 4 * dec.GetWidth(), dec.GetHeight());
                    }
                    cv::cuda::GpuMat dframe, dgray;
                    dframe = cv::cuda::GpuMat(_height, _width, CV_8UC4, pTmpImage);
                    cv::cuda::cvtColor(dframe, dgray, cv::COLOR_BGRA2GRAY);
                }
            }

            {
                std::unique_lock<std::mutex> lock(_mutex);
                _frame = cv::Mat(_height, _width, CV_8UC4, pImage.get());
            }
            nFrame += nFrameReturned;
//            std::cout << "CameraReader::run " << _working << std::endl;
//            _working = false;
        } while(nVideoBytes && _working);

        if(pTmpImage)
        {
            std::cout << "cuMemFree(pTmpImage);" << std::endl;
            cuMemFree(pTmpImage);
            pTmpImage = 0;
        }
    }
    catch(const NVDECException &ex)
    {
        std::cerr << ex.what() << "not working"<< std::endl;
        _working = false;
        if(pTmpImage)
        {
            std::cout << "cuMemFree(pTmpImage);" << std::endl;
            cuMemFree(pTmpImage);
        }
    }
    catch(const std::exception &ex)
    {
        std::cerr << ex.what() << "not working"<< std::endl;
        _working = false;
        if(pTmpImage)
        {
            std::cout << "cuMemFree(pTmpImage);" << std::endl;
            cuMemFree(pTmpImage);
        }
    }

    if(pTmpImage)
    {
        std::cout << "cuMemFree(pTmpImage);" << std::endl;
        cuMemFree(pTmpImage);
        pTmpImage = 0;
    }
    ck(cuCtxDestroy(_cuContext));
}

void CameraReader::start()
{
    std::cout << "CameraReader::start" << std::endl;
    _work_thread = new std::thread(&CameraReader::run, this);
}


cv::Mat CameraReader::get_frame()
{
    std::unique_lock<std::mutex> lock(_mutex);
    if(!_working || _frame.empty())
    {
        return cv::Mat();
    }

    return _frame.clone();
}

void CameraReader::get_frame(cv::Mat &dst)
{
    std::unique_lock<std::mutex> lock(_mutex);
    if(!_working || !pImage.get())
    {
        return;
    }

//    _frame.copyTo(dst);
    dst = cv::Mat(_height, _width, CV_8UC4, pImage.get());

}