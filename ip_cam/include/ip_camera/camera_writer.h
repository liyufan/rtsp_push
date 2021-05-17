//
// Created by nimrod on 19-9-5.
//

#ifndef IP_CAMERA_LIB_CAMERA_WRITER_H
#define IP_CAMERA_LIB_CAMERA_WRITER_H

#include <iostream>
#include <opencv2/opencv.hpp>
#ifdef __cplusplus
extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/avutil.h"
#include "libavutil/time.h"
}
#endif

typedef struct ffmpeg_enc_param_struct
{
    int bitrate; // Bitrate in bytes
    int width;
    int height;
    int fps;
    int gop_size;
    int max_b_frames;
    int pix_fmt;
    int min_qp;
    int max_qp;
}ffmpeg_enc_param_t;

class CameraWriter
{
public:
    CameraWriter(){}

    void set_server(std::string url);
    void send(const cv::Mat &frame);

    int init(const std::string& url, ffmpeg_enc_param_t *pEncParam, AVFormatContext *ifmt_ctx);
    int CopyInStreamContext(AVFormatContext *ifmt_ctx);
    int CreateEncoder(ffmpeg_enc_param_t *pEncParam);
private:
    std::string _server_url;

    //Encoder Context
    AVCodecContext *m_h264Enc = NULL;
    //output Context
    AVOutputFormat *ofmt = NULL;
    AVFormatContext *ofmt_ctx = NULL;
    AVCodec *codec = NULL;
    AVStream *out_stream[2];
};


#endif //IP_CAMERA_LIB_CAMERA_WRITER_H
