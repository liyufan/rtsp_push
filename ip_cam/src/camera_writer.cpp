//
// Created by nimrod on 19-9-5.
//

#include "ip_camera/camera_writer.h"

int CameraWriter::CopyInStreamContext(AVFormatContext *ifmt_ctx)
{
    int ret = 0;
    for(int i = 0; i < ifmt_ctx->nb_streams; i++)
    {
        //Create output AVStream according to input AVStream
        AVStream *in_stream = ifmt_ctx->streams[i];
        out_stream[i] = avformat_new_stream(ofmt_ctx, in_stream->codec->codec);
        if(!out_stream)
        {
            printf("Failed allocating output stream\n");
            return AVERROR_UNKNOWN;
        }

        //Copy the settings of AVCodecContext
        ret = avcodec_copy_context(out_stream[i]->codec, in_stream->codec);
        if(ret < 0)
        {
            printf("Failed to copy context from input to output stream codec context\n");
            return ret;
        }
        out_stream[i]->codec->codec_tag = 0;
        if(ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
            out_stream[i]->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    return 0;
}

int CameraWriter::init(const std::string& url, ffmpeg_enc_param_t *pEncParam, AVFormatContext *ifmt_ctx)
{
    int ret = 0;
    // Init output context
    avformat_alloc_output_context2(&ofmt_ctx, NULL, "rtsp", url.c_str()); //RTSP

    if(!ofmt_ctx)
    {
        printf("Could not create output context\n");
        return AVERROR_UNKNOWN;
    }

    ofmt = ofmt_ctx->oformat;
    if(ifmt_ctx != nullptr)
    {
        CopyInStreamContext(ifmt_ctx);
    }
    else
    {
        if(pEncParam)
        {
            ret = CreateEncoder(pEncParam);
            if(ret < 0)
            {
                printf("Create H264 Encoder err=%d\n", ret);
                return ret;
            }
        }
    }

    //out_stream->codec->video
    //
    av_dump_format(ofmt_ctx, 0, url.c_str(), 1);
}

int CameraWriter::CreateEncoder(ffmpeg_enc_param_t *pEncParam)
{
    int ret = 0;
    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    // Init Encoder if it needed
    m_h264Enc = avcodec_alloc_context3(codec);
    assert(m_h264Enc != NULL);

    /* put sample parameters */
    m_h264Enc->bit_rate = pEncParam->bitrate;
    /* resolution must be a multiple of two */
    m_h264Enc->width = pEncParam->width;
    m_h264Enc->height = pEncParam->height;

    /* frames per second */
    m_h264Enc->time_base.den = pEncParam->fps;
    m_h264Enc->time_base.num = 1;
    m_h264Enc->framerate.num = pEncParam->fps;
    m_h264Enc->framerate.den = 1;

    m_h264Enc->gop_size = pEncParam->gop_size; /* emit one intra frame every ten frames */
    m_h264Enc->max_b_frames = pEncParam->max_b_frames;
    m_h264Enc->pix_fmt = (enum AVPixelFormat) pEncParam->pix_fmt;
    m_h264Enc->qmax = pEncParam->max_qp;
    m_h264Enc->qmin = pEncParam->min_qp;
    m_h264Enc->delay = 0;

    AVDictionary *options;
    av_dict_set(&options, "preset", "medium", 0);
    av_dict_set(&options, "tune", "zerolatency", 0);
    av_dict_set(&options, "profile", "baseline", 0);

    ret = avcodec_open2(m_h264Enc, codec, &options);
    if (ret < 0) {

        printf("avcodec_open2 failed!");
        return -1;
    }

    av_dict_free(&options);
    return 0;
}

void CameraWriter::set_server(std::string url)
{

}

void CameraWriter::send(const cv::Mat &frame)
{

}
