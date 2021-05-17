//
// Created by nimrod on 19-4-17.
//

#include "ip_camera/camera_reader.h"
//static unsigned char _buffer[1024 * 1024];
#include "pybridge.h"

#include <iostream>
#include <stdio.h>


bool init(char *url)
{
    return CameraReader::get_instance()->init(url);
}

void start()
{
    CameraReader::get_instance()->start();
}

void stop()
{
    CameraReader::get_instance()->stop();
}

int get_frame_buf(unsigned char *pbuf)
{
     std::vector<int> param= std::vector<int>(2);
     param[0] = cv::IMWRITE_JPEG_QUALITY;
     param[1] = 90;//default(95) 0-100
    cv::Mat frame = CameraReader::get_instance()->get_frame();
    std::vector<uchar> buffer;
    if(frame.rows > 10 and frame.cols > 10)
        cv::imencode(".jpg", frame, buffer, param);
    memcpy(pbuf, &buffer[0], buffer.size());

    return buffer.size();
}