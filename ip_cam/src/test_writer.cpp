//
// Created by nimrod on 19-9-5.
//


#include "ip_camera/camera_writer.h"
#include <opencv2/opencv.hpp>

int main()
{
    cv::VideoCapture cap(0);
    CameraWriter writer;

    writer.set_server("");

    while(cv::waitKey(30) != 27)
    {
        cv::Mat frame;
        cap >> frame;

        cv::imshow("frame", frame);

        writer.send(frame);
    }

    return 0;
}