//
// Created by nimrod on 19-4-17.
//

#include "ip_camera/camera_handler.h"
#include "ip_camera/camera_reader.h"
#include <unistd.h>

void CameraHandler::init(std::string camera_url)
{
    if(! CameraReader::get_instance()->init(camera_url))
    {
        _working = false;
    }
    _working = true;
}

void CameraHandler::run()
{
    CameraReader::get_instance()->start();
    while(_working)
    {
        cv::Mat frame = CameraReader::get_instance()->get_frame();
        std::cout<< frame.rows << " "<< frame.cols << std::endl;
        if(frame.rows < 10 or frame.cols < 10)
        {
            usleep(30);
            continue;
        }
        cv::imshow("test", frame);
//        usleep(30);
        cv::waitKey(30);
    }
}

void CameraHandler::start()
{

}
