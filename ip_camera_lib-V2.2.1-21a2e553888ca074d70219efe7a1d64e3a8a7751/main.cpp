#include <iostream>

#include "ip_camera/camera_reader.h"
#include <unistd.h>

CameraReader cam_reader;

void detect_thread()
{
    while(cam_reader.is_work())
    {
        usleep(400 * 1000);
//        sleep(1);
        cv::Mat frame;
//        cv::cuda::GpuMat dframe;
//        std::cout << "get_frame waiting..." << std::endl;
//        cam_reader.get_frame(frame);
//        frame = cam_reader.get_frame();
//        dframe.upload(frame);
//        std::cout << "getframe" << frame.rows<< " " << frame.cols << " work:"<< cam_reader.is_work() << std::endl;
        if(frame.empty())
        {
//            std::cout << "get_frame empty" << std::endl;
            continue;
        }
//        cv::imwrite("test.jpg", frame);
//        cv::imshow("ip_camera_reader", frame);
//        cv::Mat dst;
//        cv::resize(frame, dst, cv::Size(800, 600));
//        cv::waitKey(30);
    }
}

int main()
{
    std::cout << "ip camera" << std::endl;
//    std::string camera_url = "rtsp://127.0.0.1:8554/live";
    std::string camera_url = "rtsp://admin:lingzhi123321@192.168.1.74";
//    std::string camera_url = "rtsp://admin:rcir219219@192.168.1.209";
//    std::string camera_url = "rtsp://admin:lingzhi123321@192.168.1.210/h264/ch34/main/av_stream/?user_name=admin?password=lingzhi123321?linkmode=tcp";

    cam_reader.init(camera_url);
    cam_reader.start();
    std::thread detect(detect_thread);
//    cam_reader.start();
    detect.join();
//    detect_thread();
//    cam_reader.wait_done();

    return 0;
}