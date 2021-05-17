//
// Created by nimrod on 19-4-17.
//

#ifndef IP_CAMERA_CAMERA_READER_H
#define IP_CAMERA_CAMERA_READER_H

#include <opencv2/opencv.hpp>
//#include <boost/thread/mutex.hpp>
#include <mutex>
#include <thread>

#include <cuda.h>


class CameraReader
{
public:
    CameraReader(){}
    ~CameraReader(){
        std::cout<< "~CameraReader waiting stop" << std::endl;
        _working = false;
//        cuCtxDestroy(_cuContext);
        _work_thread->join();
        std::cout<< "~CameraReader done" << std::endl;
    }
    bool init(std::string camera_url);
    void run();
    void start();
    void stop(){_working = false;}
    void wait_done();
    bool is_work(){ return _working;}
    cv::Mat get_frame();
    void get_frame(cv::Mat &dst);
private:
    bool init_cuda();

    cv::Mat _frame;
    cv::cuda::GpuMat _d_frame;
    bool _working;
    std::mutex _mutex;
    std::thread *_work_thread;

    CUcontext _cuContext;
    CUdeviceptr _dpFrame;

    std::string _cam_url;
    std::unique_ptr<uint8_t[]> pImage;
    int _width;
    int _height;
};


#endif //IP_CAMERA_CAMERA_READER_H
