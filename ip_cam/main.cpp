#include <iostream>

#include "ip_camera/camera_reader.h"
#include <unistd.h>
#include <darknet.h>
#include <uWS.h>
#include <json/json.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <mutex>
#include <memory>



CameraReader cam_reader;



class Object {
public:
    float x1, y1, x2, y2;
    int cls;
    Object(): x1(0), y1(0), x2(0), y2(0), cls(0) {

    }
    Object(box bbox, int c): cls(c){
        x1 = bbox.x - bbox.w/2.;
        x2 = bbox.x + bbox.w/2.;
        y1 = bbox.y - bbox.h/2.;
        y2 = bbox.y + bbox.h/2.;

    }
    Object(float xmin, int ymin, int xmax, int ymax, int cls): x1(xmin), y1(ymin), x2(xmax), y2(ymax) {

    }

};
static std::mutex globalMutex;
std::vector<Object> dets;

void draw_yolo_detections(cv::Mat& frame, detection *dets, int total, int classes, int w, int h, std::vector<Object>& valid_box) {
    int i, j;
    int draw_bbox=0;
    using namespace cv;
    using namespace std;
    for (int i=0; i<total; ++i) {
        int xmin = (dets[i].bbox.x - dets[i].bbox.w/2.) * w;
        int xmax = (dets[i].bbox.x + dets[i].bbox.w/2.) * w;
        int ymin = (dets[i].bbox.y - dets[i].bbox.h/2.) * h;
        int ymax = (dets[i].bbox.y + dets[i].bbox.h/2.) * h;
        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for (j=0; j<classes; ++j) {
            if (dets[i].prob[j]) {
                ++draw_bbox;
                valid_box.emplace_back(Object(dets[i].bbox, j));
                rectangle(frame, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 255), 3, 3, 0);
                putText(frame, to_string(j), Point(xmin, ymin-5), FONT_HERSHEY_COMPLEX, 2, Scalar(0,255,255), 1, 1, 0);
//                cout<<"bbox coordinates are: "<<dets[i].bbox.x<<" "<<dets[i].bbox.y<<" "<<dets[i].bbox.w<<" "<<dets[i].bbox.h<<endl;
//                cout<<"the prob is: "<<dets[i].prob[j]<<"the class is: "<<j<<endl;

            }
        }

    }
//    cout<<draw_bbox<<" bboxes are draw on the image"<<endl;
}

void yolo_predict(network* net, cv::Mat& frame, std::vector<Object>& valid_box) {
    using namespace cv;
    using namespace std;


    int classes = 80;
    float thresh = 0.01;
    float nms = .4;
    float iou_thresh = 0.5;
    int w_o = frame.cols;
    int h_o = frame.rows;
    int c_o = frame.channels();

    int w = net->w;
    int h = net->h;
    int c = net->c;

    cvtColor(frame,frame,CV_BGR2RGB);
    unique_ptr<float []> ddata(new float[h_o*w_o*c_o*sizeof(float)]);
    for (int i=0; i<h_o; ++i) {
        for (int j=0; j<w_o; ++j) {
            for (int k=0; k<c_o; ++k) {
                ddata[k*h_o*w_o + i * w_o + j] = (float)((frame.at<Vec3b>(i, j)[k] + 10) / 255.);

            }
        }
    }
    image boxed;
    boxed.w = w_o;
    boxed.h = h_o;
    boxed.c = c_o;
    boxed.data = ddata.get();
    image sized = letterbox_image(boxed, w, h);

    network_predict(net, sized.data);
    int nboxes = 0;
    detection *dets = get_network_boxes(net, 1, 1, 0.5, 0.5, 0, 1, &nboxes);
    cout<<nboxes<<" bbox are generated"<<endl;
    if (nms) do_nms_sort(dets, nboxes, classes, iou_thresh);
    draw_yolo_detections(frame, dets, nboxes, classes, w_o, h_o, valid_box);
    free_detections(dets, nboxes);


}

std::time_t getTimeStamp()
{
    std::chrono::time_point<std::chrono::system_clock,std::chrono::milliseconds> tp = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());//获取当前时间点
    std::time_t timestamp =  tp.time_since_epoch().count();
    return timestamp;
}

void detect_thread()
{
    char cfg[] = "/home/ganyi/darknet/cfg/yolov3.cfg";
    char weight[] = "/home/ganyi/darknet/yolov3.weights";
    network* net = load_network(cfg, weight, 0);
    layer l = net->layers[net->n-1];
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));
    std::vector<Object> detections;
    while(cam_reader.is_work())
    {
        usleep(400 * 1000);
//        sleep(1);
        cv::Mat frame;
//        cv::cuda::GpuMat
        cv::cuda::GpuMat dframe;
//        std::cout << "get_frame waiting..." << std::endl;
        cam_reader.get_frame(frame);
//        std::cout<<frame.cols<<" "<<frame.rows<<std::endl;
//        frame = cam_reader.get_frame();
//        cv::imshow("cur_fame", frame);
        cv::waitKey(1);
        dframe.upload(frame);
//        std::cout << "getframe" << frame.rows<< " " << frame.cols << " work:"<< cam_reader.is_work() << std::endl;
        if(frame.empty())
        {
//            std::cout << "get_frame empty" << std::endl;
            continue;
        }else {
            detections.clear();
//            std::cout << "getframe" << frame.rows<< " " << frame.cols << " work:"<< cam_reader.is_work() << std::endl;
            yolo_predict(net, frame, detections);
            {
                std::unique_lock<std::mutex> tmp(globalMutex);
                dets = detections;
            }

        }
//        cv::imwrite("test.jpg", frame);
//        cv::imshow("ip_camera_reader", frame);
//        cv::Mat dst;
//        cv::resize(frame, dst, cv::Size(800, 600));
//        cv::waitKey(30);
    }
}

void WebSocket_Server() {
    using namespace uWS;
    Hub h;
    Json::Value parser;
    Json::CharReaderBuilder builder;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    h.onConnection([](WebSocket<SERVER> *ws, HttpRequest req) {
        std::cout<<"Successfully build the connection with client"<<std::endl;
    });

    h.onMessage([&parser, &reader](WebSocket<SERVER> *ws, char* message, size_t length, OpCode opCode) {
        std::cout<<"message received"<<std::endl;
        JSONCPP_STRING err;
        Json::Value message_send;
        if (!reader->parse(message, message+length, &parser, &err)) {
            std::cout<<"can not successfully parse the message, and the message is: ";
            std::cout<<message<<std::endl;
        }else {
            std::string operation = parser["op"].asString();
            std::string client_message = parser["message"].asString();
            std::cout<<"client message is: "<<client_message<<std::endl;
            if (operation == "request") {
                {

//                    std::unique_lock<std::mutex>(globalMutex);
                    std::unique_lock<std::mutex> l(globalMutex);
                    time_t timeStamp= getTimeStamp();
                    message_send["timestamp"] = std::to_string(timeStamp).c_str();
                    message_send["url"] = "192.168.1.106";
                    message_send["frameId"] = "unknow";
                    Json::Value emptyObj;
                    emptyObj["cls"] = 0;
                    emptyObj["x1"] = 0.3;
                    emptyObj["x2"] = 0.7;
                    emptyObj["y1"] = 0.5;
                    emptyObj["y2"] = 0.8;
                    message_send["bboxes"].append(emptyObj);
                    for (Object& elem: dets) {
                        Json::Value jsonObj;
                        jsonObj["cls"] = elem.cls;
                        jsonObj["x1"] = elem.x1;
                        jsonObj["x2"] = elem.x2;
                        jsonObj["y1"] = elem.y1;
                        jsonObj["y2"] = elem.y2;
                        message_send["bboxes"].append(jsonObj);
                    }
                    std::string SerializedMessage = message_send.toStyledString();
                    size_t Message_length = SerializedMessage.length();
                    ws->send(SerializedMessage.c_str(), Message_length, opCode);

                }
            }
        }

    });

    h.onDisconnection([](WebSocket<SERVER> *ws, int code, char* message, size_t length) {
        std::cout<<"Connection is broken, disconnected code: "<<code<<std::endl;
        if (length > 0) {
            std::cout<<"last message is: "<<message<<std::endl;
        }
    });

    h.onError([](void* err) {
        std::cout<<"Error happend in connection"<<std::endl;
    });

    if (h.listen(3000)) {
        std::cout<<"Successfully listen to port 3000"<<std::endl;
        h.run();
    };
}

int main()
{
    std::cout << "ip camera" << std::endl;
//    std::string camera_url = "rtsp://127.0.0.1:8554/live";
//    std::string camera_url = "rtsp://admin:lingzhi123321@192.168.1.74";
    std::string camera_url = "rtsp://admin:rcir219219@192.168.1.209";
//    cv::VideoCapture cap = cv::VideoCapture(camera_url);
//    while(cap.isOpened()) {
//        cv::Mat frame;
//        auto time_start = std::chrono::steady_clock::now();
//        cap>>frame;
////        auto time_after = std::chrono::steady_clock::now();
//        auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::steady_clock::now() - time_start);
//        std::cout<<duration.count()<<std::endl;
//        cv::imshow("cur_frame", frame);
//        cv::waitKey(1);
//    }
//    return 0;

//    std::string camera_url = "rtsp://admin:lingzhi123321@192.168.1.210/h264/ch34/main/av_stream/?user_name=admin?password=lingzhi123321?linkmode=tcp";

    cam_reader.init(camera_url);
    cam_reader.start();
    std::thread detect(detect_thread);
    std::thread server(WebSocket_Server);
//    cam_reader.start();
    detect.join();
    server.join();
//    detect_thread();
//    cam_reader.wait_done();

    return 0;
}