// RTSP-Server

#include "xop/RtspServer.h"
#include "net/NetInterface.h"
#include "net/Timer.h"
#include <thread>
#include <memory>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <frame_packer.h>
#include <mutex>
#include <camera_reader.h>
#include <uWS/uWS.h>
#include <json/json.h>
#include <darknet.h>
#include <queue>

typedef std::chrono::time_point<std::chrono::steady_clock, std::chrono::milliseconds> timep;

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
//std::vector<std::vector<Object> > dets;
//std::queue<std::pair<timep, std::vector<Object>>> dets;
//std::queue<std::vector<Object> > dets;
std::queue<std::string> det_res;

cv::VideoCapture cap;
CameraReader cam_reader;
//FramePacker packer;
FramePacker *_ppacker = NULL;
xop::RtspServer *_pserver = NULL;
bool _need_sps = false;
int _num_client = 0;
std::mutex _mutex;

void snedFrameThread(xop::RtspServer* rtspServer, xop::MediaSessionId sessionId);
void on_notify(xop::MediaSessionId sessionId, uint32_t numClients);
std::time_t getTimeStamp();
void yolo_predict(network* net, cv::Mat& frame, std::vector<Object>& valid_box);
void draw_yolo_detections(cv::Mat& frame, detection *dets, int total, int classes, int w, int h, std::vector<Object>& valid_box);
void detect_thread();
void WebSocket_Server();



int main(int argc, char **argv)
{
    std::string ip = "192.168.1.106";
    std::string rtspUrl;

    std::shared_ptr<xop::EventLoop> eventLoop(new xop::EventLoop());
    xop::RtspServer server(eventLoop.get(), "0.0.0.0", 8554);
    _pserver = &server;
    std::string camera_url = "rtsp://admin:rcir219219@192.168.1.209";
    cam_reader.init(camera_url);
    cam_reader.start();

//    if (argc!=2)
//    {
//        std::cerr<<"specify the ScriptModule path"<<std::endl;
//        return -1;
//    }
//    torch::jit::script::Module model;
//    try
//    {
//        torch::jit::load(argv[1]);
//    }
//    catch (const c10::Error& e)
//    {
//        std::cerr<<"error loading the model"<<std::endl;
//        return -1;
//    }
//    model.to(at::kCUDA);


#ifdef AUTH_CONFIG
	server.setAuthConfig("-_-", "admin", "12345");
#endif

    xop::MediaSession *session = xop::MediaSession::createNew("live"); 
    rtspUrl = "rtsp://" + ip + ":18554/" + session->getRtspUrlSuffix();

    session->addMediaSource(xop::channel_0, xop::H264Source::createNew()); 
	//session->startMulticast();  /* enable multicast */
//	session->setNotifyCallback([] (xop::MediaSessionId sessionId, uint32_t clients){
//		std::cout << "Number of rtsp client : " << clients << std::endl;
//	});
    session->setNotifyCallback(on_notify);
   
    xop::MediaSessionId sessionId = server.addMeidaSession(session); 

//    std::thread t1(snedFrameThread, &server, sessionId);
    std::thread detect(detect_thread);
    std::thread WSserver(WebSocket_Server);
//    t1.detach();
    detect.detach();
    WSserver.detach();

    std::cout << "Play URL: " <<rtspUrl << std::endl;

	while (1)
	{
		xop::Timer::sleep(100);
	}

    getchar();
    return 0;
}

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
                ddata[k*h_o*w_o + i * w_o + j] = (float)((frame.at<Vec3b>(i, j)[k]) / 255.);

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
    detection *dets = get_network_boxes(net, boxed.w, boxed.h, 0.5, 0.5, 0, 1, &nboxes);
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

timep getTimeP() {
    return std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now());
}

//std::time_t getTimeStamp()
//{
//    std::chrono::time_point<std::chrono::system_clock,std::chrono::milliseconds> tp = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());//获取当前时间点
//    std::time_t timestamp =  tp.time_since_epoch().count();
//    return timestamp;
//}

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
//        cv::cuda::GpuMat dframe;
//        std::cout << "get_frame waiting..." << std::endl;
        cam_reader.get_frame(frame);
//        std::cout<<frame.cols<<" "<<frame.rows<<std::endl;
//        frame = cam_reader.get_frame();
//        cv::imshow("cur_fame", frame);
//        cv::waitKey(1);
//        dframe.upload(frame);
//        std::cout << "getframe" << frame.rows<< " " << frame.cols << " work:"<< cam_reader.is_work() << std::endl;
        if(frame.empty())
        {
            std::cout << "get_frame empty" << std::endl;
            continue;
        }else {
            detections.clear();
//            std::cout << "getframe" << frame.rows<< " " << frame.cols << " work:"<< cam_reader.is_work() << std::endl;
            timep start = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now());

            yolo_predict(net, frame, detections);
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()) - start);
            printf("the duration of detection is: %ld milisecond\n", duration.count());
            {
//                cv::imshow("simple_test", frame);
//                cv::waitKey(1);
                std::unique_lock<std::mutex> tmp(globalMutex);
//                timep timepoint = getTimeP();
//                dets.push(make_pair(timepoint,detections));
                dets = detections;
//                std::cout<<"The size of detection queue is: "<<dets.size()<<std::endl;
//                cv::imshow("rtsp_stream", frame);
//                cv::waitKey(1);
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

                    std::unique_lock<std::mutex>(globalMutex);

                    time_t timeStamp= getTimeStamp();
                    message_send["timestamp"] = std::to_string(timeStamp).c_str();
                    message_send["url"] = "192.168.1.106";
                    message_send["frameId"] = "unknow";
                    Json::Value emptyObj;
                    emptyObj["cls"] = 79;
                    emptyObj["x1"] = 0.0;
                    emptyObj["x2"] = 0.0;
                    emptyObj["y1"] = 0.0;
                    emptyObj["y2"] = 0.0;
                    message_send["bboxes"].append(emptyObj);

//                    if (det_res.size() <= 10) {
//                        std::string SerilizedMessage = message_send.toStyledString();
//                        size_t Message_length = SerilizedMessage.length();
//                        ws->send(SerilizedMessage.c_str(), Message_length, opCode);
//                    }
//                    while(true) {
//                        if (dets.empty()) continue;
//                        std::pair<timep, std::vector<Object> > Detection;
//                        {
//                            std::unique_lock<std::mutex> l(globalMutex);
//                            Detection = dets.front();
//                        }
//                            timep det_time = Detection.first;
//                            timep cur_time = getTimeP();
//                            long time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - det_time).count();
//                            printf("The time diff is %ld\n", time_diff);
//                            if ( time_diff > 300) dets.pop();
//                            else {
//                                for (Object& elem: Detection.second) {
//                                    Json::Value jsonObj;
//                                    jsonObj["cls"] = elem.cls;
//                                    jsonObj["x1"] = elem.x1;
//                                    jsonObj["x2"] = elem.x2;
//                                    jsonObj["y1"] = elem.y1;
//                                    jsonObj["y2"] = elem.y2;
//                                    message_send["bboxes"].append(jsonObj);
//                                }
//                                break;
//                            }
//                    }
//                    std::string SerializedMessage = message_send.toStyledString();
//                    det_res.push(SerializedMessage);
//                    ws->send(SerializedMessage.c_str(), SerializedMessage.length(), opCode);
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
                    det_res.push(SerializedMessage);
                    ws->send(SerializedMessage.c_str(), SerializedMessage.length(), opCode);
//                    if (det_res.size() > 10) {
//                        SerializedMessage = det_res.front();
//                        det_res.pop();
//                        size_t Message_length = SerializedMessage.length();
//                        ws->send(SerializedMessage.c_str(), Message_length, opCode);
//                    }
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

void on_notify(xop::MediaSessionId sessionId, uint32_t clients)
{
		std::cout << "Number of rtsp client : " << clients << std::endl;
		_need_sps = true;

//        if(clients == 0)
//        {
//            std::unique_lock<std::mutex> lock(_mutex);
//            if(_ppacker != NULL)
//            {
//                delete _ppacker;
//                _ppacker = NULL;
//            }
//        }


    //new client connect in
		if(_num_client < clients)
        {
            std::cout<< "on_notify:: waiting lock..." << std::endl;
            std::unique_lock<std::mutex> lock(_mutex);
            std::cout<< "on_notify:: get lock" << std::endl;
            if(_ppacker != NULL)
            {
                delete _ppacker;
                _ppacker = NULL;
            }

            _ppacker = new FramePacker();

            _ppacker->init(0);
            _ppacker->set_frame_size(640, 480);
            _ppacker->init_encoder();
        }

		_num_client = clients;

//		packer.end_encode();
//		std::vector<uint8_t> pack;
//
//		packer.get_sequence_params(pack);
//        xop::AVFrame videoFrame = {0};
//        videoFrame.type = 0;
//        videoFrame.size = pack.size();
//        videoFrame.timestamp = xop::H264Source::getTimeStamp();
//        videoFrame.buffer.reset(new uint8_t[videoFrame.size]);
//        memcpy(videoFrame.buffer.get(), pack.data(), videoFrame.size);
//        std::cout<< "push sps: "<< videoFrame.size << std::endl;
//        _pserver->pushFrame(sessionId, xop::channel_0, videoFrame);
}

void snedFrameThread(xop::RtspServer* rtspServer, xop::MediaSessionId sessionId)
{       
    cv::VideoCapture cap(0);
    bool run_flag = true;

    std::string szOutFilePath("./tmp.h264");

    std::ofstream fpOut(szOutFilePath, std::ios::out | std::ios::binary);
    if (!fpOut)
    {
        std::ostringstream err;
        err << "Unable to open output file: " << szOutFilePath << std::endl;
        throw std::invalid_argument(err.str());
    }

    while(run_flag)
    {
//        int frameSize = h264File->readFrame((char*)frameBuf, bufSize, &bEndOfFrame);

        std::vector<std::vector<uint8_t>> packs;

        {
//            std::cout<< "work thread:: waiting lock..." << std::endl;
            std::unique_lock<std::mutex> lock(_mutex);
//            std::cout<< "work thread:: get lock"<< std::endl;
            if(!_ppacker || !_ppacker->is_initialized())
            {
                xop::Timer::sleep(40);
                continue;
            }

            cv::Mat frame;
            cam_reader.get_frame(frame);
//            cv::imshow("test", frame);
//            cv::waitKey(0);
//            cap >> frame;
//            cv::cvtColor(frame, frame, CV_BGR2RGB);
//            frame.convertTo(frame, CV_32F, 1.0/255);
//            auto image_tensor = torch::from_blob(frame.data, {1,3,1920,1080}, torch::kFloat32);
//            std::vector<torch::jit::IValue> input;
//            input.emplace_back(image_tensor);
//            at::Tensor res = model.forward(input).toTensor();
//            cv::rectangle(frame, cv::Rect(100, 100, 100, 100), cv::Scalar(0,255,0), 1);
            packs = _ppacker->pack(frame);
        }

        for(auto pack: packs)
        {
            xop::AVFrame videoFrame = {0};
            videoFrame.type = 0;
            videoFrame.size = pack.size();
            videoFrame.timestamp = xop::H264Source::getTimeStamp();
            videoFrame.buffer.reset(new uint8_t[videoFrame.size]);
            memcpy(videoFrame.buffer.get(), pack.data(), videoFrame.size);
            std::cout<< "pushFrame: "<< videoFrame.size << std::endl;
            rtspServer->pushFrame(sessionId, xop::channel_0, videoFrame);

            fpOut.write(reinterpret_cast<char*>(pack.data()), pack.size());
        }

#if 0
        if(_need_sps)
        {
            _need_sps = false;
            std::vector<uint8_t> pack;

            packer.get_sequence_params(pack);
            xop::AVFrame videoFrame = {0};
            videoFrame.type = 0;
            videoFrame.size = pack.size();
            videoFrame.timestamp = xop::H264Source::getTimeStamp();
            videoFrame.buffer.reset(new uint8_t[videoFrame.size]);
            memcpy(videoFrame.buffer.get(), pack.data(), videoFrame.size);
            std::cout<< "push sps: "<< videoFrame.size << std::endl;
            rtspServer->pushFrame(sessionId, xop::channel_0, videoFrame);
        }
#endif
//        for(auto pack: packer.end_encode())

//        {
//            xop::AVFrame videoFrame = {0};
//            videoFrame.type = 0;
//            videoFrame.size = pack.size();
//            videoFrame.timestamp = xop::H264Source::getTimeStamp();
//            videoFrame.buffer.reset(new uint8_t[videoFrame.size]);
//            memcpy(videoFrame.buffer.get(), pack.data(), videoFrame.size);
//            std::cout<< "pushFrame: "<< videoFrame.size << std::endl;
//            rtspServer->pushFrame(sessionId, xop::channel_0, videoFrame);
//
//            fpOut.write(reinterpret_cast<char*>(pack.data()), pack.size());
//        }

        xop::Timer::sleep(40);
    }

}


