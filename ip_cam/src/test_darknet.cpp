//
// Created by ganyi on 20-9-28.
//
#include "stb_image.h"
#include "darknet.h"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>
#include <time.h>
#include <uWS.h>
#include <json/json.h>
#include <chrono>



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
float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

std::time_t getTimeStamp()
{
    std::chrono::time_point<std::chrono::system_clock,std::chrono::milliseconds> tp = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());//获取当前时间点
    std::time_t timestamp =  tp.time_since_epoch().count();
    return timestamp;
}

void print_yolo_detections(FILE **fps, char *id, int total, int classes, int w, int h, detection *dets)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                                         xmin, ymin, xmax, ymax);
        }
    }
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
                cout<<"bbox coordinates are: "<<dets[i].bbox.x<<" "<<dets[i].bbox.y<<" "<<dets[i].bbox.w<<" "<<dets[i].bbox.h<<endl;
                cout<<"the prob is: "<<dets[i].prob[j]<<"the class is: "<<j<<endl;

            }
        }

    }
    cout<<draw_bbox<<" bboxes are draw on the image"<<endl;
}

void one_frame_yolo(char *cfg, char* weights, cv::Mat& frame, std::vector<Object>& valid_box) {
    using namespace cv;
    using namespace std;
    network* net = load_network(cfg, weights, 0);
    layer l = net->layers[net->n-1];
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

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
//    Mat resized_frame;
//    resize(frame, resized_frame, Size(w, h));
//    cvtColor(frame,frame,CV_BGR2RGB);
    unique_ptr<float []> ddata(new float[h_o*w_o*c_o*sizeof(float)]);
    for (int i=0; i<h_o; ++i) {
        for (int j=0; j<w_o; ++j) {
            for (int k=0; k<c_o; ++k) {

//                cout<<int(frame.data[i*w_o*c_o + j*c_o + k])<<" ";
//                cout<<int(frame.at<Vec3b>(i,j)[k])<<" ";

//                ddata[k*h_o*w_o + i*w_o + j] = (float)((frame.data[i*w_o*c_o + j*c_o + k] + 7)/255.);
                ddata[k*h_o*w_o + i * w_o + j] = (float)((frame.at<Vec3b>(i, j)[k])/255. );

            }
        }
//        cout<<endl;
    }
    for (int i=0; i<10; ++i) {
        for (int j=0; j<w_o; ++j) {
            cout<<ddata[i*w_o + j]<<" ";
        }
        cout<<endl;
    }
    cout<<"end"<<endl;
//    for (int i=0; i<10; ++i) {
//        for (int j=0; j<w_o; ++j) {
//            cout<<int(((uint8_t*)frame.data)[(i*w_o+j)*3])<<" ";
//        }
//        cout<<endl;
//    }
//    cout<<"cur"<<endl;
    image im = load_image_color(
            "/home/ganyi/darknet/data/dog.jpg",0,0);
    for (int i=0; i<10; ++i) {
        for (int j=0; j<im.w; ++j) {
            cout<<int(im.data[i*im.w + j]*255)<<" ";
        }
        cout<<endl;
    }
//    for (int k=0; k<c_o; ++k) {
//        for (int j=0; j<h_o; ++j) {
//            for (int i=0; i<w_o; ++i) {
//                ddata[i + w*j + w*h*k] = frame.at<Vec3b>(j, i)[k] / (float)255.;
////                ddata[j*w_o*c_o + i*c_o + k] = frame.at<Vec3b>(j, i)[k] / (float)255.;
//            }
//        }
//    }
    image boxed;
    boxed.w = w_o;
    boxed.h = h_o;
    boxed.c = c_o;
    boxed.data = ddata.get();
    save_image(boxed, "/home/ganyi/vscodeproject/rtsp/ip_cam/before_test");
    printf("the origin width and height is %, %d", w_o, h_o);
    image sized = letterbox_image(boxed, w, h);

    printf("sized width and height is %d, %d", sized.w, sized.h);
    save_image(sized, "/hodme/ganyi/vscodeproject/rtsp/ip_cam/text");
    network_predict(net, sized.data);
    int nboxes = 0;
    detection *dets = get_network_boxes(net, im.w, im.h, 0.5, 0.5, 0, 1, &nboxes);
    //detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
    cout<<nboxes<<" bbox are generated"<<endl;
//    if (nms) do_nms_sort(dets, l.side*l.side*l.n, classes, iou_thresh);
    if (nms) do_nms_sort(dets, nboxes, classes, iou_thresh);

    draw_yolo_detections(frame, dets, nboxes, classes, w_o, h_o, valid_box);
    free_detections(dets, nboxes);
//    free_image(sized);


}

void test_yolo(char *cfgfile, char *weightfile, char *filename, float thresh)
{
    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    layer l = net->layers[net->n-1];
    set_batch_network(net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    float nms=.4;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net->w, net->h);
        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));

        int nboxes = 0;
        detection *dets = get_network_boxes(net, 1, 1, thresh, 0.5, 0, 1, &nboxes);
        if (nms) do_nms_sort(dets, l.side*l.side*l.n, l.classes, nms);

        draw_detections(im, dets, l.side*l.side*l.n, thresh, voc_names, alphabet, 20);
        save_image(im, "predictions");
        show_image(im, "predictions", 0);
        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}


float get_color(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    //printf("%f\n", r);
    return r;
}
static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}


void embed_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float val = get_pixel(source, x,y,k);
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

void draw_detections_cur(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes)
{
    int i,j;

    for(i = 0; i < num; ++i){
        char labelstr[4096] = {0};
        int class_now = -1;
        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j] > thresh){
                if (class_now < 0) {
                    strcat(labelstr, names[j]);
                    class_now = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                printf("%s: %.0f%%\n", names[j], dets[i].prob[j]*100);
            }
        }
        if(class_now >= 0){
            int width = im.h * .006;

            /*
               if(0){
               width = pow(prob, 1./2.)*10+1;
               alphabet = 0;
               }
             */

            //printf("%d %s: %.0f%%\n", i, names[class], prob*100);
            int offset = class_now*123457 % classes;
            float red = get_color(2,offset,classes);
            float green = get_color(1,offset,classes);
            float blue = get_color(0,offset,classes);
            float rgb[3];

            //width = prob*20+2;

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = dets[i].bbox;
            printf("bbox in the draw detector test\n");
            printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

            draw_box_width(im, left, top, right, bot, width, red, green, blue);
            if (alphabet) {
                image label = get_label(alphabet, labelstr, (im.h*.03));
                draw_label(im, top + width, left, label, rgb);
                free_image(label);
            }
            if (dets[i].mask){
                image mask = float_to_image(14, 14, 1, dets[i].mask);
                image resized_mask = resize_image(mask, b.w*im.w, b.h*im.h);
                image tmask = threshold_image(resized_mask, .5);

                embed_image(tmask, im, left, top);
                free_image(mask);
                free_image(resized_mask);
                free_image(tmask);
            }
        }
    }
}


void validate_yolo(char *cfg, char *weights)
{
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("/home/pjreddie/data/voc/2007_test.txt");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    int j;
    FILE **fps = static_cast<FILE**>(calloc(classes, sizeof(FILE *)));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 8;

    image *val = static_cast<image*>(calloc(nthreads, sizeof(image)));
    image *val_resized = static_cast<image*>(calloc(nthreads, sizeof(image)));
    image *buf = static_cast<image*>(calloc(nthreads, sizeof(image)));
    image *buf_resized = static_cast<image*>(calloc(nthreads, sizeof(image)));
    pthread_t *thr = static_cast<pthread_t*>(calloc(nthreads, sizeof(pthread_t)));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, 0, 0, 0, &nboxes);
            if (nms) do_nms_sort(dets, l.side*l.side*l.n, classes, iou_thresh);
            print_yolo_detections(fps, id, l.side*l.side*l.n, classes, w, h, dets);
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}


void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
//    using namespace std;
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        using namespace std;
        image im = load_image_color(input,0,0);
        unique_ptr<uchar[]> data(new uchar[im.w * im.h * im.c]);
        for (int i=0; i<10; ++i) {
            for (int j=0; j<im.w; ++j) {
                for (int k=0; k<im.c; ++k) {
                    data[i*im.w*im.c + j*im.c + k] = (unsigned char)(im.data[k*im.h*im.w + i*im.w + j] * 255);
                    std::cout<<int(data[i*im.w*im.c + j*im.c + k])<<" ";
                }
            }
            cout<<endl;
//            std::cout<<std::endl;
        }
        cout<<endl;

        cout<<"cut "<<endl;
        cv::Mat frame(im.h, im.w, CV_8UC3, data.get());


        for (int i=0; i<10; ++i) {
            for (int j=0; j<im.w; ++j) {
                for (int k=0; k<im.c; ++k) {

                    cout<<int(frame.data[i*im.w*im.c + j*im.c + k])<<" ";
//                cout<<int(frame.at<Vec3b>(i,j)[k])<<" ";
                }
            }
            cout<<endl;
        }

        cv::cvtColor(frame, frame, CV_RGB2BGR);


        cv::imwrite("/home/ganyi/vscodeproject/rtsp/ip_cam/test_true.jpg", frame);
        cv::imshow("test", frame);
        cv::waitKey(0);
        printf("the width and height is %d %d", im.w, im.h);
//        save_image(im, )
        image sized = letterbox_image(im, net->w, net->h);
        printf("the sized width and height is %d %d", sized.w, sized.h);
//        save_image(sized, "/home/ganyi/vscodeproject/rtsp/ip_cam/test_true");
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];


        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, 80, nms);
        draw_detections_cur(im, dets, nboxes, thresh, names, alphabet, 80);
        free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            make_window("predictions", 512, 512, 0);
            show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}

image load_image_stb(char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
        exit(0);
    }
    if(channels) c = channels;
    int i,j,k;
    image im = make_image(w, h, c);
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index]/255.;
            }
        }
    }
    free(data);
    return im;
}


int main(int argc, char**argv) {
    using namespace std;
    using namespace cv;
    using namespace uWS;

    char* weight = "/home/ganyi/darknet/yolov3.weights";
    char* cfg = "/home/ganyi/darknet/cfg/yolov3.cfg";
    char* img_path = "/home/ganyi/darknet/data/dog.jpg";
    char* data_path = "/home/ganyi/darknet/cfg/coco.data";
    Mat frame = imread("/home/ganyi/darknet/data/dog.jpg");
    std::vector<Object> dets;
    cvtColor(frame, frame, CV_BGR2RGB);

//    char* image_path = "/home/ganyi/darknet/data/dog.jpg";
//    cout<<frame.cols<<" "<<frame.rows<<endl;
//    image im = load_image_color(image_path, 0, 0);
//    image sized = letterbox_image(im, net->w, net->h);
//    resize(frame, frame, Size(int(frame.cols * 0.25), int(frame.rows * 0.25)));
    one_frame_yolo(cfg, weight, frame, dets);
    cvtColor(frame, frame, CV_BGR2RGB);
    imshow("test", frame);
    waitKey(0);
//    Json::Value root;
//    time_t timestamp = getTimeStamp();
//
//    root["timestamp"] = to_string(timestamp).c_str();
//    root["url"] = "192.168.1.106";
//    root["frameId"] = "unknow";
//    for (Object& elem: dets) {
//        Json::Value jsonObj;
//        jsonObj["cls"] = elem.cls;
//        jsonObj["x1"] = elem.x1;
//        jsonObj["y1"] = elem.y1;
//        jsonObj["x2"] = elem.x2;
//        jsonObj["y2"] = elem.y2;
//        root["BoundingBox"].append(jsonObj);
//    }
//    Hub h;
//    h.onConnection([](WebSocket<SERVER> *ws, HttpRequest req) {
//        cout<<"Connection is successfully build"<<endl;
//    });
//
//    h.onMessage([&root](WebSocket<SERVER>* ws, char* message, size_t length, OpCode opCode) {
//        cout<<"received message is: ";
//        cout<<message<<endl;
//        JSONCPP_STRING err;
//        Json::Value client_message;
//        Json::CharReaderBuilder builder;
//        const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
//        if (!reader->parse(message, message + length, &client_message, &err)) {
//            cout<<"the message seems can not parse in json format, the message is: ";
//            cout<<message<<endl;
//            return ;
//        }
//        if (client_message["op"] == "bbox_request") {
//            string json_bbox = root.toStyledString();
//            size_t json_length = json_bbox.length();
//            ws->send(json_bbox.c_str(), json_length, opCode);
//            cout<<"bounding box messages are successfully send out"<<endl;
//        }
//
//
//    });
//
//    h.onDisconnection([](WebSocket<SERVER> *ws, int code, char* message, size_t length) {
//       cout<<"The connection is break, this is the last message: "<<endl;
//       cout<<message<<endl;
//       cout<<"disconnection code: "<<code<<endl;
//    });
//    cout<<root<<endl;
//    if (h.listen(3000)) {
//        h.run();
//    }
//    imshow("text_yolo", frame);
//    test_yolo(cfg, weight, img_path, 0.01);
//    test_detector(data_path, cfg, weight,img_path, 0.5, 0.5, "/home/ganyi/vscodeproject/rtsp/ip_cam/prediction", 0 );
//    waitKey(0);
//    for (int i=0; i<10; ++i) {
//        for (int j=0; j<frame.cols; ++j) {
//            for (int k=0; k<3; ++k) {
//
//                cout<<int(frame.data[i*frame.cols*3 + j*3 + k])<<" ";
////                cout<<int(frame.at<Vec3b>(i,j)[k])<<" ";
//            }
//        }
//        cout<<endl;
//    }
    return 0;

}

