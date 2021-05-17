//
// Created by ganyi on 20-12-2.
//
#include <algorithm>  // std::generate
#include <assert.h>
#include <string.h>
#include <string>
#include <iostream>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <setjmp.h>
#include <algorithm>
#include <memory>
#include <atomic>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <core/providers/cuda/cuda_provider_factory.h>
//#include <core/framework/ml_value.h>
//#include <core/framework/execution_provider.h>
//#include <onnxruntime_c_api.h>
//#include <core/framework/allocator.h>
//#include <core/framework/ml_value.h>
//#include <core/framework/alloc_kind.h>
#include <opencv2/opencv.hpp>
#include "gpu_nms.h"
//#include "providers.h"
//#include "local_filesystem.h"
//#include <sync_api.h>
#include <mutex>
//#define GPU_NMS
#define DEVICE_ID 0

//#include <experimental_onnxruntime_cxx_api.h>
using namespace std::chrono;
typedef std::chrono::time_point<std::chrono::steady_clock, std::chrono::milliseconds> timep;
using namespace std;
//typedef struct{
//    float x, y, w, h;
//} box;
//
//typedef struct detection{
//    box bbox;
//    int classes;
//    float *prob;
//    float *mask;
//    float objectness;
//    int sort_class;
//} detection;
class bbox {
public:
    float l;
    float r;
    float t;
    float b;
//    float scale;
    bbox(){
        l=r=t=b=0;
    }
    bbox(float left, float right, float top, float bottom):l(left),r(right),t(top),b(bottom){}
    bbox(bbox &&box):l(std::move(box.l)),r(std::move(box.r)),t(std::move(box.t)),b(std::move(box.b)){}
    bbox(const bbox &box) {
        l = box.l;
        r = box.r;
        t = box.t;
        b = box.b;
//        scale= box.scale;
    }
    bbox &operator=(const bbox &rhs)
    {
        if ( this == &rhs )
        {
            return *this;
        }

        this->l = rhs.l;
        this->r = rhs.r;
        this->t = rhs.t;
        this->b = rhs.b;
//        this->scale= rhs.scale;

        return *this;
    }
    ~bbox(){}

};

class detection{
public:
    bbox b;
//    float* cls;
    std::pair<int ,float> cls;
    float centerness;
    int object;

    detection(bbox bb, std::pair<int, float> c, float cen, int obj):b(bb), cls(c), centerness(cen), object(obj){
//        object = 1;
    }
    detection(const detection& det) {
        b = det.b;
        cls = det.cls;
        centerness = det.centerness;
        object = det.object;
    }
    detection(detection&& det):b(std::move(det.b)),cls(std::move(det.cls)), centerness(det.centerness), object(det.object)  {
    }
    detection &operator=(const detection &rhs)
    {
        if ( this == &rhs )
        {
            return *this;
        }

        this->b = rhs.b;
        this->cls = rhs.cls;
        this->centerness = rhs.centerness;
        this->object = rhs.object;

        return *this;
    }
    ~detection(){}
};
float sigmoid(float val);
float load_img(cv::Mat &img, string path, int img_h, int img_w, bool to_rgb);
void select_box_single_scale(float *reg_src, vector<bbox> &reg_dst, int feat_h, int feat_w, int stride, int img_h, int img_w, bool debug_scale, float scale);
void select_cls_single_scale(float *cls_src, vector<std::pair<int,float>> &cls_dst, int feat_h, int feat_w, int cls_num, bool debug_scale);
void select_center_single_scale(float *cen_src, vector<float> &cen_dst, int feat_h, int feat_w, bool debug_scale);
void one_scale_preprocess(vector<pair<int, float>> &cls_scores, vector<bbox> &bbox_regs, vector<float> &centers, vector<detection>& dst, int nms_pre, float nms_scores, bool debug_scale);
void decode_detection(vector<detection> &dst, std::vector<Ort::Value> &output_tensor, std::vector<std::vector<int64_t>> &total_output_dims, std::vector<int>& strides, int scale, int img_h, int img_w, int nms_pre, float nms_score, float);
float iou(bbox &b1, bbox &b2);
void nms(vector<detection>& dets, float nms_iou, vector<detection>& summary);
void normalize_img(cv::Mat &img, vector<float> &mean, vector<float> &norm, float* data);
void transpose_img(float* src, float* dst, int img_h, int img_w);
void draw_result(cv::Mat& img, std::vector<detection>& dets);

void print_img(float* img, int height, int width) {
    for (int c=0; c<3; ++c) {
        cout << "channel: " << c + 1 << endl;

        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 10; ++j) {
                cout << img[c*width*height + i * width + j] << " ";
            }
            cout << endl;
        }
    }
}

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<int64_t>& v) {
    std::stringstream ss("");
    for (size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

int calculate_product(const std::vector<int64_t>& v) {
    int total = 1;
    for (auto& i : v) total *= i;
    return total;
}

void strcp(char* ptr1, char* ptr2) {
    strcpy(ptr1, ptr2);
}



int main(int argc, char** argv) {
//    if (argc != 2) {
//        std::cout << "Usage: ./onnx-api-example <onnx_model.onnx>" << std::endl;
//        return -1;
//    }
    std::string model_file = "/home/ganyi/fcos_1200_1600.onnx";
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions sessionOptions;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, DEVICE_ID));
//    sessionOptions.SetIntraOpNumThreads(1);
    printf("using onnx cxx api\n");

    Ort::Session session(env, model_file.c_str(), sessionOptions);
    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_node_names{"input.1"};
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<const char*> output_node_namess{"651", "763", "875", "987", "1099", "710", "822", "934", "1046", "1158", "705", "817",
    "929", "1041", "1153" };
    std::vector<float> mean{103.530, 116.280, 123.675};
    std::vector<float> norm{1.0, 1.0, 1.0};
    std::vector<int64_t> input_node_dims;
    std::vector<int64_t> output_node_dims;
    std::vector<std::vector<int64_t>> total_output_node_dims;



    printf("The number of input %zu\n", num_input_nodes);
    printf("The number of output %zu\n", num_output_nodes);

    for (int i=0; i<num_input_nodes; i++) {
//        char* input_name = session.GetInputName(i, allocator);
//        printf("Input %d: name=%s\n",i, input_name);
//        input_node_names[i] = input_name;

        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
//        printf("Input %d : type=%d\n", i ,type);

        input_node_dims = tensor_info.GetShape();
//        printf("Input name: %s\n", input_node_names[i]);
        printf("Input %d: num_dim=%zu\n", i, input_node_dims.size());
        for (int j=0; j<input_node_dims.size(); ++j)
            printf("Input %d: dim %d=%jd\n", i, j, input_node_dims[j]);
//        free(input_name);

    }
//    input_node_dims[2] = 600;
//    input_node_dims[3] = 800;
    size_t input_tensor_size = size_t(input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3]);
    std::unique_ptr<float[]> data(new float[input_tensor_size]);
    std::unique_ptr<float[]> normal_data(new float[input_tensor_size]);

    printf("output done!\n");
    cv::Mat img;
    float scale_factor = load_img(img, "/home/ganyi/darknet/data/dog.jpg", int(input_node_dims[2]), int(input_node_dims[3]), false);
    scale_factor = 1.04166666;
    cv::Mat img_to_show = cv::imread("/home/ganyi/darknet/data/dog.jpg");
    normalize_img(img, mean, norm, normal_data.get());
    transpose_img(normal_data.get(), data.get(), int(input_node_dims[2]), int(input_node_dims[3]));
//    print_img(data.get(), int(input_node_dims[2]), int(input_node_dims[3]));
//    size_t  input_tensor_size = input_node_dims[2] * input_node_dims[3] * input_node_dims[1];
//    size_t input_tensor_size = 800 * 800 * 3;
//    std::unique_ptr<float> input_tensor_values(new float[input_tensor_size]);
//    std::vector<float> input_tensor_values(input_tensor_size);
//    std::vector<const char*> output_node_names = {"out"};

//    printf("try allocate\n");
//    for (unsigned int i=0; i<input_tensor_size; ++i) {
//        input_tensor_values[i] = (float)i / (input_tensor_size + 1);
//    }

    printf("print done!\n");
    for (int i=0; i<num_output_nodes; ++i){
//        output_node_namess[i] = session.GetOutputName(i, allocator);

        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
//        session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
//        printf("Output %d : type=%d\n", i ,type);

        output_node_dims = tensor_info.GetShape();
        total_output_node_dims.push_back(output_node_dims);
//        printf("Output %d: num_dim=%zu\n", i, output_node_dims.size());
//        printf("The name of this node is %s\n", output_node_namess[i]);
//        for (int j=0; j<input_node_dims.size(); ++j)
//            printf("Output %d: dim %d=%jd\n", i, j, output_node_dims[j]);
    }
    vector<detection> final_result;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, data.get(), input_tensor_size, input_node_dims.data(), 4);
//    Ort::Value input_tensor =
    printf("tensor generate done!\n");

    assert(input_tensor.IsTensor());

    vector<int> strides{8, 16, 32, 64, 128};
    vector<detection> nms_res;
    timep start = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now());
//    for (int i=0; i<1000; ++i) {
//        final_result.clear();
//        nms_res.clear();
    for (int i=0; i<1000; ++i) {
        auto output_tensor = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, size_t(1),
                                         output_node_namess.data(), num_output_nodes);
//    }

        decode_detection(final_result, output_tensor, total_output_node_dims, strides, 5, int(input_node_dims[2]),
                         int(input_node_dims[3]), 1000, 0.3, scale_factor);
        nms(final_result, 0.5, nms_res);
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()) - start);
    printf("the duration of detection is: %ld milisecond\n", duration.count()/1000);
    printf("%d bbox left\n", nms_res.size());
    draw_result(img_to_show, nms_res);
    cv::imshow("result", img_to_show);
    cv::waitKey(0);
//    }


//    for (int i=0; i<output_tensor.size(); ++i) {
//        std::cout<<output_tensor[i].GetCount()<<std::endl;
//    }

//    float* floatarr = output_tensor.front().GetTensorMutableData<float>();
//    assert(abs(floatarr[0] - 0.000045) < 1e-6);

    printf("Done!\n");



    return 0;
}
float sigmoid(float val) {
    return 1 / (1 + exp(-val));
}

float load_img(cv::Mat &img, string path, int img_h, int img_w, bool to_rgb) {
    img = cv::imread(path);
    if (img.empty()) {
        cout<<"Load img failed"<<endl;
        exit(0);
    }
    if (to_rgb)
        cv::cvtColor(img, img, CV_BGR2RGB);
    float scale_factor = float(img.rows) / float(img_h);
    cv::resize(img, img, cv::Size(img_w, img_h));
    return scale_factor;


}

void normalize_img(cv::Mat &img, vector<float> &mean, vector<float> &norm, float* data) {
//    img.convertTo(img, CV_32FC3, 1.0, 0.0);
    float mean1 = mean[0];
    float mean2 = mean[1];
    float mean3 = mean[2];
    float norm1 = 1 / norm[0];
    float norm2 = 1 / norm[1];
    float norm3 = 1 / norm[2];
    float* ptr = data;
    for (int i=0; i<img.rows; ++i) {
        for (int j=0; j<img.cols; ++j) {
            *(ptr++) = (img.at<cv::Vec3b>(i,j)[0] - mean1) * norm1;
            *(ptr++) = (img.at<cv::Vec3b>(i,j)[1] - mean2) * norm2;
            *(ptr++) = (img.at<cv::Vec3b>(i,j)[2] - mean3) * norm3;
//            img.at<cv::Vec3b>(i, j)[0] = (img.at<cv::Vec3b>(i, j)[0] - mean1) * norm1;
//            img.at<cv::Vec3b>(i, j)[1] = (img.at<cv::Vec3b>(i, j)[1] - mean2) * norm2;
//            img.at<cv::Vec3b>(i, j)[2] = (img.at<cv::Vec3b>(i, j)[2] - mean3) * norm3;
        }
    }
}



void transpose_img(float* src, float* dst, int img_h, int img_w) {

    float* ptr = dst;
    for (int c=0; c<3; ++c) {
        for (int i = 0; i < img_h; ++i) {
            for (int j = 0; j < img_w; ++j) {
                int idx = i * img_w * 3 + j * 3 + c;
                *(ptr++) = src[idx];
            }
        }
    }
}

void select_cls_single_scale(float *cls_src, vector<std::pair<int,float>> &cls_dst, int feat_h, int feat_w, int cls_num, bool debug_scale) {
    for (int i=0; i<feat_h; ++i) {
        for (int j=0; j<feat_w; ++j) {
            int ids = -1;
            float prob=-10000.00;
            for (int k=0; k<cls_num; ++k) {
                int idx = k * feat_h * feat_w + i * feat_w + j;
                float cur_prob = cls_src[idx];
//                if (debug_scale) cout<<sigmoid(cur_prob)<<" ";
                if (prob < sigmoid(cur_prob)) {
                    prob = sigmoid(cur_prob);
                    ids = k;
                }
            }
//            cout<<endl;
            cls_dst.push_back(make_pair(ids, prob));
        }
    }
}

void select_box_single_scale(float *reg_src, vector<bbox> &reg_dst, int feat_h, int feat_w, int stride, int img_h, int img_w, bool debug_scale, float scale) {
    int offset = stride >> 1;
//    cout<<"scale factor"<<scale<<endl;
    for (int i=0; i<feat_h; ++i) {
        for (int j=0; j<feat_w; ++j) {
            float x_center = j * stride + offset;
            float y_center = i * stride + offset;
//            if (debug_scale) {
//                cout<<"("<<x_center<<", "<<y_center<<"), ";
//            }
            float left = reg_src[i * feat_w + j];
            float top = reg_src[feat_h * feat_w + i * feat_w + j];
            float right = reg_src[2 * feat_h * feat_w + i * feat_w + j];
            float bottom = reg_src[3 * feat_h * feat_w + i * feat_w + j];
            float left_cord = max(x_center - left, float(0));
            float right_cord = min(x_center + right, float(img_w));
            float top_cord = max(y_center - top, float(0));
            float bottom_cord = min(y_center + bottom, float(img_h));
            reg_dst.push_back(bbox(left_cord/scale, right_cord/scale, top_cord/scale, bottom_cord/scale));
//            if (debug_scale) {
//                cout << "(" << left_cord << ", " << top_cord << ", "<<right_cord<<", "<<bottom_cord<<")"<<endl;
//                cout<<"after push: "<<reg_dst.back().l<<", "<<reg_dst.back().t<<", "<<reg_dst.back().r<<", "<<reg_dst.back().b<<endl;
//            }
        }
    }
//    cout<<endl;
}

void select_center_single_scale(float *cen_src, vector<float> &cen_dst, int feat_h, int feat_w, bool debug_scale) {
    for (int i=0; i<feat_h; ++i) {
        for (int j=0; j<feat_w; ++j) {
            int idx = i * feat_w + j;
            cen_dst.push_back(sigmoid(cen_src[idx]));
//            if (debug_scale) printf("%f, \n", sigmoid(cen_src[idx]));
        }
    }
}



void one_scale_preprocess(vector<pair<int, float>> &cls_scores, vector<bbox> &bbox_regs, vector<float> &centers, vector<detection>& dst, int nms_pre, float nms_scores, bool debug_scale) {
    if (cls_scores.size() != bbox_regs.size() || bbox_regs.size() != cls_scores.size()) {
        cout<<"The decode size can not match, please check the size"<<endl;
        cout<<"cls size: "<<cls_scores.size()<<" , bbox size: "<<bbox_regs.size() << " , center size: "<<centers.size()<<endl;
        abort();
    }

    if (cls_scores.size() > nms_pre) {
        vector<detection> local_dst;
        for (int i=0; i<cls_scores.size(); ++i) {
            local_dst.push_back(detection(bbox_regs[i], cls_scores[i], centers[i], 1));
        }
        std::nth_element(local_dst.begin(), local_dst.begin() + nms_pre, local_dst.end(),
                [](detection& d1, detection& d2) {return (d1.cls.second * d1.centerness) > (d2.cls.second * d2.centerness);});
        for (int i=0; i<nms_pre; ++i) {
            if (local_dst[i].cls.second > nms_scores)
            dst.push_back(local_dst[i]);
        }
    }
    else{
        for (int i=0; i<cls_scores.size(); ++i) {

            if (cls_scores[i].second > nms_scores)
            dst.push_back(detection(bbox_regs[i], cls_scores[i], centers[i], 1));
//            if (debug_scale) {
//                printf("cls %d, score %f, bbox (%f, %f, %f, %f), center %f\n", cls_scores[i].first, cls_scores[i].second, bbox_regs[i].l, bbox_regs[i].t, bbox_regs[i].r, bbox_regs[i].b, centers[i]);
//                printf("the objective is %d", dst.back().object);
//            }

        }
    }

}

void draw_result(cv::Mat& img, std::vector<detection>& dets) {
    for (auto& elem: dets) {
        if (elem.object) {
            cv::rectangle(img, cv::Point(elem.b.l, elem.b.t), cv::Point(elem.b.r, elem.b.b), cv::Scalar(255, 0, 0), 2);
        }
    }
}

void decode_detection(vector<detection> &dst,
        std::vector<Ort::Value> &output_tensor,
        std::vector<std::vector<int64_t>> &total_output_dims,
        std::vector<int> &strides,
        int scale, int img_h, int img_w, int nms_pre=1000,
        float nms_score=0.05, float bbox_scale_factor=1.0) {
    vector<pair<int ,float>> cls_scores;
    vector<bbox> bbox_regs;
    vector<float> centers;

    for (int i=0; i<scale; ++i) {
        bool scale_for_debug = false;
//        cout<<"decode scale "<<i<<endl;
        float* cls_score_one_scale = output_tensor[i].GetTensorMutableData<float>();
        float* reg = output_tensor[scale + i].GetTensorMutableData<float>();
        float* centerness = output_tensor[2 * scale + i].GetTensorMutableData<float>();
        vector<int64_t> cls_dim = total_output_dims[i];
        vector<int64_t> reg_dim = total_output_dims[scale + i];
        vector<int64_t> cen_dim = total_output_dims[2 * scale + i];
        int feat_h = int(cls_dim[2]);
        int feat_w = int(cls_dim[3]);
        int cls_num = int(cls_dim[1]);
        int stride = strides[i];

//        printf("at scale %d, the feature map size is: (%d, %d), the stride is %d, the cls num is %d\n", i, feat_h, feat_w, stride, cls_num);

//        if (i==4) scale_for_debug = true;
        select_cls_single_scale(cls_score_one_scale, cls_scores, feat_h, feat_w, cls_num, scale_for_debug);
        select_box_single_scale(reg, bbox_regs, feat_h, feat_w, stride, img_h, img_w, scale_for_debug, bbox_scale_factor);
        select_center_single_scale(centerness, centers, feat_h, feat_w, scale_for_debug);
        one_scale_preprocess(cls_scores, bbox_regs, centers, dst, nms_pre, nms_score, scale_for_debug);
        cls_scores.clear();
        bbox_regs.clear();
        centers.clear();
    }
}

float iou(bbox &b1, bbox &b2) {
    float area1 = (b1.r - b1.l) * (b1.b - b1.t);
    float area2 = (b2.r - b2.l) * (b2.b - b2.t);
    float inter_r = min(b1.r, b2.r);
    float inter_l = max(b1.l, b2.l);
    float inter_t = max(b1.t, b2.t);
    float inter_b = min(b1.b, b2.b);
    float inter_area = (inter_r - inter_l) * (inter_b - inter_t);
    if (inter_area <= 0) return 0;
    return inter_area / (area1 + area2 - inter_area);
}
void data_transfer(vector<detection>& dets, float* dst) {
    float* ptr = dst;
    for (int i=0; i<dets.size(); ++i) {
        *(ptr++) = dets[i].b.l;
        *(ptr++) = dets[i].b.t;
        *(ptr++) = dets[i].b.r;
        *(ptr++) = dets[i].b.b;
        *(ptr++) = dets[i].cls.second;
    }
}

void nms(vector<detection>& dets, float nms_iou, vector<detection>& summary) {
//    cout<<"init dets size: "<<dets.size()<<endl;
//    vector<detection> summary;

    sort(dets.begin(), dets.end(), [](detection& d1, detection& d2) {return d1.cls.second * d1.centerness > d2.cls.second * d2.centerness;});
//    cout<<"sort done"<<endl;
#ifdef GPU_NMS
    /*the detection class format is not suitable with the gpu nms format, which may cause some time consumption.
     * TODO: change the detection format or the nms cuda code*/
    std::unique_ptr<float[]> dets_data(new float[dets.size() * 5]);

    int num_out=0;

    std::unique_ptr<int[]> keep_out(new int[300]);
    data_transfer(dets, dets_data.get());
//    timep start = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now());
//    for (int i=0; i<1000; ++i) {
    _nms(keep_out.get(), &num_out, dets_data.get(), dets.size(), 5, nms_iou, DEVICE_ID);
//    }
//    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()) - start);
//    printf("the duration of detection is: %ld microsecond\n", duration.count());
    for (int i = 0; i < num_out; ++i) {
//        int idx = keep_out[i] * 5;
        summary.push_back(dets[keep_out[i]]);
    }

//    printf("free det data is ok\n");
#else
//    timep start = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now());

        for (int i = 0; i < dets.size(); ++i) {
//        printf("box %d: (%f, %f, %f, %f) and objctive status is %d, cls is %d, with score %f\n", i, dets[i].b.l, dets[i].b.t, dets[i].b.r, dets[i].b.b, dets[i].object, dets[i].cls.first, dets[i].cls.second);
            if (dets[i].object != 0) {
                detection dets_cur = dets[i];
                for (int j = i + 1; j < dets.size(); ++j) {
                    if (dets[j].object == 0) continue;
                    if (dets[j].cls.first == dets_cur.cls.first && iou(dets[j].b, dets_cur.b) > nms_iou)
                        dets[j].object = 0;
                }
                summary.push_back(dets_cur);
            }
        }


#endif
//    cout<<"the bbox left after nms: "<<summary.size()<<endl;
}
