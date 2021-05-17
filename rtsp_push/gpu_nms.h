//
// Created by ganyi on 20-12-6.
//

#ifndef RTSP_GPU_NMS_H
#define RTSP_GPU_NMS_H
void _nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id);

#endif //RTSP_GPU_NMS_H
