//
// Created by nimrod on 19-4-17.
//

#ifndef IP_CAMERA_PYBRIDGE_H
#define IP_CAMERA_PYBRIDGE_H

#include <python2.7/Python.h>

extern "C"
{
    bool init(char *url);
    void start();
    void stop();
//    void get_frame_pos(unsigned char ** pbuf);
    int get_frame_buf(unsigned char *pbuf);
};


#endif //IP_CAMERA_PYBRIDGE_H
