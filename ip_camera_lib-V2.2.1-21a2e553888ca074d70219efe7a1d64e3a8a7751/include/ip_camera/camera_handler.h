//
// Created by nimrod on 19-4-17.
//

#ifndef IP_CAMERA_CAMERA_HANDLER_H
#define IP_CAMERA_CAMERA_HANDLER_H

#include <iostream>

class CameraHandler
{
public:
    CameraHandler (){}
    void init(std::string camera_url);
    void run();
    void start();
private:
    bool _working;
};


#endif //IP_CAMERA_CAMERA_HANDLER_H
