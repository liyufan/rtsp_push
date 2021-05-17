//
// Created by nimrod on 19-11-21.
//

#include "ip_camera/frame_packer.h"

int main()
{
    FramePacker packer;
    cv::Mat frame;

    auto packs = packer.pack(frame);

    return 0;
}

