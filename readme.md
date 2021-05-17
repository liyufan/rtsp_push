### Main Feature 

- 本项目分别由服务端代码和客户端代码两部分组成，其中服务端代码运行在服务器上主要由C++完成，客户端代码运行在客户端用户上，主要由Java完成
- 目前客户端代码可实时接收外部rtsp视频流并显示，rtsp视频流由网络摄像头发送
- 服务端与客户端的通信是通过websocket协议完成，通信内容格式为json格式




**目录 (Table of Contents)**

[TOCM]

[TOC]

# 部分实现细节

##服务端与客户端数据交换内容目前定义为如下

public class DetObj implements Serializable {
    private String name;
        private int Id;
            private boolean chosen;

            public DetObj(String name, int Id) {
                    this.name = name;
                            this.Id = Id;
                                    this.chosen = false;
                                        
            }

            public String getName() {
                    return name;
                        
            }
                public void setName(String name) {this.name = name;}
                public int getId() {
                        return Id;
                            
                }
                public boolean getchosen() {
                        return chosen;
                            
                }
                public void setChosen(){
                        chosen = !chosen;
                            
                }
                    public String toString() {return name;}
                        
}
这部分之后可以根据具体需要进行修改，以实现更多feature，这部分数据会被序列化为json格式，进行数据传输，在服务端使用的序列化工具为[jsoncpp](https://github.com/open-source-parsers/jsoncpp), 在客户端使用的序列化工具为[gson](https://github.com/google/gson)

##目标检测算法
目前本项目使用的目标检测算法为yolov3，后续可以采用自研目标检测算法来进行集成，目标检测算法研发的推荐工具为[mmdetection](https://github.com/open-mmlab/mmdetection) 研究出对应算法后，可以通过[onnx](https://github.com/onnx/onnx)与[onnxruntime](https://github.com/microsoft/onnxruntime)来进行使用部署，onnx与onnxruntime目前是较为合适的开源部署工具之一，存在一些小坑，部署方法可见
    /home/ganyi/vscodeproject/rtsp/rtsp_push/test_onnx.cpp

##使用相关

开启服务端数据推送：`bash /home/ganyi/vscodeproject/rtsp/rtsp_push/cmake-build-debug/rtsp_push`
该版本是之前实验使用的的debug版本，基本满足要求，后期如果需要，可以编译为相关的release版本

android代码地址为 `/home/ganyi/vscodeproject/vlc-example-streamplayer.zip` 代码需要采用AndroidStudio进行编译后，安装在对应的android机器上。在未开启服务器推送前，android代码仅能获取到网络摄像头转发的视频流信号，在开启服务器推送功能之后，才可以进行目标检测

