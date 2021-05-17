# 网络摄像头调用模块

## 模块
    - c++: CameraReader
    - python: ip_camera_lib

## 说明
    通过c++多线程读取摄像头,解决网络摄像头单线程时视频流卡死现象。
    python库为libip_camera_lib.so，默认安装位置/usr/local/lib

## 编译&安装
目前仅安装python插件
```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
sudo make install
```

## 安装python模块
```shell
sudo python setup.py install
```
