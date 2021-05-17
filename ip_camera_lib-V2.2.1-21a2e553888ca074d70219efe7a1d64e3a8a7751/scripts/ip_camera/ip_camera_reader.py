# coding=utf-8

import ctypes
import time
import cv2
import numpy as np

load_lib = ctypes.cdll.LoadLibrary
cam_lib = load_lib('libip_cam_reader.so')
IMAGE_SIZE = 1024 * 1024


class IPCameraReader:
    def __init__(self):
        self.pbuf = ctypes.create_string_buffer(IMAGE_SIZE)
        self.buf = ctypes.cast(self.pbuf, ctypes.POINTER(ctypes.c_ubyte))
        self.buf_size = ctypes.c_int(cam_lib.get_frame_buf(self.pbuf))

    def start(self, url):
        url += '\0'
        c_url = ctypes.create_string_buffer(len(url))
        c_url.value = url
        print(len(url))
        ret = ctypes.c_bool(cam_lib.init(c_url))
        if ret.value:
            cam_lib.start()
        else:
            print 'IPCameraReader: cam {url} cannot start'.format(url=url)
        return ret.value

    def read(self):
        self.buf_size = ctypes.c_int(cam_lib.get_frame_buf(self.pbuf))
        if self.buf_size.value == 0:
            return False, None
        return True, cv2.imdecode(np.array(self.buf[:self.buf_size.value], np.uint8), cv2.IMREAD_COLOR)

    def get_jpg(self):
        return self.buf_size, self.pbuf

    def stop(self):
        cam_lib.stop()


if __name__ == '__main__':
    camera = IPCameraReader()
    camera.start('rtsp://admin:lingzhi123321@192.168.1.108:554')
    while cv2.waitKey(30) != 27:
        ret, frame = camera.read()
        if ret:
            cv2.imshow('test', frame)
