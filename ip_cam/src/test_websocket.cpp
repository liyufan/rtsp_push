//
// Created by ganyi on 20-10-3.
//

#include <uWS.h>
#include <iostream>
#include <memory>

using namespace std;

void testConnections() {
    uWS::Hub h;

    h.onError([](void *user) {
        switch ((long) user) {
            case 1:
                std::cout << "Client emitted error on invalid URI" << std::endl;
                break;
            case 2:
                std::cout << "Client emitted error on resolve failure" << std::endl;
                break;
            case 3:
                std::cout << "Client emitted error on connection timeout (non-SSL)" << std::endl;
                break;
            case 5:
                std::cout << "Client emitted error on connection timeout (SSL)" << std::endl;
                break;
            case 6:
                std::cout << "Client emitted error on HTTP response without upgrade (non-SSL)" << std::endl;
                break;
            case 7:
                std::cout << "Client emitted error on HTTP response without upgrade (SSL)" << std::endl;
                break;
            case 10:
                std::cout << "Client emitted error on poll error" << std::endl;
                break;
            case 11:
                static int protocolErrorCount = 0;
                protocolErrorCount++;
                std::cout << "Client emitted error on invalid protocol" << std::endl;
                if (protocolErrorCount > 1) {
                    std::cout << "FAILURE:  " << protocolErrorCount << " errors emitted for one connection!" << std::endl;
                    exit(-1);
                }
                break;
            default:
                std::cout << "FAILURE: " << user << " should not emit error!" << std::endl;
                exit(-1);
        }
    });

    h.onConnection([](uWS::WebSocket<uWS::CLIENT> *ws, uWS::HttpRequest req) {
        switch ((long) ws->getUserData()) {
            case 8:
                std::cout << "Client established a remote connection over non-SSL" << std::endl;
                ws->close(1000);
                break;
            case 9:
                std::cout << "Client established a remote connection over SSL" << std::endl;
                ws->close(1000);
                break;
            default:
                std::cout << "FAILURE: " << ws->getUserData() << " should not connect!" << std::endl;
                exit(-1);
        }
    });

    h.onDisconnection([](uWS::WebSocket<uWS::CLIENT> *ws, int code, char *message, size_t length) {
        std::cout << "Client got disconnected with data: " << ws->getUserData() << ", code: " << code << ", message: <" << std::string(message, length) << ">" << std::endl;
    });

    h.connect("invalid URI", (void *) 1);
    h.connect("invalid://validButUnknown.yolo", (void *) 11);
    h.connect("ws://validButUnknown.yolo", (void *) 2);
    h.connect("ws://echo.websocket.org", (void *) 3, {}, 10);
    h.connect("ws://echo.websocket.org", (void *) 8);
    h.connect("wss://echo.websocket.org", (void *) 5, {}, 10);
    h.connect("wss://echo.websocket.org", (void *) 9);
    h.connect("ws://baidu.com", (void *) 6);
    h.connect("wss://baidu.com", (void *) 7);
    h.connect("ws://127.0.0.1:6000", (void *) 10, {}, 60000);

    h.run();
    std::cout << "Falling through testConnections" << std::endl;
}

void serveBenchmark() {
    uWS::Hub h;

    h.onConnection([](uWS::WebSocket<uWS::SERVER>* ws, uWS::HttpRequest req) {
        uWS::Header clientHeader = req.getUrl();
        std::cout<<"client: "<<clientHeader<<" join the connection"<<std::endl;
    });

    h.onMessage([&h](uWS::WebSocket<uWS::SERVER> *ws, char *message, size_t length, uWS::OpCode opCode) {
        ws->send(message, length, opCode);
    });

    h.onDisconnection([](uWS::WebSocket<uWS::SERVER> *ws, int code, char* message, size_t length) {
        std::cout << "Client got disconnected with data: " << ws->getUserData() << ", code: " << code << ", message: <" << std::string(message, length) << ">" << std::endl;
    });

    //h.getDefaultGroup<uWS::SERVER>().startAutoPing(1000);
    h.listen(3000);
    h.run();
}

void test_client() {
    using namespace uWS;
    using namespace std;
    Hub h;
    h.onConnection([](WebSocket<SERVER> *ws, HttpRequest req) {
        cout<<"This is Server, there is one client join the connection"<<endl;
    });

    h.onMessage([](WebSocket<SERVER> *ws, char* message, size_t length, OpCode opCode) {
        cout<<"This is Server, the message we received is: ";
        cout<<message<<endl;
        cout<<"the Server will repeat this message back to the client"<<endl;
        string repeat_message = message;
        cout<<"received messgae size is :"<<length<<endl;
        cout<<"the out message size is "<<repeat_message.size()<<endl;
        ws->send(repeat_message.c_str(), length, opCode);
        cout<<"the message is successfully send out"<<endl;


    });

    h.onDisconnection([](WebSocket<SERVER> *ws, int code, char* message, size_t length) {
        cout<<"disconnection code: "<<code<<endl;
        cout<<"the connection is shutdown simply by client"<<endl;
        cout<<"this is the last message we received: ";
        cout<<message<<endl;
//        ws->close();

    });

    if (h.listen(3000)){
        h.run();
    }
//    h.run();
}

int main(int argc, char** argv) {
    test_client();
//    serveBenchmark();
    return 0;
}
