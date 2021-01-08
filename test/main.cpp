// g++ ./main.cpp base64.cpp -o main `pkg-config --cflags --libs opencv`

#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <array>
#include <string>

#include "base64.h"

using namespace std;

int main(int argc, char **argv) {
    std::string floatString;
    float data[5] = {1.1, 2.2, 3.3, 4.4, 5.5};

    for(int i=0; i<5; i++) {
        string num = std::to_string(data[i]);
        floatString.append(num);
        floatString.append(" ");
    }
    

    // cv::Mat mat = cv::Mat(1, 5, CV_32F, data);

    // unsigned int in_len = mat.total();
    // cout << mat << " " << in_len << endl;
    // const uchar* inBuffer = mat.data;
    string code = base64_encode(inBuffer, in_len).c_str();
    // cout << code << endl;

    // string dec_jpg =  base64_decode(code);
    // std::vector<uchar> data_char(dec_jpg.begin(), dec_jpg.end());
    // cv::Mat img = cv::imdecode(cv::Mat(data_char), 1);
    cout << floatString << endl;
    return 0;
}
