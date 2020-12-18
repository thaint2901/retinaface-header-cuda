#include <vector>
#include <iostream>
#include <fstream>

#include "../csrc/engine.h"

using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " core_model.onnx engine.plan" << endl;
    }

    ifstream onnxFile;
	onnxFile.open(argv[1], ios::in | ios::binary);
    cout << "Load model from " << argv[1] << endl;

    if (!onnxFile.good()) {
		cerr << "\nERROR: Unable to read specified ONNX model " << argv[1] << endl;
		return -1;
	}

    onnxFile.seekg (0, onnxFile.end);
	size_t size = onnxFile.tellg();
	onnxFile.seekg (0, onnxFile.beg);

	auto *buffer = new char[size];
	onnxFile.read(buffer, size);
	onnxFile.close();

    bool verbose = false;
	size_t workspace_size =(1ULL << 30);
    const vector<int> dynamic_batch_opts{1, 8, 16};
    float score_thresh = 0.2f;
    int top_n = 250;
    float nms_thresh = 0.4;
    float resize = 0.5;
    int detections_per_im = 50;
    vector<vector<float>> anchors;
    anchors = {{10.0, 20.0},
            {32.0, 64.0},
            {128.0, 256.0}};
    
    cout << "Building engine..." << endl;
	auto engine = retinaface::Engine(buffer, size, dynamic_batch_opts, score_thresh, resize, top_n,
		anchors, nms_thresh, detections_per_im, verbose, workspace_size);
	engine.save(string(argv[2]));

	delete [] buffer;

	return 0;
}