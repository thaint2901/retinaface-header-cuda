#include "engine.h"
#include "plugins/DecodePlugin.h"
#include "plugins/NMSPlugin.h"

#include <iostream>
#include <fstream>

#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>

using namespace nvinfer1;
using namespace nvonnxparser;

namespace retinaface {

class Logger : public ILogger {
public:
    Logger(bool verbose)
        : _verbose(verbose) {
    }

    void log(Severity severity, const char *msg) override {
        if (_verbose || (severity != Severity::kINFO) && (severity != Severity::kVERBOSE))
            cout << msg << endl;
    }

private:
   bool _verbose{false};
};

void Engine::_load(const string &path) {
    ifstream file(path, ios::in | ios::binary);
    file.seekg (0, file.end);
    size_t size = file.tellg();
    file.seekg (0, file.beg);

    char *buffer = new char[size];
    file.read(buffer, size);
    file.close();

    _engine = _runtime->deserializeCudaEngine(buffer, size, nullptr);

    delete[] buffer;
}

void Engine::_prepare() {
    _context = _engine->createExecutionContext();
    _context->setOptimizationProfile(0);
    cudaStreamCreate(&_stream);
}

Engine::Engine(const string &engine_path, bool verbose) {
    Logger logger(verbose);
    _runtime = createInferRuntime(logger);
    _load(engine_path);
    _prepare();
}

Engine::~Engine() {
    if (_stream) cudaStreamDestroy(_stream);
    if (_context) _context->destroy();
    if (_engine) _engine->destroy();
    if (_runtime) _runtime->destroy();
}

Engine::Engine(const char *onnx_model, size_t onnx_size, const vector<int>& dynamic_batch_opts,
    float score_thresh, float resize, const vector<int>&steps, int top_n, const vector<vector<float>>&anchors,
    float nms_thresh, int detections_per_im,
    bool verbose, size_t workspace_size) {

    Logger logger(verbose);
    _runtime = createInferRuntime(logger);

    // Create builder
    auto builder = createInferBuilder(logger);
    const auto builderConfig = builder->createBuilderConfig();
    builderConfig->setFlag(BuilderFlag::kFP16);
    builderConfig->setMaxWorkspaceSize(workspace_size);

    // Parse ONNX
    const auto flags = 1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(flags);
    auto parser = createParser(*network, logger);
    parser->parse(onnx_model, onnx_size);

    auto input = network->getInput(0);
    auto inputDims = input->getDimensions();
    auto profile = builder->createOptimizationProfile();
    auto inputName = input->getName();
    auto profileDimsmin = Dims4{dynamic_batch_opts[0], inputDims.d[1], inputDims.d[2], inputDims.d[3]};
    auto profileDimsopt = Dims4{dynamic_batch_opts[1], inputDims.d[1], inputDims.d[2], inputDims.d[3]};
    auto profileDimsmax = Dims4{dynamic_batch_opts[2], inputDims.d[1], inputDims.d[2], inputDims.d[3]};

    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, profileDimsmin);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, profileDimsopt);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, profileDimsmax);

    if(profile->isValid())
        builderConfig->addOptimizationProfile(profile);
    
    // Add decode plugins
    cout << "Building accelerated plugins..." << endl;
    vector<DecodePlugin> decodePlugins;
    vector<ITensor *> scores, boxes, landms;
    auto nbOutputs = network->getNbOutputs();
    int height = inputDims.d[2];
    int width = inputDims.d[3];

    for (int i = 0; i < nbOutputs / 3; i++) {
        auto boxOutput = network->getOutput(i);
        auto scoreOutput = network->getOutput(nbOutputs / 3 + i);
        auto landmOutput = network->getOutput(2 * nbOutputs / 3 + i);
        auto decodePlugin = DecodePlugin(score_thresh, top_n, anchors[i], resize, steps[i], height, width);
        decodePlugins.push_back(decodePlugin); 
        vector<ITensor *> inputs = {scoreOutput, boxOutput, landmOutput};
        auto layer_decode = network->addPluginV2(inputs.data(), inputs.size(), decodePlugin);

        scores.push_back(layer_decode->getOutput(0));
        boxes.push_back(layer_decode->getOutput(1));
        landms.push_back(layer_decode->getOutput(2));
    }

    // Cleanup outputs
    for (int i = 0; i < nbOutputs; i++) {
        auto output = network->getOutput(0);
        network->unmarkOutput(*output);
    }

    // Concat tensors from each feature map
    vector<ITensor *> concat;
    for (auto tensors : {scores, boxes, landms}) {
        auto layer = network->addConcatenation(tensors.data(), tensors.size());
        concat.push_back(layer->getOutput(0));
    }

    // Add NMS plugin
    size_t count = top_n;
    auto nmsPlugin = NMSPlugin(nms_thresh, detections_per_im);
    auto layer_nms = network->addPluginV2(concat.data(), concat.size(), nmsPlugin);

    vector<string> names = {"scores", "boxes", "landms"};
    for (int i = 0; i < 3; i++) {
        // auto output = concat[i];
        auto output = layer_nms->getOutput(i);
        network->markOutput(*output);
        output->setName(names[i].c_str());
    }

    // Build engine
    cout << "Applying optimizations and building TRT CUDA engine..." << endl;
    _engine = builder->buildEngineWithConfig(*network, *builderConfig);

    int numCreators = 0;
    nvinfer1::IPluginCreator* const* tmpList = getPluginRegistry()->getPluginCreatorList(&numCreators);

    for (int k = 0; k < numCreators; ++k)
    {
        if (!tmpList[k])
        {
            std::cout << "Plugin Creator for plugin " << k << " is a nullptr." << std::endl;
            continue;
        }
        std::string pluginName = tmpList[k]->getPluginName();
        std::cout << k << ": " << pluginName << std::endl;
    }

    // Housekeeping
    parser->destroy();
    network->destroy();
    builderConfig->destroy();
    builder->destroy();

    _prepare();
}

void Engine::save(const string &path) {
    cout << "Writing to " << path << "..." << endl;
    auto serialized = _engine->serialize();
    ofstream file(path, ios::out | ios::binary);
    file.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());

    serialized->destroy();    
}

void Engine::infer(vector<void *> &buffers, int batch){
    auto dims = _engine->getBindingDimensions(0);
    _context->setBindingDimensions(0, Dims4(batch, dims.d[1], dims.d[2], dims.d[3]));
    _context->enqueueV2(buffers.data(), _stream, nullptr);
    cudaStreamSynchronize(_stream);
}

vector<int> Engine::getInputSize() {
    auto dims = _engine->getBindingDimensions(0);
    return {dims.d[2], dims.d[3]};
}

int Engine::getMaxBatchSize() {
    return _engine->getMaxBatchSize();
}

int Engine::getMaxDetections() {
    return _engine->getBindingDimensions(1).d[1];
}

}