#pragma once 

#include <NvInfer.h>

#include <cassert>
#include <vector>

#include "../cuda/decode.h"

using namespace nvinfer1;

#define RETINAFACE_PLUGIN_NAME "RetinaFaceDecode"
#define RETINAFACE_PLUGIN_VERSION "1"
#define RETINAFACE_PLUGIN_NAMESPACE ""

namespace retinaface {

class DecodePlugin : public IPluginV2DynamicExt {
    float _score_thresh;
    int _top_n;
    std::vector<float> _anchors;
    float _resize;
    int _height;
    int _width;
    int _num_anchors;

    mutable int size = -1;

protected:
    void deserialize(void const* data, size_t length) {
        const char* d = static_cast<const char*>(data);
        read(d, _score_thresh);
        read(d, _top_n);
        int anchors_size;
        read(d, anchors_size);
        while( anchors_size-- ) {
            float val;
            read(d, val);
            _anchors.push_back(val);
        }
        read(d, _resize);
        read(d, _height);
        read(d, _width);
        read(d, _num_anchors);
    }

    size_t getSerializationSize() const override {
        return sizeof(_score_thresh) + sizeof(_top_n)
            + sizeof(size_t) + sizeof(float) * _anchors.size() + sizeof(_resize)
            + sizeof(_height) + sizeof(_width) + sizeof(_num_anchors);
    }

    void serialize(void *buffer) const override {
        char* d = static_cast<char*>(buffer);
        write(d, _score_thresh);
        write(d, _top_n);
        write(d, _anchors.size());
        for( auto &val : _anchors ) {
            write(d, val);
        }
        write(d, _resize);
        write(d, _height);
        write(d, _width);
        write(d, _num_anchors);
    }

public:
    DecodePlugin(float score_thresh, int top_n, std::vector<float> const& anchors, int resize,
        int height, int width, int num_anchors)
        : _score_thresh(score_thresh), _top_n(top_n), _anchors(anchors), _resize(resize),
        _height(height), _width(width), _num_anchors(num_anchors) {}

// Sử dụng khi load engine
    DecodePlugin(void const* data, size_t length) {
        this->deserialize(data, length);
    }

    const char *getPluginType() const override {
        return RETINAFACE_PLUGIN_NAME;
    }
 
    const char *getPluginVersion() const override {
        return RETINAFACE_PLUGIN_VERSION;
    }
  
    int getNbOutputs() const override {
        return 3;
    }

    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs *inputs,
        int nbInputs, IExprBuilder &exprBuilder) override 
    {
        DimsExprs output(inputs[0]);
        if (outputIndex == 1) {
            output.d[1] = exprBuilder.constant(_top_n * 4);
        } else if (outputIndex == 2) {
            output.d[1] = exprBuilder.constant(_top_n * 10);
        } else {
            output.d[1] = exprBuilder.constant(_top_n);
        }
        output.d[2] = exprBuilder.constant(1);
        output.d[3] = exprBuilder.constant(1);

        return output;
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, 
        int nbInputs, int nbOutputs) override
    {
        assert(nbInputs == 3);
        assert(nbOutputs == 3);
        assert(pos < 6);
        return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR;
    }

    int initialize() override { return 0; }

    void terminate() override {}

    size_t getWorkspaceSize(const PluginTensorDesc *inputs, 
        int nbInputs, const PluginTensorDesc *outputs, int nbOutputs) const override 
    {
        if (size < 0) {
        size = cuda::decode(inputs->dims.d[0], nullptr, nullptr,
            _num_anchors, _anchors, _width, _height, _resize, _score_thresh, _top_n, 
            nullptr, 0, nullptr);
        }
        return size;
    }

    int enqueue(const PluginTensorDesc *inputDesc, 
        const PluginTensorDesc *outputDesc, const void *const *inputs, 
        void *const *outputs, void *workspace, cudaStream_t stream)  
    {
        
        return cuda::decode(inputDesc->dims.d[0], inputs, outputs,
        _num_anchors, _anchors, _width, _height, _resize, _score_thresh, _top_n,
        workspace, getWorkspaceSize(inputDesc, 3, outputDesc, 3), stream);
        
    }

    void destroy() override {
        delete this;
    };

    const char *getPluginNamespace() const override {
        return RETINAFACE_PLUGIN_NAMESPACE;
    }
  
    void setPluginNamespace(const char *N) override {}

    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const
    {
        assert(index < 3);
        return DataType::kFLOAT;
    }

    void configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs, 
        const DynamicPluginTensorDesc *out, int nbOutputs)
    {
        assert(nbInputs == 3);
        assert(nbOutputs == 3);
    }

    IPluginV2DynamicExt *clone() const  {
        return new DecodePlugin(_score_thresh, _top_n, _anchors, _resize, _height, _width, 
        _num_anchors);
    }
    
private:
    // đẩy data từ val vào buffer --- Trong khi save model
    template<typename T> void write(char*& buffer, const T& val) const {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

// đẩy data từ buffer vào val --- Trong khi read model
    template<typename T> void read(const char*& buffer, T& val) {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
};

class DecodePluginCreator : public IPluginCreator {
public:
    DecodePluginCreator() {}

    const char *getPluginName () const override {
        return RETINAFACE_PLUGIN_NAME;
    }

    const char *getPluginVersion () const override {
        return RETINAFACE_PLUGIN_VERSION;
    }
    
    const char *getPluginNamespace() const override {
        return RETINAFACE_PLUGIN_NAMESPACE;
    }

    
    IPluginV2DynamicExt *deserializePlugin (const char *name, const void *serialData, size_t serialLength) override {
        return new DecodePlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char *N) override {}
    const PluginFieldCollection *getFieldNames() override { return nullptr; }
    IPluginV2DynamicExt *createPlugin (const char *name, const PluginFieldCollection *fc) override { return nullptr; }
};

REGISTER_TENSORRT_PLUGIN(DecodePluginCreator);

}

#undef RETINAFACE_PLUGIN_NAME
#undef RETINAFACE_PLUGIN_VERSION
#undef RETINAFACE_PLUGIN_NAMESPACE