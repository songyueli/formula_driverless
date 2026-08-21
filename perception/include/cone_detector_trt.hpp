#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

// TensorRT-backed replacement for ConeDetector (see cone_detector.hpp), used
// on the Jetson build only (see CMakeLists.txt's PERCEPTION_USE_TENSORRT
// guard). Same public interface -- same Detection struct, same Detect()
// signature, same letterbox pre/post-processing -- so perception.cpp only
// needs to swap which class it instantiates, not how it's used.
//
// Loads a PRE-BUILT .engine file (produced once, offline, via trtexec).
// Unlike best.onnx, a .engine is tied to the exact GPU + TensorRT version +
// precision it was built with -- not portable even across two Jetsons on
// different TensorRT versions.
class ConeDetectorTrt
{
public:
    struct Detection
    {
        float x1, y1, x2, y2;
        float confidence;
        int classId;
    };

    explicit ConeDetectorTrt(const std::string &enginePath, float confThreshold = 0.25f);
    ~ConeDetectorTrt();

    ConeDetectorTrt(const ConeDetectorTrt &) = delete;
    ConeDetectorTrt &operator=(const ConeDetectorTrt &) = delete;

    std::vector<Detection> Detect(const cv::Mat &image);

private:
    class Logger : public nvinfer1::ILogger
    {
        void log(Severity severity, const char *msg) noexcept override;
    };

    Logger m_logger;
    nvinfer1::IRuntime *m_runtime = nullptr;
    nvinfer1::ICudaEngine *m_engine = nullptr;
    nvinfer1::IExecutionContext *m_context = nullptr;
    cudaStream_t m_stream = nullptr;

    std::string m_inputName;
    std::string m_outputName;
    int64_t m_inputWidth = 0;
    int64_t m_inputHeight = 0;
    int64_t m_numDetections = 0;
    int64_t m_stride = 0;

    void *m_deviceInput = nullptr;
    void *m_deviceOutput = nullptr;
    std::vector<float> m_hostOutput;

    float m_confThreshold;
};
