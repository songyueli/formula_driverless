#include "cone_detector_trt.hpp"

#include <NvInferRuntime.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

void ConeDetectorTrt::Logger::log(Severity severity, const char *msg) noexcept
{
    // Same threshold as cone_detector.cpp's ORT_LOGGING_LEVEL_WARNING
    // sibling -- skip TensorRT's very chatty kINFO/kVERBOSE build-time
    // diagnostics.
    if (severity <= Severity::kWARNING)
    {
        std::cerr << "[TensorRT] " << msg << '\n';
    }
}

ConeDetectorTrt::ConeDetectorTrt(const std::string &enginePath, float confThreshold)
    : m_confThreshold(confThreshold)
{
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("could not open engine file: " + enginePath);
    }
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(static_cast<size_t>(size));
    if (!file.read(engineData.data(), size))
    {
        throw std::runtime_error("failed to read engine file: " + enginePath);
    }

    m_runtime = nvinfer1::createInferRuntime(m_logger);
    m_engine = m_runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!m_engine)
    {
        throw std::runtime_error("deserializeCudaEngine failed for: " + enginePath);
    }
    m_context = m_engine->createExecutionContext();

    // Exactly 2 IO tensors expected: one input, one output -- matches this
    // model's export (same assumption cone_detector.cpp makes via
    // GetInputNameAllocated(0)/GetOutputNameAllocated(0)).
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i)
    {
        const char *name = m_engine->getIOTensorName(i);
        if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            m_inputName = name;
            const nvinfer1::Dims dims = m_engine->getTensorShape(name);
            m_inputHeight = dims.d[2];
            m_inputWidth = dims.d[3];
        }
        else
        {
            m_outputName = name;
            const nvinfer1::Dims dims = m_engine->getTensorShape(name);
            m_numDetections = dims.d[1];
            m_stride = dims.d[2];
        }
    }
    if (m_inputName.empty() || m_outputName.empty())
    {
        throw std::runtime_error("engine did not expose exactly one input and one output tensor");
    }

    cudaStreamCreate(&m_stream);
    cudaMalloc(&m_deviceInput, static_cast<size_t>(3 * m_inputWidth * m_inputHeight) * sizeof(float));
    cudaMalloc(&m_deviceOutput, static_cast<size_t>(m_numDetections * m_stride) * sizeof(float));
    m_hostOutput.resize(static_cast<size_t>(m_numDetections * m_stride));

    m_context->setTensorAddress(m_inputName.c_str(), m_deviceInput);
    m_context->setTensorAddress(m_outputName.c_str(), m_deviceOutput);

    std::cout << "[cone_detector_trt] loaded " << enginePath << " -- input '" << m_inputName
               << "' " << m_inputWidth << "x" << m_inputHeight << ", output '" << m_outputName
               << "' [" << m_numDetections << "x" << m_stride << "], conf threshold "
               << m_confThreshold << '\n';
}

ConeDetectorTrt::~ConeDetectorTrt()
{
    if (m_deviceInput) cudaFree(m_deviceInput);
    if (m_deviceOutput) cudaFree(m_deviceOutput);
    if (m_stream) cudaStreamDestroy(m_stream);
    delete m_context;
    delete m_engine;
    delete m_runtime;
}

std::vector<ConeDetectorTrt::Detection> ConeDetectorTrt::Detect(const cv::Mat &image)
{
    // Identical letterbox pre/post-processing to ConeDetector::Detect() (see
    // cone_detector.cpp) -- backend-agnostic, deliberately kept as a
    // duplicate here rather than shared via a common helper: a single-file
    // duplicate is simpler to keep correct than threading it through both
    // classes' otherwise-unrelated bodies right now.
    const double scale = std::min(static_cast<double>(m_inputWidth) / image.cols,
                                    static_cast<double>(m_inputHeight) / image.rows);
    const int newW = static_cast<int>(std::round(image.cols * scale));
    const int newH = static_cast<int>(std::round(image.rows * scale));
    const int padLeft = static_cast<int>((m_inputWidth - newW) / 2);
    const int padTop = static_cast<int>((m_inputHeight - newH) / 2);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, newH));
    cv::Mat canvas(static_cast<int>(m_inputHeight), static_cast<int>(m_inputWidth), CV_8UC3,
                   cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(padLeft, padTop, newW, newH)));

    std::vector<cv::Mat> channels(3);
    cv::split(canvas, channels);
    std::vector<float> blob(static_cast<size_t>(3 * m_inputWidth * m_inputHeight));
    const size_t planeSize = static_cast<size_t>(m_inputWidth) * static_cast<size_t>(m_inputHeight);
    for (int c = 0; c < 3; ++c)
    {
        cv::Mat plane(static_cast<int>(m_inputHeight), static_cast<int>(m_inputWidth),
                      CV_32FC1, blob.data() + static_cast<size_t>(c) * planeSize);
        channels[static_cast<size_t>(c)].convertTo(plane, CV_32FC1, 1.0 / 255.0);
    }

    cudaMemcpyAsync(m_deviceInput, blob.data(), blob.size() * sizeof(float),
                     cudaMemcpyHostToDevice, m_stream);
    m_context->enqueueV3(m_stream);
    cudaMemcpyAsync(m_hostOutput.data(), m_deviceOutput, m_hostOutput.size() * sizeof(float),
                     cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    std::vector<Detection> detections;
    for (int64_t i = 0; i < m_numDetections; ++i)
    {
        const float *row = m_hostOutput.data() + i * m_stride;
        const float confidence = row[4];
        if (confidence < m_confThreshold)
        {
            continue;
        }
        Detection d;
        d.x1 = static_cast<float>((row[0] - padLeft) / scale);
        d.y1 = static_cast<float>((row[1] - padTop) / scale);
        d.x2 = static_cast<float>((row[2] - padLeft) / scale);
        d.y2 = static_cast<float>((row[3] - padTop) / scale);
        d.confidence = confidence;
        d.classId = static_cast<int>(row[5]);
        detections.push_back(d);
    }
    return detections;
}
