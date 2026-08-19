#include "cone_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

ConeDetector::ConeDetector(const std::string &modelPath, float confThreshold)
    : m_env(ORT_LOGGING_LEVEL_WARNING, "cone_detector")
    , m_session(m_env, modelPath.c_str(), Ort::SessionOptions())
    , m_memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    , m_confThreshold(confThreshold)
{
    Ort::AllocatorWithDefaultOptions allocator;

    auto inputNameAlloc = m_session.GetInputNameAllocated(0, allocator);
    m_inputName = inputNameAlloc.get();
    auto outputNameAlloc = m_session.GetOutputNameAllocated(0, allocator);
    m_outputName = outputNameAlloc.get();

    // Input shape is NCHW: [1, 3, H, W].
    const auto inputShape = m_session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    m_inputHeight = inputShape[2];
    m_inputWidth = inputShape[3];

    std::cout << "[cone_detector] loaded " << modelPath << " -- input '" << m_inputName
               << "' " << m_inputWidth << "x" << m_inputHeight << ", output '" << m_outputName
               << "', conf threshold " << m_confThreshold << '\n';
}

std::vector<ConeDetector::Detection> ConeDetector::Detect(const cv::Mat &image)
{
    // Letterbox: scale to fit inside the model's square input, preserving
    // aspect ratio, padding the remainder with gray (114,114,114) --
    // matches ultralytics' own training-time preprocessing. A plain stretch
    // resize would re-distort cone shapes, undoing the reason this pipeline
    // uses a cylindrical (not planar) panorama in the first place.
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

    // HWC uint8 -> CHW float32 in [0,1] (standard YOLO input, no mean/std
    // normalization).
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

    const std::array<int64_t, 4> inputShape{1, 3, m_inputHeight, m_inputWidth};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        m_memoryInfo, blob.data(), blob.size(), inputShape.data(), inputShape.size());

    const char *inputNames[] = {m_inputName.c_str()};
    const char *outputNames[] = {m_outputName.c_str()};
    auto outputTensors = m_session.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1,
                                         outputNames, 1);

    const float *outData = outputTensors[0].GetTensorData<float>();
    const auto outShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    // outShape is [1, numDetections, 6]: x1, y1, x2, y2, confidence, classId,
    // in MODEL-INPUT pixel space (640x640-ish, not the original image) --
    // already NMS'd (confirmed via this exact model's own output shape when
    // first loaded and tested, back during the Jetson training work).
    const int64_t numDetections = outShape[1];
    const int64_t stride = outShape[2];

    std::vector<Detection> detections;
    for (int64_t i = 0; i < numDetections; ++i)
    {
        const float *row = outData + i * stride;
        const float confidence = row[4];
        if (confidence < m_confThreshold)
        {
            continue;
        }
        // Un-letterbox back into the caller's own image pixel space.
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
