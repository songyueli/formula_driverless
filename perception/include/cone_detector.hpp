#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

// Runs the trained YOLO cone-detection model (exported to ONNX by
// ml/train.py) via ONNX Runtime's CPU execution provider.
//
// Input: fed the stitched cylindrical panorama directly, RGB, whatever size
// it happens to be. Cone shapes are already consistent across that
// panorama's width (that's the whole reason it's cylindrical, not planar --
// see camera_stitcher.hpp), so this doesn't need to run per-camera.
//
// Preprocessing uses a LETTERBOX resize (scale to fit inside the model's
// square input, pad the rest with gray) rather than a plain stretch -- the
// panorama's aspect ratio is far from square (roughly 2.15:1), and a
// stretch resize would squash cone shapes right back into the kind of
// distortion the cylindrical projection was specifically chosen to avoid.
// Detection coordinates are un-letterboxed back into the input image's own
// pixel space before being returned, so callers never see model-input-space
// coordinates.
class ConeDetector
{
public:
    struct Detection
    {
        float x1, y1, x2, y2; // pixel coords in the INPUT image passed to Detect()
        float confidence;
        int classId;
    };

    // modelPath: path to the exported .onnx file.
    // confThreshold: detections below this confidence are dropped.
    explicit ConeDetector(const std::string &modelPath, float confThreshold = 0.25f);

    // image must be CV_8UC3, RGB (matches what the sim's cameras and
    // CameraStitcher::Stitch() both already produce -- no BGR conversion
    // needed anywhere in this pipeline).
    std::vector<Detection> Detect(const cv::Mat &image);

private:
    Ort::Env m_env;
    Ort::Session m_session;
    Ort::MemoryInfo m_memoryInfo;
    std::string m_inputName;
    std::string m_outputName;
    int64_t m_inputWidth;
    int64_t m_inputHeight;
    float m_confThreshold;
};
