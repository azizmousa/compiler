#ifndef OBJECT_DETECTOR_
#define OBJECT_DETECTOR_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "Loader.h"
#include "compiler/Blob.h"

class Detector{
private:

    static long objectsCounter, filesCounter;

    float confThreshold; // Confidence threshold
    float nmsThreshold;  // Non-maximum suppression threshold
    int inpWidth;  // Width of network's input image
    int inpHeight; // Height of network's input image
    
    std::string tempPath, objectOutputFile, objectsOutputDir;

    std::vector<cv::dnn::Net> loadedNets;
    std::vector<std::vector<std::string>> classes;

    //detect object in the image using specific weight 
    void detectObjects(const int netIndex, const std::string imagePath, const int classesIndex, bool &found, Blob objectBox);

    // Remove the bounding boxes with low confidence using non-maxima suppression
    void postprocess(cv::Mat& frame, std::vector<cv::Mat>& outs, const int classesIndex, bool &foundFlage, Blob objectBox);

    // Draw the predicted bounding box
    void drawPred(const int classId, const float conf, int left, int top, int right, 
    int bottom, cv::Mat& frame, const int classesIndex);

    // Get the names of the output layers
    std::vector<std::string> getOutputsNames(const cv::dnn::Net& net);


public:
    Detector(std::vector<cv::dnn::Net> nets, std::vector<std::vector<std::string>> classes, 
    std::string tepmPaht, float confThreshold = 0.30, float nmsThreshold = 0.04, 
    int inpWidth = 416, int inpHeight = 416);
    ~Detector();

    void startDetection(const std::string image, const Blob blob, bool &foundFlage);

    std::string getObjectsOutputFile()const;
    void setObjcetsOutputFile(std::string file);
    std::string getObjectsOutputDir()const;
    void setObjcetsOutputDir(std::string file);


};
#endif