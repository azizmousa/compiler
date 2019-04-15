
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include "compiler/Detector.h"
#include "compiler/Files.h"


long Detector::objectsCounter = 0, Detector::filesCounter = 0;

Detector::Detector(std::vector<cv::dnn::Net> nets, std::vector<std::vector<std::string>> classes, 
    std::string tempPath, float confThreshold, float nmsThreshold, 
    int inpWidth, int inpHeight){
        
    this->loadedNets = nets;
    this->classes = classes;
    this->tempPath = tempPath;
    this->confThreshold = confThreshold;
    this->nmsThreshold = nmsThreshold;
    this->inpWidth = inpWidth;
    this->inpHeight = inpHeight;
}

Detector::~Detector(){}

void Detector::detectObjects(int netIndex, const std::string imagePath, int classesIndex, bool &found, Blob objectBox){
    // Open image file.
    std::string str;//, outputFile;
    // cv::VideoCapture cap;
    
    cv::Mat frame, blob;
    // Open the image file
    str = imagePath;
    std::ifstream ifile(str);
    // TODO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> CREATE SPECIFIC EXCEPTION
    if (!ifile) throw("error");
    frame = cv::imread(str);
    // cap.open(str);
    //str.replace(str.end()-4, str.end());
    // outputFile =  this->tempPath + "/" + OBJECTS_OUTPUT_FILE+std::to_string(cn++)+".jpg";
    

    // cap >> frame;

    // Create a 4D blob from a frame.
    cv::dnn::blobFromImage(frame, blob, 1/255.0, cvSize(this->inpWidth, this->inpHeight), cv::Scalar(0,0,0), true, false);

    //Sets the input to the network
    this->loadedNets[netIndex].setInput(blob);
    
    
    // Runs the forward pass to get output of the output layers
    std::vector<cv::Mat> outs;
    this->loadedNets[netIndex].forward(outs, this->getOutputsNames(this->loadedNets[netIndex]));
    
    // Remove the bounding boxes with low confidence
    this->postprocess(frame, outs, classesIndex, found, objectBox);
    
    // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = loadedNets[netIndex].getPerfProfile(layersTimes) / freq;
    std::string label = cv::format("Inference time for a frame : %.2f ms", t);
    cv::putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
    
    // cap.release();
}

void Detector::postprocess(cv::Mat& frame, std::vector<cv::Mat>& outs, const int classesIndex, bool &foundFlage, Blob objectBox){
   
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i){
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols){
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > this->confThreshold){
                foundFlage = true;
            	// std::cout<<"confidence: "<<confidence<<"\t class: "<<classIdPoint<<endl;
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
    
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        if(objectBox.boundingRect.area() > 0){
            this->drawPred(classIds[idx], confidences[idx], objectBox.boundingRect.x, objectBox.boundingRect.y,
                 objectBox.boundingRect.x + objectBox.boundingRect.width, objectBox.boundingRect.y + objectBox.boundingRect.height,
                  frame, classesIndex);
        }else{
            this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame, classesIndex);
        }
       
    }
}


void Detector::drawPred(const int classId, const float conf, int left, int top, int right, 
int bottom, cv::Mat& frame, const int classesIndex){
    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);

    std::ofstream file;
    file.open(Detector::objectOutputFile + ".viw", std::ios::app);
    file << this->classes[classesIndex][classId] << std::endl;
    file << this->classes[classesIndex][classId] << "_" << Detector::objectsCounter++<<std::endl;
    file << left << std::endl;
    file << top << std::endl;
    file << right << std::endl;
    file << bottom << std::endl;
    //file << conf <<std::endl;
    file << "-----------------------------------" << std::endl;
    file.close();
    
    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if (!this->classes[classesIndex].empty())
    {
        CV_Assert(classId < (int)this->classes[classesIndex].size());
        label = this->classes[classesIndex][classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = std::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - std::round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), 
    top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0),1);
    std::string outputFile =  this->tempPath + "/" + this->objectOutputFile + std::to_string(Detector::filesCounter++) + ".jpg";
    // Write the frame with the detection boxes
    cv::Mat detectedFrame;
    frame.convertTo(detectedFrame, CV_8U);
    cv::imwrite(outputFile, detectedFrame);
}

std::vector<std::string> Detector::getOutputsNames(const cv::dnn::Net& net){
    static std::vector<std::string> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        std::vector<std::string> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

void Detector::startDetection(const std::string image, const Blob blob, bool &foundFlage){
    for(size_t i =0; i < this->loadedNets.size(); ++i){
        this->detectObjects(i, image, i, foundFlage, blob);
    }
}

std::string Detector::getObjectsOutputFile()const{
    return this->objectOutputFile;
}
void Detector::setObjcetsOutputFile(std::string file){
    this->objectOutputFile = Files::getFileName(file);
}
