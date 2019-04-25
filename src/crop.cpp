
#include<compiler/crop.h>

#include<algorithm>
#include<vector>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<compiler/Blob.h>
#include<iostream>

// global variables ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
// const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
// const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
// const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);


bool maxArea(Blob b1, Blob b2){
    return b1.boundingRect.area() > b2.boundingRect.area();
}


void CropImage::cropImageFrame(){

    //  cv::Mat imgFrame1 = this->image.clone;
    // char chCheckForEscKey = 0;
    std::vector<Blob> blobs;
    cv::Mat imgFrame1Copy = this->image.clone();
    cv::Mat imgDifference;
    cv::Mat imgThresh;
    
    cv::cvtColor(imgFrame1Copy, imgFrame1Copy, cv::COLOR_BGR2GRAY);
    
    cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
    
    double min, max;
    cv::Point minp, maxp;
    cv::minMaxLoc(imgFrame1Copy, &min, &max, &minp, &maxp);
    
    cv::adaptiveThreshold(imgFrame1Copy, imgThresh, 130 , cv::ADAPTIVE_THRESH_GAUSSIAN_C,
     cv::THRESH_BINARY_INV, 11, 5);

    // cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    // cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat structuringElement9x9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));

    cv::dilate(imgThresh, imgThresh, structuringElement5x5);
    cv::dilate(imgThresh, imgThresh, structuringElement5x5);
    cv::erode(imgThresh, imgThresh, structuringElement5x5);

    
    // cv::imshow("imgThresh", imgThresh);
    cv::Mat imgThreshCopy = imgThresh.clone();

    std::vector<std::vector<cv::Point> > contours;

    cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    cv::Mat imgContours(imgThresh.size(), imgThresh.type(), SCALAR_BLACK);

    cv::drawContours(imgContours, contours, -1, SCALAR_WHITE, -1);

    for(int i =0; i<this->erodContoursNumber;++i)
        cv::erode(imgContours, imgContours, structuringElement9x9);
    
    cv::findContours(imgContours, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_L1);
    
    //  cv::imshow("imgContours", imgContours);

    std::vector<std::vector<cv::Point> > convexHulls(contours.size());

    for (unsigned int i = 0; i < contours.size(); i++) {
        cv::convexHull(contours[i], convexHulls[i]);
    }

    for (auto &convexHull : convexHulls) {
        Blob possibleBlob(convexHull);
        // std::cout << convexHull << std::endl;
        if (possibleBlob.boundingRect.area() > 100 &&
            possibleBlob.boundingRect.width > 15 &&
            possibleBlob.boundingRect.height > 20) {
                // possibleBlob.boundingRect.x -= 4;
                // possibleBlob.boundingRect.y -= 4;
                // possibleBlob.boundingRect.width += 4;
                // possibleBlob.boundingRect.height += 4;
            blobs.push_back(possibleBlob);
        }
    }

    cv::Mat imgConvexHulls(imgThresh.size(), CV_8UC3, SCALAR_BLACK);

    convexHulls.clear();

    for (auto &blob : blobs) {
        convexHulls.push_back(blob.contour);
    }

    cv::drawContours(imgConvexHulls, convexHulls, -1, SCALAR_WHITE, -1);

    // cv::imshow("imgConvexHulls", imgConvexHulls);
    // cv::waitKey();
    std::sort(blobs.begin(), blobs.end(), maxArea);
// std::cerr << "blobsize = " << blobs.size() << std::endl;
    if(blobs.size()>0)
        this->image = image(blobs[0].boundingRect);
    else{
        std::cerr << std::endl << "no detected objects ! ."<< std::endl;
        exit(1);
    }
// std::cerr << "starting detection operation ..." << std::endl;
    this->objectsCoordinates = blobs;

    // for (auto &blob : blobs) {                                                  // for each blob
    //     cv::rectangle(imgFrame1Copy, blob.boundingRect, SCALAR_RED, 2);             // draw a red box around the blob
    //     cv::circle(imgFrame1Copy, blob.centerPosition, 3, SCALAR_GREEN, -1);        // draw a filled-in green circle at the center
    // }

    // cv::imshow("imgFrame1Copy", imgFrame1Copy);
}

CropImage::CropImage(cv::Mat image, int erodContoursNumber){
    this->image = image;
    this->erodContoursNumber = erodContoursNumber;
    this->cropImageFrame();
}

CropImage::~CropImage(){
    if(!this->image.empty())
        this->image.release();
}

void CropImage::setSourceImage(cv::Mat image){
    this->image = image;
    this->cropImageFrame();
}

cv::Mat CropImage::getCropedImage(){
    return this->image;
}

void CropImage::setErodContoursNumber(int n){
     this->erodContoursNumber = n;
}
int CropImage::getErodContoursNumber()const{
    return this->erodContoursNumber;
}

void CropImage::getOjectsCoordinates(std::vector<Blob> &vec)const{
    vec = this->objectsCoordinates;
}