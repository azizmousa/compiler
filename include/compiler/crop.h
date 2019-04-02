#ifndef _CROP_IMAGE_
#define _CROP_IMAGE_
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<compiler/Blob.h>
#include<vector>

class CropImage{
private:
    int erodContoursNumber;
    cv::Mat image;
    cv::Point topLef;
    cv::Point bottomRight;
    void cropImageFrame();
    std::vector<Blob> objectsCoordinates;

public:
    CropImage(cv::Mat image, int erodContoursNumber);
    ~CropImage();
    void setSourceImage(cv::Mat image);
    cv::Mat getCropedImage();
    int getErodContoursNumber()const;
    void getOjectsCoordinates(std::vector<Blob> &vec)const;
    void setErodContoursNumber(int n);
   
};

#endif