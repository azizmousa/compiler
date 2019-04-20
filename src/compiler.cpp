
#include <iostream>
#include <sstream>
#include <vector>
#include <exception>
#include <string>
#include <time.h>
#include <thread>

#include "compiler/Loader.h"
#include "compiler/Detector.h"
#include "compiler/crop.h"
#include "compiler/Files.h"

std::string const EMPTY_IMAGE_PATH = "crop";
std::string const OUTPUP_VIW_FILES_DIR = "views";
std::string const TEMP_PATH = "temp";
std::string const IMAGE_EXTENTION = ".JPEG";

enum struct Flages{
    drawnFlage,
    paperFlage,
    clearTemp
};

void imageFileProcess(std::string imagePath, Flages imageFlage);

//the main function to classify the image
void runOperation(int end, Flages imageFlage, char *argv[]);

void executeThread(cv::Mat objectsImage, int width, int height, Blob box);

void clearTemp();

Loader *loader;
Detector *detector;

int main(int argc, char* argv[]){

    auto t1 = std::chrono::system_clock::now();
    if(argc < 3){
        std::cout << "you should pass the input path" << std::endl;
        return 1;
    }
    
    try{
        loader = new Loader("weights", "names", "configurations");
        detector = new Detector(loader->getAllNets(), loader->getAllClasses(), TEMP_PATH);
        Files::initializeOutputDirectory(EMPTY_IMAGE_PATH);
        Files::initializeOutputDirectory(OUTPUP_VIW_FILES_DIR);
        Files::initializeOutputDirectory(TEMP_PATH);
    }catch(std::exception &e){
        std::cout << e.what() << std::endl;
        return 1;
    }
    
    std::cout << argv[1] << std::endl;
    
    if(std::string(argv[1]) == "-d" || std::string(argv[1]) == "--drawn")
        runOperation(argc, Flages::drawnFlage, argv);
    else if(std::string(argv[1]) == "-p" || std::string(argv[1]) == "--paper")
        runOperation(argc, Flages::paperFlage, argv);
    else{
        std::cout << "invalid image type flage" << std::endl;
        delete loader;
        delete detector;
        return 1;
    }
        

    delete loader;
    delete detector;
    
    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = t2-t1;
    std::time_t end_time = std::chrono::system_clock::to_time_t(t2);

    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    return 0;
}


void runOperation(int end, Flages imageFlage, char *argv[]){
    for(int i =2; i<end; ++i){
        std::cerr << "itration: " <<i<< std::endl;
        if(Files::is_file(argv[i])){
            std::cerr << "file !!!!!!" << std::endl;
            imageFileProcess(argv[i], imageFlage);
        }else{
            std::cerr << "folder !!!!!!" << std::endl;
            std::vector<std::string> images;
            Files::getDirFiles(argv[i], "", images);
            std::cerr << "imageSize = " << images.size() << std::endl;
            for(size_t j = 0; j< images.size() ;++j){
                imageFileProcess(std::string(argv[i])+Files::slash()+images[j], imageFlage);
            }
        }
       
    }
}

void imageFileProcess(std::string imagePath, Flages imageFlage){
    std::cerr << std::endl << "imagePath: " << imagePath << std::endl;
    std::ifstream file(imagePath);
    if(file.good()){
        std::string image = imagePath;
        cv::Mat img = imread(image, cv::IMREAD_GRAYSCALE);
        // int fw, fh;
        // fw = img.cols;
        // fh = img.rows;
        // cv::resize(img, img, cv::Size(fw - (fw*50/100), fh - (fh*50/100)));
        cv::resize(img, img, cv::Size(600, 900));
        // std::cerr << img.cols << "  " << img.rows << std::endl;
        
        CropImage cImg(img, 4);
        
        
        img = cImg.getCropedImage().clone();
        int width = img.cols;
        int height = img.rows;
        std::cerr << width << " " << height << std::endl;
        std::string cropedPath = EMPTY_IMAGE_PATH + Files::slash() + Files::getFileName(image) + IMAGE_EXTENTION;
        cv::imwrite(cropedPath, img);
        img.release();

        detector->setObjcetsOutputFile(image);
        detector->setObjcetsOutputDir(OUTPUP_VIW_FILES_DIR);
        std::ofstream sFile;
        sFile.open(OUTPUP_VIW_FILES_DIR +Files::slash()+ detector->getObjectsOutputFile() + ".viw");
        sFile << width << std::endl;
        sFile << height << std::endl;
        sFile.close();
        bool foundFlage = false;
        Blob nullBlob;
        if(imageFlage == Flages::paperFlage){
            detector->startDetection(cropedPath, nullBlob, foundFlage);
        }else if(imageFlage == Flages::drawnFlage){
            
            cv::Mat objectsImage = cv::imread(cropedPath, cv::IMREAD_GRAYSCALE);
            CropImage cObjects(objectsImage, 0);
            std::vector<Blob> objectsBoxs;
            cObjects.getOjectsCoordinates(objectsBoxs);
            
            // std::thread *imgThreads = new std::thread[objectsBoxs.size()];

            for(size_t i =0; i<objectsBoxs.size();++i){
                // bool *pff = false;
                std::cerr << "starting thread number: " << i << std::endl;
                executeThread(objectsImage, width, height, objectsBoxs[i]);
                // *(imgThreads+i) = std::thread(executeThread, objectsImage, width, height, objectsBoxs[i]);
            }
            // for(size_t i =0; i<objectsBoxs.size();++i){
            //     (imgThreads+i)->join();
            // }
            // delete [] imgThreads;
            cv::imwrite(cropedPath, objectsImage);   
        }
    }else{
        std::cout << imagePath << " file not found." << std::endl;
    }
}

void executeThread(cv::Mat objectsImage, int width, int height, Blob box){
    
    cv::Mat emptyImage(height, width, CV_8UC1, cv::Scalar(255, 255, 255));
    std::cerr << box.boundingRect << std::endl;
    cv::Mat obj = objectsImage(box.boundingRect);
    obj.copyTo(emptyImage(cv::Rect( box.boundingRect.x,  box.boundingRect.y, obj.cols, obj.rows)));
    auto id = std::this_thread::get_id();
    std::stringstream ss;
    ss << id;
    std::string threadEmptyPath = EMPTY_IMAGE_PATH + Files::slash() + ss.str() + IMAGE_EXTENTION;
    std::cerr << "empty image path = " << threadEmptyPath << std::endl;
    cv::imwrite(threadEmptyPath, emptyImage);
    bool ff = false;
    detector->startDetection(threadEmptyPath, box, ff);
    cv::rectangle(objectsImage, box.boundingRect, cv::Scalar(255, 0, 0), 2);
    cv::circle(objectsImage, box.centerPosition, 3, cv::Scalar(0, 255, 0), -1);

}

void clearTemp(){
    std::vector<std::string> files;
    Files::getDirFiles(TEMP_PATH, "", files);
    for(size_t i =0; i<files.size(); ++i){
        remove(files[i].c_str());
    }
}