#ifndef NETWORKS_LOADER_
#define NETWORKS_LOADER_

#include <string>
#include <vector>
#include <opencv2/dnn/dnn.hpp>

class Loader{
private:
    std::string weightsPath, namesPath, configurationsPath;

    std::vector<std::string> weightsVector, namesVector, cfgVector;
    std::vector<cv::dnn::Net> networks;
    std::vector<std::vector<std::string>> classes;
    
    void loadVectors();
    void loadNets();
    void loadClasses();

public:

    Loader(const std::string weightsPath, const std::string namesPath, const std::string configurationsPath);
    ~Loader();

    size_t getNetworksNumber()const;
    size_t getPublicClassesNumber()const;
    
    cv::dnn::Net getNet(const size_t index)const;
    std::vector<cv::dnn::Net> getAllNets()const;
    void getClassesVector(const size_t index, std::vector<std::string> &classes)const;
    std::string getClass(const size_t vIndex, const size_t cIndex)const;

};

#endif