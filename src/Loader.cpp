
#include "compiler/Loader.h"
#include "compiler/Files.h"

#include <fstream>

Loader::Loader(const std::string weightsPath, const std::string namesPath, const std::string configurationsPath){
    this->weightsPath = weightsPath;
    this->namesPath = namesPath;
    this->configurationsPath = configurationsPath;

    this->loadVectors();
    this->loadClasses();
    this->loadNets();    
}
Loader::~Loader(){}

void Loader::loadVectors(){
    Files::getDirFiles(this->weightsPath, ".weights", this->weightsVector);
    Files::getDirFiles(this->namesPath, ".names", this->namesVector);
    Files::getDirFiles(this->configurationsPath, ".cfg", this->cfgVector);
}

void Loader::loadNets(){
    for(size_t i =0; i < this->weightsVector.size(); ++i){
        // Give the configuration and weight files for the model
        std::string modelConfiguration = this->configurationsPath + Files::slash() + cfgVector[i];
        std::string modelWeights = this->weightsPath + Files::slash() +weightsVector[i];
        // std::cerr << modelConfiguration << "\t" << modelWeights << std::endl;
        // Load the network
        cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        this->networks.push_back(net);
     }
}
void Loader::loadClasses(){
    for(size_t i = 0; i < this->weightsVector.size();++i){
        // Load names of classes
        std::string classesFile = this->namesPath + Files::slash() +namesVector[i];
        std::ifstream ifs(classesFile.c_str());
        std::string line;
        std::vector<std::string> temp;
        while (getline(ifs, line)) temp.push_back(line);

        this->classes.push_back(temp);
    }
}

size_t Loader::getNetworksNumber()const{
    return this->networks.size();
}
size_t Loader::getPublicClassesNumber()const{
    return this->classes.size();
}

cv::dnn::Net Loader::getNet(const size_t index)const{
    return this->networks[index];
}
void Loader::getClassesVector(const size_t index, std::vector<std::string> &classes)const{
    classes = this->classes[index];
}
std::string Loader::getClass(const size_t vIndex, const size_t cIndex)const{
    return this->classes[vIndex][cIndex];
}

std::vector<cv::dnn::Net> Loader::getAllNets()const{
    static std::vector<cv::dnn::Net> myNets = this->networks;
    return myNets;
}

std::vector<std::vector<std::string>> Loader::getAllClasses()const{
    static std::vector<std::vector<std::string>> myClasses = this->classes;
    return myClasses;
}