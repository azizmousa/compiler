#ifndef DETECTOR_EXCEPTIONS_
#define DETECTOR_EXCEPTIONS_

#include <string>
#include <exception>

struct DirectoryNotFoundException : public std::exception
{
    DirectoryNotFoundException(std::string &file){
        message = file + " can not open.";
    }
    virtual const char* what()const throw(){
        return message.c_str();
    }
private:
    std::string message;
};

#endif