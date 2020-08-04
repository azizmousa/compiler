# Sketch Builder (compiler part)

sketch builder is an application that use AI and computer Vision technologies to convert design sketch to real code in multiple programming languages, the compiler part is the part which responsible for read the sketch image and translate it to readable information

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes:

### Prerequisites

- gcc compiler(build version 8.3.0)
- cmake build tool
- opencv library
- Linux System (ubuntu, debian are tested)
- Download the "config" zip from the lik below


### Installing

1- clone the project to your device and unzip it.

2- open the compiler directory suppose it called COMPILER_HOME and create directory called build then open it in the terminal.

```
$ cd COMPILER_HOME
$ mkdir build
$ cd build
```

3- compile the project and create the executable file

```
$ cmake ..
$ make
```

4- open the executable file directory

```
$ cd ..
$ cd bin
```

5- extract the config zip in the bin dir along side of compiler file "you should extract the three dirs directly not in another directory"

```
$ ls
compiler configurations names weights
```

## How to use the application

```
$ ./compiler [INPUT_TYPE][INIPUT_PATH][CONFIDANCE]
```

1- [INPUT_TYPE] (required) :=> there are two parameters -p or -d
	>use -p if you want detect the objects using nural network (fast but less accurate)
	>use -d if you want detect the objects using image processing but it required symmetric image (time longer but more accurate) 

2- [INPUT_PATH] (required):=> you can enter multiple image paths separated by space or path to directory the containe multiple images or just single image.

3- [CONFIDANCE] :=> select the confidance that you want but should be between [0.0:1.0] the deafault "0.30"

### Example

- lets suppose we have file "/home/Downloads/test.jpg" and want to process this file with confidance 50%

```
$ ./compiler -p /home/Downloads/test.jpg 0.50
```

## After running this command:
	1- you will find views dir include readable file called test.viw containe the detected objects inforamtion
	2- you will find temp dir include processed images that show which object is detected and where "just for seen"

### Help Links

- [the objects that involved to this project] (https://drive.google.com/open?id=1b-340IdraxEP1fVrbMghkT0Jri7GZ9Vy)
- [testset] (https://drive.google.com/open?id=1ZSLnwHn2d6g6KblDj-P3Hp_Dm_IgQlbz)
- [config zip "REQUIRED"] (https://drive.google.com/file/d/1mGTFCG2KgpQlczzK4I9A8G0yjxkBMS4J/view)


## Main Project:
- [Sketch builder] (https://github.com/azizmousa/Sketch-Builder)
