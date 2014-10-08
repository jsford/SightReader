all: main.cpp
	g++ main.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -g -o sightread


