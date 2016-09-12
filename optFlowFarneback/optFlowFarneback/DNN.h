#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <iomanip>
using namespace std;

struct probLabel
{
	string label;
	float probability;
};


class DNN
{

public:



/* Find best class for the blob (i. e. class with maximal probability) */
void getMaxClass(dnn::Blob &probBlob, int *classId, double *classProb)
{
	Mat probMat = probBlob.matRefConst().reshape(1, 1); //reshape the blob to 1x1000 matrix
	Point classNumber;

	minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
	*classId = classNumber.x;
}

std::vector<String> readClassNames(const char *filename = "synset_words.txt")
{
	std::vector<String> classNames;

	std::ifstream fp(filename);
	if (!fp.is_open())
	{
		std::cerr << "File with classes labels not found: " << filename << std::endl;
		exit(-1);
	}

	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name.substr(name.find(' ') + 1));
	}

	fp.close();
	return classNames;
}

probLabel predictLabel(Mat img)
{
	String modelTxt = "bvlc_googlenet.prototxt";
	String modelBin = "bvlc_googlenet.caffemodel";
	//String imageFile = (argc > 1) ? argv[1] : "3.jpg";

	//! [Create the importer of Caffe model]
	Ptr<dnn::Importer> importer;
	try                                     //Try to import Caffe GoogleNet model
	{
		importer = dnn::createCaffeImporter(modelTxt, modelBin);
	}
	catch (const cv::Exception &err)        //Importer can throw errors, we will catch them
	{
		std::cerr << err.msg << std::endl;
	}
	//! [Create the importer of Caffe model]

	if (!importer)
	{
		std::cerr << "Can't load network by using the following files: " << std::endl;
		std::cerr << "prototxt:   " << modelTxt << std::endl;
		std::cerr << "caffemodel: " << modelBin << std::endl;
		std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
		std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
		exit(-1);
	}

	//! [Initialize network]
	dnn::Net net;
	importer->populateNet(net);
	importer.release();                     //We don't need importer anymore
	//! [Initialize network]

	//! [Prepare blob]
	//Mat img = imread(imageFile);
	//if (img.empty())
	//{
	//	std::cerr << "Can't read image from the file: " << imageFile << std::endl;
	//	exit(-1);
	//}

	resize(img, img, Size(224, 224));       //GoogLeNet accepts only 224x224 RGB-images
	dnn::Blob inputBlob = dnn::Blob(img);   //Convert Mat to dnn::Blob image batch
	//! [Prepare blob]

	//! [Set input blob]
	net.setBlob(".data", inputBlob);        //set the network input
	//! [Set input blob]

	//! [Make forward pass]
	net.forward();                          //compute output
	//! [Make forward pass]

	//! [Gather output]
	dnn::Blob prob = net.getBlob("prob");   //gather output of "prob" layer

	int classId;
	double classProb;
	getMaxClass(prob, &classId, &classProb);//find the best class
	//! [Gather output]

	probLabel obj;
	

	//! [Print results]
	std::vector<String> classNames = readClassNames();
	std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
	std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
	//! [Print results]

	obj.label = classNames.at(classId);
	obj.probability = classProb * 100;

	return obj;
} //main

};