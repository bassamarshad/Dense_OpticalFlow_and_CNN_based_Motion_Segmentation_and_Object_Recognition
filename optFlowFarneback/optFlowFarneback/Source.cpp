#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/optflow/motempl.hpp"
//#include "Python.h"
#include <opencv2/dnn.hpp>
#include "opencv2/video.hpp"
/*
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
*/
#include <iostream>
#include<ctime>
#include "DNN.h"

using namespace cv;
using namespace std;
using namespace cv::dnn;
//using namespace cv::cuda;


Mat findBlobs(Mat src,Mat original,int frameCount) ;
Mat findBlobsWithLabels(Mat src, Mat original, int frameCount);

DNN dnn1;
probLabel pl;

static void help()
{
	cout <<
		"\nThis program demonstrates dense optical flow algorithm by Gunnar Farneback\n"
		"Mainly the function: calcOpticalFlowFarneback()\n"
		"Call:\n"
		"./fback\n"
		"This reads from video camera 0\n" << endl;
}
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
	double, const Scalar& color)
{
	for (int y = 0; y < cflowmap.rows; y += step)
		for (int x = 0; x < cflowmap.cols; x += step)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x);
			line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
				color);
			circle(cflowmap, Point(x, y), 2, color, -1);
		}
}

int optMag=4;
RNG rng(12345);

float MHI_DURATION = 0.05;





int main(int argc, char** argv)
{
	namedWindow("Magnitude Thresh Trackbar", 0);
	//setMouseCallback("S-V Trackbar", onMouse, 0);
	createTrackbar("Optical Flow Magnitude", "Magnitude Thresh Trackbar", &optMag, 50, 0);

	cv::CommandLineParser parser(argc, argv, "{help h||}");
	if (parser.has("help"))
	{
		help();
		return 0;
	}

	VideoCapture cap;
	
	Mat frame1, ret, motion_mask;
	cap.read(frame1);

	Size frame_size = frame1.size();
	int h = frame_size.height;
	int w = frame_size.width;
	
	//Mat motion_history(h, w, CV_32FC1, Scalar(0, 0, 0));
	Mat seg_mask(h, w, CV_32FC1, Scalar(0, 0, 0));
	vector<Rect> seg_bounds;
	
	cap.open(0);
    //cap.open("volleyball.mp4");
	cout << "\n Camera Frame width \n" << cap.get(CV_CAP_PROP_FRAME_WIDTH);
	cout << "Camera Frame height \n" << cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	help();
	if (!cap.isOpened())
		return -1;

	Mat flow, cflow, frame,lastFrame;
	UMat gray, prevgray, uflow;
	//namedWindow("flow", 1);

	Mat res;
	int frameCount = 0;
	bool useGPU = false;
	
	//cv::cuda::GpuMat d_flow;
	//cv::Ptr<cv::cuda::FarnebackOpticalFlow> d_calc = cv::cuda::FarnebackOpticalFlow::create();

	//WE initialze the frame size over here 
	// 780p resolution : 1280x720
	//1080p resoltuion : 1920 x 1080
	Size videoSize = Size(1280, 720);


	VideoWriter motionVideo("MotionDetectedObjects.wmv", CV_FOURCC('W', 'M', 'V', '1'), 10, videoSize, true);
	VideoWriter labelVideo("ObjectLabels.wmv", CV_FOURCC('W', 'M', 'V', '1'), 10, videoSize, true);


	Mat labelRect;
	for (;;)
	{
		
		cap >> frame;
		if (!cap.read(frame))
			break;

		resize(frame, frame, videoSize, 0, 0, INTER_CUBIC);
		cvtColor(frame, gray, COLOR_BGR2GRAY);

		if (!prevgray.empty())
		{

			Mat xy[2];

			/*
			if (useGPU)
			{ 
		    cv::cuda::GpuMat d_frameL(prevgray), d_frameR(gray);

			d_calc->calc(d_frameL, d_frameR, d_flow);
		
			cuda::GpuMat planes[2];
			cuda::split(d_flow, planes);

			planes[0].download(xy[0]);
			planes[1].download(xy[1]);
			}
			else
			{ 
			*/
			calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);
			cvtColor(prevgray, cflow, COLOR_GRAY2BGR);

			uflow.copyTo(flow);
		
			//drawOptFlowMap(flow, cflow, 8, 1.5, Scalar(0, 255, 0));
			//imshow("flow", cflow);
			split(flow, xy);
			//}

			//calculate angle and magnitude
			Mat magnitude, angle;
			cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);

			
			//translate magnitude to range [0;1]
			//double mag_max;
			//minMaxLoc(magnitude, 0, &mag_max);
			//magnitude.convertTo(magnitude, -1, 1.0 / mag_max);

			
			cv::threshold(magnitude, res, optMag,100, CV_THRESH_BINARY);
			imshow("flow magnitude", res);
			
			Mat res2;
			res.convertTo(res2, CV_8UC1, 255);
		
			// Create a structuring element (SE) and do some morphological operation in order to close holes, get unified connected components
			int morph_size = 2;
			Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
			morphologyEx(res2, res2, MORPH_CLOSE, element, Point(-1, -1), 12);
			//morphologyEx(src, src, MORPH_ERODE, element, Point(-1, -1), 8);

			/*
			//Motion History and segmentMotion
			motion_mask = res2.clone();

			// Create a structuring element (SE)
			int morph_size = 2;
			Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
			morphologyEx(motion_mask, motion_mask, MORPH_CLOSE, element, Point(-1, -1), 4);

			double timestamp = 1000.0*clock() / CLOCKS_PER_SEC;

			Mat motion_history(frame.rows, frame.cols, CV_32FC1, Scalar(0, 0, 0));
			cv::motempl::updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION);
			cv::motempl::segmentMotion(motion_history, seg_mask, seg_bounds, timestamp, 10000);


			
			for (int i = 0; i < seg_bounds.size(); i++)
			{
				if(seg_bounds[i].area() > 3000)
				{ 
					Mat ROI = frame(seg_bounds[i]);
					String fileName = to_string(frameCount) + ".jpg";
						imwrite(fileName, ROI);
				rectangle(frame, seg_bounds[i], CV_RGB(255, 0, 0), 2, 8);
				}
			}

			Mat seg1;
			seg_mask.convertTo(seg1, CV_8UC1);
			imshow("rects", frame);
			imshow("segMask", seg1);
			seg_bounds.clear();
			*/



			Mat motionRect=findBlobs(res2, frame, frameCount);
			motionVideo.write(motionRect);

		
			if (frameCount%5==0)
			{ 
			labelRect=findBlobsWithLabels(res2, frame, frameCount);
			labelVideo.write(labelRect);
			}
			else
				labelVideo.write(labelRect);

		}
		if (waitKey(10) >= 0)
			break;

		char c = (char)waitKey(1);
		if (c == 27)
			break;
		switch (c)
		{
		case 'e':
		 break;
		}

		std::swap(prevgray, gray);
		//std::swap(lastFrame, frame);
		frameCount++;
	}
	return 0;
}

Mat findBlobs(Mat src,Mat original,int frameCount) {

	Mat dst;
	cvtColor(src, dst, CV_GRAY2BGR);

	//wifi passwd - c10c5b03

	vector<vector<Point>> contours; // storing contour
	vector<Vec4i> hierarchy;
	
	findContours(src.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	Mat localFrame = original.clone();
	vector<double> areas(contours.size());


	for (int i = 0; i < contours.size(); i++) {
		areas[i] = contourArea(Mat(contours[i]));
		if (areas[i] > 1500)
		{
			Rect rect1 = boundingRect(contours[i]);
	        rectangle(dst, rect1, CV_RGB(255, 0, 0), 2, 8);
			rectangle(localFrame, rect1, CV_RGB(255, 0, 0), 2, 8);

			//Below was to store the ROI in the project directory !
			//Mat ROI = original(rect1);
			//String fileName = to_string(frameCount) + ".jpg";
		    //imwrite(fileName, ROI);
			//Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		
		}
	}

	imshow("Blobs Rect", dst);
	
	imshow("On original", localFrame);

	return localFrame;

}


Mat findBlobsWithLabels(Mat src, Mat original, int frameCount) {

	
	vector<vector<Point>> contours; // storing contour
	vector<Vec4i> hierarchy;
	
	findContours(src.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	vector<double> areas(contours.size());

	for (int i = 0; i < contours.size(); i++) {
		Rect rect1 = boundingRect(contours[i]);
		if (rect1.area()>2000)
		{
			Mat i1 = original(rect1);
				pl = dnn1.predictLabel(i1);
				String label = pl.label;
				std::stringstream ss;
				ss << std::fixed << std::setprecision(2) << pl.probability;
				std::string mystring = ss.str();
				String probability = " " + mystring;
				if (!mystring.empty() || !label.empty() )
				{
					//Below to check if Point above Rect (for label text) is in the Mat/frame
					cv::Rect rect(cv::Point(), original.size());
					cv::Point p(rect1.x - 10, rect1.y - 10);
					if (rect.contains(p))
					{ 
						putText(original, label + probability, p, FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255, 255, 255), 2, CV_AA);
					}
					else
					{
						putText(original, label + probability, Point(rect1.x, rect1.y + rect1.height + 20), FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255, 255, 255), 2, CV_AA);
					}
					rectangle(original, rect1, CV_RGB(255, 0, 0), 2, 8);
				}
				waitKey(10);
		}
	}

	imshow("Object Labels", original);

	return original;


}


