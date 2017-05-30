#include <ComputerVisionLib/Common/Match.h>
#include <ComputerVisionLib/ImageProcessing/Chessboard/ChessboardCornerDetector.h>
#include <ComputerVisionLib/ImageProcessing/Chessboard/ChessboardSegmentation.h>
#include <ComputerVisionLib/CameraModel/CameraModel.h>
#include <ComputerVisionLib/CameraModel/DistortionModel/BrownModel.h>
#include <ComputerVisionLib/CameraModel/ProjectionModel/PinholeModel.h>
#include <ComputerVisionLib/Reconstruction/HomographyCalculation.h>
#include <ComputerVisionLib/Reconstruction/ModelViewFromHomography.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <chrono>
#include <string>

using namespace std;


int main(int argc, char *argv[])
{
	// template points
	int const cornersPerRow = 9;
	int const cornersPerCol = 6;
	double const squareLength = 26; // mm
	Eigen::Array2Xd templatePoints(2, cornersPerRow*cornersPerCol);
	int templateIndex = 0;
	for (int y = 0; y < cornersPerCol; ++y)
	{
		for (int x = 0; x < cornersPerRow; ++x)
		{
			templatePoints.col(templateIndex) = Eigen::Array2d(x*squareLength, y*squareLength);
			++templateIndex;
		}
	}

	// camera model
	Cvl::CameraModel cameraModel = Cvl::CameraModel::create<Cvl::BrownModel, Cvl::PinholeModel>();
	Eigen::VectorXd cameraParameters(8);
	cameraParameters(0) = 0.0;
	cameraParameters(1) = 0.0;
	cameraParameters(2) = 0.0;
	cameraParameters(3) = 0.0;
	cameraParameters(4) = 0.0;
	cameraParameters(5) = 0.0;
	cameraParameters(6) = 0.0;
	cameraParameters(7) = 0.0;
	cameraModel.setAllParameters(cameraParameters);

	// open video stream
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		std::cout << "Could not open camera stream." << std::endl;
		system("pause");
		return -1;
	}

	// main loop
	cv::Mat imageC;
	cv::Mat image;

	Eigen::Array2Xd alignedTemplatePoints;
	Eigen::Array2Xd alignedImagePoints;
	Eigen::Array2Xd alignedPinholePoints;
	Eigen::Matrix3d homography;
	Eigen::Affine3d modelView;
	bool success = false;
	double error = 0.0;
	double const maxError = 1.0;


	for (;;)
	{
		cap >> imageC; 
		cvtColor(imageC, image, cv::COLOR_BGR2GRAY);

		Eigen::Array2Xd corners = Cvl::ChessboardCornerDetector::findCorners(image, cornersPerRow, cornersPerCol, false);
		Cvl::ChessboardSegmentation::Result segmentationResult = Cvl::ChessboardSegmentation::match(image, corners, cornersPerRow, cornersPerCol);
		if (segmentationResult.mSuccessful && segmentationResult.mUnambiguous)
		{
			std::tie(alignedTemplatePoints, alignedImagePoints) = Cvl::Match::alignMatches(templatePoints, corners, segmentationResult.mMatches);
			alignedPinholePoints = cameraModel.transformToPinhole(alignedImagePoints);
			std::tie(success, homography) = Cvl::HomographyCalculation::calculate(alignedTemplatePoints, alignedPinholePoints);
			if (success)
			{
				std::tie(success, error, modelView) = Cvl::ModelViewFromHomography::calculate(cameraModel, homography, alignedTemplatePoints, alignedImagePoints);
				if (success && error < maxError)
				{
					std::cout << error << std::endl;
					std::cout << modelView.matrix() << std::endl;
				}
			}
		}

		cv::imshow("Image", image);
		if (cv::waitKey(30) >= 0) 
			break;
	}

	return 0;
}