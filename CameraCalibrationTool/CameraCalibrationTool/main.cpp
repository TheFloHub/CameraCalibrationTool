#include <ComputerVisionLib/Common/Match.h>
#include <ComputerVisionLib/ImageProcessing/Chessboard/ChessboardCornerDetector.h>
#include <ComputerVisionLib/ImageProcessing/Chessboard/ChessboardSegmentation.h>
#include <ComputerVisionLib/Calibration/IntrinsicCameraCalibration.h>
#include <ComputerVisionLib/CameraModel/CameraModel.h>
#include <ComputerVisionLib/CameraModel/DistortionModel/BrownModel.h>
#include <ComputerVisionLib/CameraModel/ProjectionModel/PinholeModel.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <string>
#include <memory>

using namespace std;


void someTests()
{
	//Eigen::Vector4d vec(1,2,3,4);
	//std::cout << vec.tail(2) << std::endl;
}


int main(int argc, char *argv[])
{
	someTests();
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

	std::vector<cv::Mat> images;
	images.push_back(cv::imread("D:\\Projekte\\Pics\\c1.jpg", cv::IMREAD_GRAYSCALE));
	images.push_back(cv::imread("D:\\Projekte\\Pics\\c2.jpg", cv::IMREAD_GRAYSCALE));
	images.push_back(cv::imread("D:\\Projekte\\Pics\\c3.jpg", cv::IMREAD_GRAYSCALE));
	images.push_back(cv::imread("D:\\Projekte\\Pics\\c4.jpg", cv::IMREAD_GRAYSCALE));
	images.push_back(cv::imread("D:\\Projekte\\Pics\\c5.jpg", cv::IMREAD_GRAYSCALE));

	std::vector<Eigen::Array2Xd> imagePointsPerFrame;
	std::vector<std::vector<Cvl::Match>> matchesPerFrame;

	for (auto const& image : images)
	{
		Eigen::Array2Xd corners = Cvl::ChessboardCornerDetector::findCorners(image, cornersPerRow, cornersPerCol, true);
		Cvl::ChessboardSegmentation::Result segmentationResult = Cvl::ChessboardSegmentation::match(image, corners, cornersPerRow, cornersPerCol);
		if (segmentationResult.mSuccessful)
		{
			imagePointsPerFrame.push_back(corners);
			matchesPerFrame.push_back(segmentationResult.mMatches);
		}
		std::cout << segmentationResult.mSuccessful << " / " << segmentationResult.mUnambiguous << std::endl;

		// all corners
		for (int i = 0; i<corners.cols(); ++i)
		{
			cv::circle(image, cv::Point((int)corners(0, i), (int)corners(1, i)), 3, 255, 2);
		}

		// matches
		for (auto const& match : segmentationResult.mMatches)
		{
			Eigen::Array2d p = corners.col(match.mMeasuredId);
			cv::putText(image, std::to_string(match.mTemplateId), cv::Point((int)p.x() + 5, (int)p.y() - 4), cv::FONT_HERSHEY_SIMPLEX, 0.5, 255, 1);
		}

		//cv::imshow("Points", image);
		//cv::waitKey(0);
	}

	Cvl::CameraModel cameraModel(
		std::unique_ptr<Cvl::DistortionModel>(new Cvl::BrownModel(0.0, 0.0, 0.0, 0.0)),
		std::unique_ptr<Cvl::ProjectionModel>(new Cvl::PinholeModel(840.0, 850.0, 320.0, 240.0)));
	Cvl::IntrinsicCameraCalibration::calibrate(templatePoints, imagePointsPerFrame, matchesPerFrame, cameraModel);


	system("pause");
	return 0;

}