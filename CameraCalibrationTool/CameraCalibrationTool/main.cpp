#include <ComputerVisionLib/Common/Match.h>
#include <ComputerVisionLib/ImageProcessing/Chessboard/ChessboardCornerDetector.h>
#include <ComputerVisionLib/ImageProcessing/Chessboard/ChessboardSegmentation.h>
#include <ComputerVisionLib/Calibration/IntrinsicCameraCalibration.h>
#include <ComputerVisionLib/Calibration/Circle.h>
#include <ComputerVisionLib/CameraModel/CameraModel.h>
#include <ComputerVisionLib/CameraModel/DistortionModel/BrownModel.h>
#include <ComputerVisionLib/CameraModel/ProjectionModel/PinholeModel.h>
#include <ComputerVisionLib/CameraModel/ProjectionModel/EquidistantModel.h>
#include <ComputerVisionLib/CameraModel/ProjectionModel/EquisolidModel.h>
#include <ComputerVisionLib/CameraModel/ProjectionModel/OrthographicModel.h>
#include <ComputerVisionLib/CameraModel/ProjectionModel/StereographicModel.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <string>
#include <memory>
#include <random>

using namespace std;

void testCircle()
{
	cv::Mat image(400, 400, CV_8UC1);
	image.setTo(0);

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 4.0);

	double cx = 200.0;
	double cy = 200.0;
	double radius = 100.0;
	size_t numSamples = 200;
	double angleSpan = 6.3;
	double angleStep = angleSpan / numSamples;
	Eigen::Array2Xd points(2, numSamples);
	double x, y;
	for (size_t i = 0; i<numSamples; ++i)
	{
		x = cx + cos(i*angleStep)*radius + distribution(generator);
		y = cy + sin(i*angleStep)*radius + distribution(generator);
		points(0, i) = x;
		points(1, i) = y;
		cv::circle(image, cv::Point((int)std::round(x), (int)std::round(y)), 1, 255, 2);
	}

	Cvl::Circle circle(points, 0, numSamples / 4, numSamples / 2);

	cv::circle(image, cv::Point((int)std::round(circle.getCenter().x()), (int)std::round(circle.getCenter().y())), (int)std::round(circle.getRadius()), 255, 1);

	cv::imshow("Circle", image);
	cv::waitKey(0);
}


void testProjectionModel()
{
	Eigen::Array2Xd normalizedCameraPoints(2, 6);
	double v = 0.2;
	normalizedCameraPoints.col(0) = Eigen::Vector2d( v,  v);
	normalizedCameraPoints.col(1) = Eigen::Vector2d( 2*v, -2 * v);
	normalizedCameraPoints.col(2) = Eigen::Vector2d(-4 * v, 4 * v);
	normalizedCameraPoints.col(3) = Eigen::Vector2d(-8 * v, -8 * v);
	normalizedCameraPoints.col(4) = Eigen::Vector2d(-16 * v, -16 * v);
	normalizedCameraPoints.col(5) = Eigen::Vector2d(-32 * v, -32 * v);
	
	Cvl::StereographicModel model(-1200.0, -600.0, 0.0, 0.0); // PinholeModel EquidistantModel EquisolidModel StereographicModel OrthographicModel
	Eigen::Array2Xd imagePoints = model.project(normalizedCameraPoints);
	std::cout << imagePoints << std::endl << std::endl;
	Eigen::Array2Xd comparePoints = model.unproject(imagePoints);
	std::cout << comparePoints << std::endl << std::endl;
	
	std::cout << normalizedCameraPoints - comparePoints << std::endl;
	system("pause");
}

void testDistortionModel()
{
	Eigen::Array2Xd normalizedCameraPoints(2, 6);
	double v = 0.2;
	normalizedCameraPoints.col(0) = Eigen::Vector2d(v, v);
	normalizedCameraPoints.col(1) = Eigen::Vector2d(2 * v, -2 * v);
	normalizedCameraPoints.col(2) = Eigen::Vector2d(-4 * v, 4 * v);
	normalizedCameraPoints.col(3) = Eigen::Vector2d(-8 * v, -8 * v);
	normalizedCameraPoints.col(4) = Eigen::Vector2d(-16 * v, -16 * v);
	normalizedCameraPoints.col(5) = Eigen::Vector2d(-32 * v, -32 * v);

	Cvl::BrownModel model(0.5, 0.1, 0.2, 0.2);
	std::cout << "START" << std::endl;
	Eigen::Array2Xd distortedPoints = model.distort(normalizedCameraPoints);
	std::cout << distortedPoints << std::endl << std::endl;
	Eigen::Array2Xd comparePoints = model.undistort(distortedPoints);
	std::cout << comparePoints << std::endl << std::endl;

	std::cout << normalizedCameraPoints - comparePoints << std::endl;
	system("pause");
}

void testAllModels(
	Eigen::Array2Xd const& templatePoints,
	std::vector<Eigen::Array2Xd> const & imagePointsPerFrame,
	std::vector<std::vector<Cvl::Match>> const & matchesPerFrame)
{
	Cvl::CameraModel pinholeModel = Cvl::CameraModel::create<Cvl::PinholeModel>(); 
	Cvl::CameraModel equidistantModel = Cvl::CameraModel::create<Cvl::EquidistantModel>();
	Cvl::CameraModel equisolidModel = Cvl::CameraModel::create<Cvl::EquisolidModel>();
	Cvl::CameraModel orthographicModel = Cvl::CameraModel::create<Cvl::OrthographicModel>();
	Cvl::CameraModel stereographicModel = Cvl::CameraModel::create<Cvl::StereographicModel>();

	Cvl::CameraModel pinholeModelD = Cvl::CameraModel::create<Cvl::BrownModel, Cvl::PinholeModel>();
	Cvl::CameraModel equidistantModelD = Cvl::CameraModel::create<Cvl::BrownModel, Cvl::EquidistantModel>();
	Cvl::CameraModel equisolidModelD = Cvl::CameraModel::create<Cvl::BrownModel, Cvl::EquisolidModel>();
	Cvl::CameraModel orthographicModelD = Cvl::CameraModel::create<Cvl::BrownModel, Cvl::OrthographicModel>();
	Cvl::CameraModel stereographicModelD = Cvl::CameraModel::create<Cvl::BrownModel, Cvl::StereographicModel>();

	std::vector<std::tuple<bool, double>> results;

	results.push_back(Cvl::IntrinsicCameraCalibration::calibrate(templatePoints, imagePointsPerFrame, matchesPerFrame, pinholeModel));
	results.push_back(Cvl::IntrinsicCameraCalibration::calibrate(templatePoints, imagePointsPerFrame, matchesPerFrame, equidistantModel));
	results.push_back(Cvl::IntrinsicCameraCalibration::calibrate(templatePoints, imagePointsPerFrame, matchesPerFrame, equisolidModel));
	results.push_back(Cvl::IntrinsicCameraCalibration::calibrate(templatePoints, imagePointsPerFrame, matchesPerFrame, orthographicModel));
	results.push_back(Cvl::IntrinsicCameraCalibration::calibrate(templatePoints, imagePointsPerFrame, matchesPerFrame, stereographicModel));

	results.push_back(Cvl::IntrinsicCameraCalibration::calibrate(templatePoints, imagePointsPerFrame, matchesPerFrame, pinholeModelD));
	results.push_back(Cvl::IntrinsicCameraCalibration::calibrate(templatePoints, imagePointsPerFrame, matchesPerFrame, equidistantModelD));
	results.push_back(Cvl::IntrinsicCameraCalibration::calibrate(templatePoints, imagePointsPerFrame, matchesPerFrame, equisolidModelD));
	results.push_back(Cvl::IntrinsicCameraCalibration::calibrate(templatePoints, imagePointsPerFrame, matchesPerFrame, orthographicModelD));
	results.push_back(Cvl::IntrinsicCameraCalibration::calibrate(templatePoints, imagePointsPerFrame, matchesPerFrame, stereographicModelD));

	for (auto const & result : results)
	{
		std::cout << std::get<0>(result) << " " << std::get<1>(result) << std::endl;
	}
}

void someTests()
{
	//testProjectionModel();
	//testDistortionModel();
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
	//std::string imagePath = "D:\\Eigene Dokumente\\Visual Studio 2015\\Projects\\Pics\\";
	//std::string imagePath = "D:\\Projekte\\Pics\\";
	std::string imagePath = "D:\\Projekte\\HandyCam\\Wide\\";

	images.push_back(cv::imread(imagePath + "1.jpg", cv::IMREAD_GRAYSCALE));
	images.push_back(cv::imread(imagePath + "2.jpg", cv::IMREAD_GRAYSCALE));
	images.push_back(cv::imread(imagePath + "3.jpg", cv::IMREAD_GRAYSCALE));
	images.push_back(cv::imread(imagePath + "4.jpg", cv::IMREAD_GRAYSCALE));
	images.push_back(cv::imread(imagePath + "5.jpg", cv::IMREAD_GRAYSCALE));
	//images.push_back(cv::imread(imagePath + "6.jpg", cv::IMREAD_GRAYSCALE));
	//images.push_back(cv::imread(imagePath + "7.jpg", cv::IMREAD_GRAYSCALE));

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

	// EquidistantModel EquisolidModel OrthographicModel StereographicModel PinholeModel
	//Cvl::CameraModel cameraModel = Cvl::CameraModel::create<Cvl::BrownModel, Cvl::PinholeModel>(); // Cvl::BrownModel, 
	//Cvl::IntrinsicCameraCalibration::calibrate(templatePoints, imagePointsPerFrame, matchesPerFrame, cameraModel);

	testAllModels(templatePoints, imagePointsPerFrame, matchesPerFrame);

	system("pause");
	return 0;

}