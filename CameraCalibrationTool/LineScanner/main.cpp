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


void reconstructImage(cv::Mat const & image, Eigen::Affine3d const & modelView, Cvl::CameraModel const & cameraModel, double offset)
{
	Eigen::Vector3d normal = modelView.linear().col(2);
	double d = normal.dot(modelView.translation()+(offset*normal)); 
	Eigen::Vector3d ray;
	Eigen::Vector3d intersection;

	for (size_t y = 0; y < image.rows; ++y)
	{
		for (size_t x = 0; x < image.rows; ++x)
		{
			ray = cameraModel.unprojectAndUndistort(Eigen::Array2d((double)x, (double)y)).matrix().homogeneous().normalized();
			intersection = ray * d / (ray.dot(normal));
		}
	}
}

std::tuple<bool, double, Eigen::Affine3d> calculateModelView(
	cv::Mat const & image, 
	Cvl::CameraModel const & cameraModel, 
	Eigen::Array2Xd const & templatePoints, 
	size_t const cornersPerRow, 
	size_t const cornersPerCol)
{
	Eigen::Array2Xd corners = Cvl::ChessboardCornerDetector::findCorners(image, cornersPerRow, cornersPerCol, false);
	Cvl::ChessboardSegmentation::Result segmentationResult = Cvl::ChessboardSegmentation::match(image, corners, cornersPerRow, cornersPerCol);
	if (segmentationResult.mSuccessful && segmentationResult.mUnambiguous)
	{
		Eigen::Array2Xd alignedTemplatePoints;
		Eigen::Array2Xd alignedImagePoints;
		Eigen::Array2Xd alignedPinholePoints;
		Eigen::Matrix3d homography;
		bool success = false;
		std::tie(alignedTemplatePoints, alignedImagePoints) = Cvl::Match::alignMatches(templatePoints, corners, segmentationResult.mMatches);
		alignedPinholePoints = cameraModel.transformToPinhole(alignedImagePoints);
		std::tie(success, homography) = Cvl::HomographyCalculation::calculate(alignedTemplatePoints, alignedPinholePoints);
		if (success)
		{
			return  Cvl::ModelViewFromHomography::calculate(cameraModel, homography, alignedTemplatePoints, alignedImagePoints);
		}
	}
	return std::make_tuple(false, 100000.0, Eigen::Affine3d());
}




int main(int argc, char *argv[])
{
	// template points
	size_t const cornersPerRow = 9;
	size_t const cornersPerCol = 6;
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

	// initialization
	cv::Mat cleanImage;
	cv::Mat imageMask;
	cv::Mat imageC;
	cv::Mat image;

	Eigen::Affine3d modelView;

	bool success = false;
	double error = 0.0;
	double const maxError = 1.0;
	double const laserPointerOffset = 3.0;

	// set a clean image
	for (;;)
	{
		cap >> imageC;
		cvtColor(imageC, image, cv::COLOR_BGR2GRAY);
		cv::imshow("Set a clean image!", image);
		if (cv::waitKey(30) >= 0)
		{
			cleanImage = image;
			break;
		}
	}

	// main loop
	for (;;)
	{
		cap >> imageC;
		cvtColor(imageC, image, cv::COLOR_BGR2GRAY);
		std::tie(success, error, modelView) = calculateModelView(image, cameraModel, templatePoints, cornersPerRow, cornersPerCol);
		std::cout << error << std::endl;
		if (success && error < maxError)
		{

		}

		cv::imshow("Image", image);
		if (cv::waitKey(30) >= 0)
			break;
	}

	return 0;
}


/*
int main(int argc, char *argv[])
{
	// template points
	size_t const cornersPerRow = 9;
	size_t const cornersPerCol = 6;
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

	// initialization
	cv::Mat imageMask;
	cv::Mat imageC;
	cv::Mat imageT0;
	cv::Mat imageTM1;
	cv::Mat imageTM2;
	cv::Mat imageDiffTM1;

	Eigen::Affine3d modelViewT0;
	Eigen::Affine3d modelViewTM1;
	bool validModelViewT0 = false;
	bool validModelViewTM1 = false;

	bool success = false;
	double error = 0.0;
	double const maxError = 1.0;
	double const laserPointerOffset = 3.0;


	// TODO: vll reicht auch einfach ein vergleichsbild am anfang und der rest geht über die helligkeit und man braucht gar keine 3 bilder!!!

	// main loop
	for (;;)
	{
		// get current image and shift the old ones
		cap >> imageC;
		imageTM2 = imageTM1;
		imageTM1 = imageT0;
		cvtColor(imageC, imageT0, cv::COLOR_BGR2GRAY);

		// calculate the current modelview
		modelViewTM1 = modelViewT0;
		validModelViewTM1 = validModelViewT0;
		std::tie(success, error, modelViewT0) = calculateModelView(imageT0, cameraModel, templatePoints, cornersPerRow, cornersPerCol);

		validModelViewT0 = false;
		std::cout << error << std::endl;
		if (success && error < maxError)
		{
			validModelViewT0 = true;
		}

		// reconstruct the last image
		if (validModelViewTM1)
		{
			//std::cout << modelView.matrix() << std::endl;
			//reconstructImage(image, modelView, cameraModel, laserPointerOffset);
		}



		cv::imshow("Image", imageT0);
		if (cv::waitKey(30) >= 0) 
			break;
	}

	return 0;
}
*/