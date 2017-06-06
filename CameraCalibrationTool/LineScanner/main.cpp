#include <ComputerVisionLib/Common/Match.h>
#include <ComputerVisionLib/ImageProcessing/Chessboard/ChessboardCornerDetector.h>
#include <ComputerVisionLib/ImageProcessing/Chessboard/ChessboardSegmentation.h>
#include <ComputerVisionLib/CameraModel/CameraModel.h>
#include <ComputerVisionLib/CameraModel/DistortionModel/BrownModel.h>
#include <ComputerVisionLib/CameraModel/ProjectionModel/PinholeModel.h>
#include <ComputerVisionLib/Reconstruction/HomographyCalculation.h>
#include <ComputerVisionLib/Reconstruction/ModelViewFromHomography.h>
#include <ComputerVisionLib/Reconstruction/ReconstructionError.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <chrono>
#include <string>
#include <fstream>

using namespace std;


void reconstructImage(
	cv::Mat const & image, 
	Eigen::Affine3d const & modelView, 
	Cvl::CameraModel const & cameraModel, 
	double const offset, 
	size_t const cornersPerRow,
	size_t const cornersPerCol,
	Eigen::Array2Xd const & projectedPoints,
	cv::Mat & recoImage,
	cv::Mat & recoWeightsImage)
{
	cv::Mat testImage = image.clone();
	
	// params
	int const intensityThreshold = 40;
	int const minLength = 5;
	int const maxLength = 30;

	// find min and max x
	int rangeMinX = 0;
	int rangeMaxX = image.rows;
	Eigen::Index minRow, minCol;
	double minPx = projectedPoints.row(0).minCoeff(&minRow, &minCol);

	if ((size_t) minCol >= cornersPerRow*(cornersPerCol-1))
	{
		rangeMaxX = (int)minPx - 50;
		cv::line(testImage, cv::Point(rangeMaxX, 0), cv::Point(rangeMaxX, image.rows), 255);
	}
	else
	{
		rangeMinX = (int) projectedPoints.row(0).maxCoeff() + 50;
		cv::line(testImage, cv::Point(rangeMinX, 0), cv::Point(rangeMinX, image.rows), 255);
	}
	
	// plane params
	Eigen::Vector3d normal = modelView.linear().col(2);
	double d = normal.dot(modelView.translation()+(offset*normal)); 
	Eigen::Vector3d ray;
	Eigen::Vector3d intersection;

	// find laser line in each image row
	for (int y = 0; y < image.rows; ++y)
	{
		// find x coordinate of laser line
		bool lineStarted = false;
		double xTotal = 0.0;
		int numPixels = 0;
		double totalWeight = 0.0;
		unsigned char intensity = 0;
		bool lineFound = false;
		double lineX = 0.0;
		int x = rangeMinX;

		while (x < rangeMaxX && !lineFound)
		{
			intensity = image.at<unsigned char>(y, x);
			if (intensity >= intensityThreshold)
			{
				lineStarted = true;
				xTotal += (double)(x*intensity);
				++numPixels;
				totalWeight += intensity;
			}
			else if (lineStarted && intensity < intensityThreshold)
			{
				lineStarted = false;
				if (numPixels > minLength && numPixels < maxLength)
				{
					lineX = xTotal/totalWeight;
					lineFound = true;
				}
				xTotal = 0.0;
				numPixels = 0;
				totalWeight = 0.0;
			}
			++x;
		}

		if (lineFound)
		{
			testImage.at<unsigned char>(y, (int)lineX) = 255;
			ray = cameraModel.unprojectAndUndistort(Eigen::Array2d(lineX + 0.5, (double)y + 0.5)).matrix().homogeneous().normalized();
			intersection = ray * d / (ray.dot(normal));

			recoImage.at<cv::Vec3d>    (y, (int)lineX) += cv::Vec3d(intersection.x(), intersection.y(), intersection.z());
			recoWeightsImage.at<double>(y, (int)lineX) += 1.0;
		}
	}

	cv::imshow("testImage", testImage);
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
	// other params
	double const maxError = 1.0;
	double const laserPointerOffset = -10.0; // mm

	// template points
	size_t const cornersPerRow = 3;
	size_t const cornersPerCol = 4;
	double const squareLength = 10; // mm
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
	cameraParameters(0) = -665.96;
	cameraParameters(1) = -666.215;
	cameraParameters(2) = 328.892;
	cameraParameters(3) = 241.901;
	cameraParameters(4) = -4.5598e-05;
	cameraParameters(5) = 5.71782e-05;
	cameraParameters(6) = -0.0265902;
	cameraParameters(7) = 0.0454921;
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
	cv::Mat coloredImage;
	cv::Mat greyImage;
	cv::Mat guiImage;
	cv::Mat diffImage;
	cv::Mat channels[3];
	cv::Mat recoImage;
	cv::Mat recoWeightsImage;

	Eigen::Affine3d modelView; 
	Eigen::Array2Xd projectedPoints;

	bool success = false;
	double error = 0.0;

	// set a clean image
	while (cv::waitKey(30) != 32)
	{
		cap >> coloredImage;
		cv::imshow("Set a first image.", coloredImage);
	}
	cleanImage = cv::Mat::zeros(coloredImage.size(), CV_8UC1);
	int const numInitImages = 20;
	for (int i = 0 ; i < numInitImages; ++i)
	{
		cap >> coloredImage;
		cv::split(coloredImage, channels);
		cleanImage += channels[2] / numInitImages;
	}
	recoImage = cv::Mat::zeros(cleanImage.rows, cleanImage.cols, CV_64FC3);
	recoWeightsImage = cv::Mat::zeros(cleanImage.rows, cleanImage.cols, CV_64FC1);
	cv::destroyAllWindows();

	// main loop
	while (cv::waitKey(30) != 27)
	{
		cap >> coloredImage;
		cvtColor(coloredImage, greyImage, cv::COLOR_BGR2GRAY);
		guiImage = coloredImage.clone();
		std::tie(success, error, modelView) = calculateModelView(greyImage, cameraModel, templatePoints, cornersPerRow, cornersPerCol);
		if (success && error < maxError)
		{
			projectedPoints = Cvl::ReconstructionError::project(modelView, cameraModel, templatePoints);
			for (Eigen::Index i=0; i<projectedPoints.cols(); ++i)
			{
				cv::circle(guiImage, cv::Point((int)std::round(projectedPoints(0, i)), (int)std::round(projectedPoints(1, i))), 3, 255, 2);
			}
			cv::split(coloredImage, channels);
			cv::absdiff(channels[2], cleanImage, diffImage);
			cv::GaussianBlur(diffImage, diffImage, cv::Size(0, 0), 2.0);

			reconstructImage(
				diffImage, modelView, cameraModel, laserPointerOffset, 
				cornersPerRow, cornersPerCol, projectedPoints, 
				recoImage, recoWeightsImage);

			cv::imshow("Diff", diffImage);
		} 
		else
		{
			std::cout << "Tracking lost or high error: " << error << " / " << success << std::endl;
		}
		cv::imshow("Scan ...", guiImage);
	}

	// write obj file
	std::ofstream objFile("scan.obj");
	if (objFile.is_open())
	{
		cv::Vec3d p;
		double w;
		for (int y = 0; y < recoImage.rows; ++y)
		{
			for (int x = 0; x < recoImage.rows; ++x)
			{
				w = recoWeightsImage.at<double>(y, x);
				if (w > 0.0)
				{
					p = recoImage.at<cv::Vec3d>(y, x);
					objFile << "v " << p(0) / w << " " << p(1) / w << " " << p(2) / w << " \n";
				}
			}
		}
		objFile.close();
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