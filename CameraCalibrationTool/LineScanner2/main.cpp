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
	cv::Mat const & objectMask,
	Eigen::Affine3d const & leftModelView,
	Eigen::Affine3d const & rightModelView,
	Cvl::CameraModel const & cameraModel,
	cv::Mat & recoImage,
	cv::Mat & recoWeightsImage,
	cv::Mat & guiImage) // colored
{
	// params 
	unsigned char const minIntensity = 30;
	int const minNumberOfPixels = 4;

	// reconstruct laser plane

	
	
	
	
	//std::vector<cv::Vec4i> lines;
	//cv::HoughLinesP(image, lines, 1, CV_PI / 180, 80, 30, 10);
	//for (size_t i = 0; i < lines.size(); i++)
	//{
	//	cv::line(guiImage, cv::Point(lines[i][0], lines[i][1]),
	//		cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 255, 255), 3, 8);
	//}




	// find laser line in each image column and reconstruct the point
	Eigen::Vector3d leftPlaneNormal = leftModelView.linear().col(2);
	double leftPlaneD = leftPlaneNormal.dot(leftModelView.translation());
	Eigen::Vector3d rightPlaneNormal = rightModelView.linear().col(2);
	double rightPlaneD = rightPlaneNormal.dot(rightModelView.translation());

	Eigen::Matrix3Xd planePoints(3, image.cols);
	size_t numberOfPlanePoints = 0;
	Eigen::Vector3d ray;
	Eigen::Vector3d intersection;
	for (int x = 0; x < image.cols; ++x)
	{
		// find y coordinate of laser line
		double yTotal = 0.0;
		int numPixels = 0;
		double totalWeight = 0.0;
		unsigned char intensity = 0;
		double lineY = 0.0;

		for (int y = 0; y < image.rows; ++y)
		{
			if (objectMask.at<unsigned char>(y, x) == 0)
			{
				intensity = image.at<unsigned char>(y, x);
				if (intensity > minIntensity)
				{
					yTotal += (double)intensity*(double)y;
					totalWeight += (double)intensity;
					++numPixels;
				}
			}
		}

		if (numPixels > minNumberOfPixels)
		{
			lineY = yTotal / totalWeight + 0.5;
			guiImage.at<cv::Vec3b>((int)lineY, x) = cv::Vec3b(255, 0, 255);
			ray = cameraModel.unprojectAndUndistort(Eigen::Array2d((double)x + 0.5, lineY)).matrix().homogeneous().normalized();
			if (x < image.cols/2)
			{
				intersection = ray * leftPlaneD / (ray.dot(leftPlaneNormal));
			}
			else
			{
				intersection = ray * rightPlaneD / (ray.dot(rightPlaneNormal));
			}

			planePoints.col(numberOfPlanePoints) = intersection;
			++numberOfPlanePoints;
		}
	}
	planePoints.conservativeResize(3, numberOfPlanePoints);


	Eigen::Vector3d mean = planePoints.rowwise().mean();
	Eigen::Matrix3Xd centeredPlanePoints = planePoints.colwise() - mean;
	Eigen::Matrix3d covar = centeredPlanePoints*centeredPlanePoints.transpose();

	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(covar);
	if (eigensolver.info() != Eigen::Success)
		return;

	Eigen::Vector3d normal = eigensolver.eigenvectors().col(0); // correct one???????????????????????????????????????????
	//Eigen::Vector3d normal = centeredPlanePoints.jacobiSvd(Eigen::ComputeFullV).matrixV().col(3).head<3>();
	double d = normal.dot(mean);
	
	// reconstruct object
	// find laser line in each image column and reconstruct the point
	for (int x = 0; x < image.cols; ++x)
	{
		// find y coordinate of laser line
		double yTotal = 0.0;
		int numPixels = 0;
		double totalWeight = 0.0;
		unsigned char intensity = 0;
		double lineY = 0.0;

		for (int y = 0; y < image.rows; ++y)
		{
			if (objectMask.at<unsigned char>(y, x) != 0)
			{
				intensity = image.at<unsigned char>(y, x);
				if (intensity > minIntensity)
				{
					yTotal += (double)intensity*(double)y;
					totalWeight += (double)intensity;
					++numPixels;
				}
			}
		}

		if (numPixels > minNumberOfPixels)
		{
			lineY = yTotal / totalWeight + 0.5;
			guiImage.at<cv::Vec3b>((int)lineY, x) = cv::Vec3b(255, 255, 0);
			ray = cameraModel.unprojectAndUndistort(Eigen::Array2d((double)x + 0.5, lineY)).matrix().homogeneous().normalized();
			intersection = ray * d / (ray.dot(normal));

			recoImage.at<cv::Vec3d>((int)lineY, x) += cv::Vec3d(intersection.x(), intersection.y(), intersection.z());
			recoWeightsImage.at<double>((int)lineY, x) += 1.0;
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
	Eigen::Array2Xd corners = Cvl::ChessboardCornerDetector::findCorners(image, cornersPerRow, cornersPerCol, true);
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
	int const numInitImages = 20;
	int diffThreshold = 15;
	int dilateIters = 3;
	int erodeIters = 3;
	
	// template points
	size_t const cornersPerRow = 3;
	size_t const cornersPerCol = 8;
	double const squareLength = 20; // mm
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
	cv::Mat coloredImage;
	cv::Mat greyImage;
	cv::Mat planesGreyImage;	
	cv::Mat planesColoredImage;
	cv::Mat planesImageLeft;
	cv::Mat planesImageRight;
	cv::Mat objectGreyImage;
	cv::Mat objectColoredImage;
	cv::Mat diffImage;
	cv::Mat recoImage;
	cv::Mat recoWeightsImage;
	cv::Mat objectMaskColored;
	cv::Mat objectMask;

	bool success = false;
	double error = 0.0;
	Eigen::Affine3d leftModelView;
	Eigen::Affine3d rightModelView;

	//--------------- 1. PLANE RECONSTRUCTION ----------------------------
	std::cout << "Place the intersection of the two planes in the middle of the image and press space. Do not place any other objects." << std::endl;
	while (cv::waitKey(30) != 32)
	{
		cap >> coloredImage;
		cv::equalizeHist(coloredImage, coloredImage); // ?????????????????????????????????????????????
		cv::line(coloredImage, cv::Point(coloredImage.cols/2, 0), cv::Point(coloredImage.cols / 2, coloredImage.rows), 255);
		cv::imshow("Place the two planes and press space.", coloredImage);
	}
	int imageWidthHalf = coloredImage.cols / 2;
	cv::destroyAllWindows();

	planesGreyImage = cv::Mat::zeros(coloredImage.size(), CV_8UC1);
	planesColoredImage = cv::Mat::zeros(coloredImage.size(), CV_8UC3);
	for (int i = 0; i < numInitImages; ++i)
	{
		cap >> coloredImage;
		cv::cvtColor(coloredImage, greyImage, cv::COLOR_BGR2GRAY);
		planesGreyImage += greyImage / numInitImages;
		planesColoredImage += coloredImage / numInitImages;
	}
	planesImageLeft = planesGreyImage.clone();
	planesImageLeft.colRange(imageWidthHalf, coloredImage.cols).setTo(0);
	planesImageRight = planesGreyImage.clone();
	planesImageRight.colRange(0, imageWidthHalf).setTo(0);

	std::tie(success, error, leftModelView) = calculateModelView(planesImageLeft, cameraModel, templatePoints, cornersPerRow, cornersPerCol);
	if (!success || error>maxError)
	{
		std::cout << "Left plane could not be reconstructed." << std::endl;
		return 0;
	}
	std::tie(success, error, rightModelView) = calculateModelView(planesImageRight, cameraModel, templatePoints, cornersPerRow, cornersPerCol);
	if (!success || error>maxError)
	{
		std::cout << "Right plane could not be reconstructed." << std::endl;
		return 0;
	}
	std::cout << "Plane reconstruction was successful." << std::endl;
	
	//--------------- 2. SET A CLEAN IMAGE ----------------------------
	std::cout << "Place the object now an press space." << std::endl;
	while (cv::waitKey(30) != 32)
	{
		cap >> coloredImage;
		cv::imshow("Place the object and press space.", coloredImage);
	}
	cv::destroyAllWindows();
	objectGreyImage = cv::Mat::zeros(coloredImage.size(), CV_8UC1);
	objectColoredImage = cv::Mat::zeros(coloredImage.size(), CV_8UC3);
	for (int i = 0; i < numInitImages; ++i)
	{
		cap >> coloredImage;
		cv::cvtColor(coloredImage, greyImage, cv::COLOR_BGR2GRAY);
		objectGreyImage += greyImage / numInitImages;
		objectColoredImage += coloredImage / numInitImages;
	}

	//--------------- 3. CREATE THE OBJECT MASK ----------------------------
	cv::namedWindow("Mask");
	cv::createTrackbar("Threshold", "Mask", &diffThreshold, 200);
	cv::createTrackbar("Dilate Iters", "Mask", &dilateIters, 100);
	cv::createTrackbar("Erode Iters", "Mask", &erodeIters, 100);
	cv::Mat planeHSV;
	cv::Mat objectHSV;
	cv::cvtColor(objectColoredImage, objectHSV, cv::COLOR_BGR2HSV);
	cv::cvtColor(planesColoredImage, planeHSV, cv::COLOR_BGR2HSV);
	objectMask = cv::Mat::zeros(objectGreyImage.size(), CV_8UC1);
	cv::Vec3b c1;
	cv::Vec3b c2;
	while (cv::waitKey(30) != 32)
	{
		cv::absdiff(planeHSV, objectHSV, objectMaskColored);
		cv::Mat channels[3];
		cv::split(objectMaskColored, channels);

		cv::imshow("H", channels[0]);
		cv::imshow("S", channels[1]);
		cv::imshow("V", channels[2]);

		//cvtColor(objectMaskColored, objectMask, cv::COLOR_BGR2GRAY);
		//cv::threshold(objectMask, objectMask, diffThreshold, 255, cv::THRESH_BINARY);

		//cv::dilate(objectMask, objectMask, cv::Mat(), cv::Point(-1, -1), dilateIters);
		//cv::erode(objectMask, objectMask, cv::Mat(), cv::Point(-1, -1), erodeIters);
		//cv::imshow("Mask", objectMask);
		
		// only max value
		//for (int x = 0; x < objectMask.cols; ++x)
		//{
		//	for (int y = 0; y < objectMask.rows; ++y)
		//	{
		//		c1 = 
		//	}
		//}
	}


	//--------------- 3. SCAN ----------------------------
	std::cout << "Now you can scan your object." << std::endl;
	recoImage = cv::Mat::zeros(objectGreyImage.rows, objectGreyImage.cols, CV_64FC3);
	recoWeightsImage = cv::Mat::zeros(objectGreyImage.rows, objectGreyImage.cols, CV_64FC1);

	while (cv::waitKey(30) != 27)
	{
		// convert to HSV??
		cap >> coloredImage;
		cvtColor(coloredImage, greyImage, cv::COLOR_BGR2GRAY);
		cv::absdiff(greyImage, objectGreyImage, diffImage);
		//cv::GaussianBlur(diffImage, diffImage, cv::Size(0, 0), 2.0);
		reconstructImage(diffImage, objectMask, leftModelView, rightModelView, cameraModel, recoImage, recoWeightsImage, coloredImage);
		cv::imshow("Scan", coloredImage);
	}

	// write obj file
	std::ofstream objFile("scan.obj");
	if (objFile.is_open())
	{
		cv::Vec3d p;
		cv::Vec3b c;
		double w;
		for (int y = 0; y < recoImage.rows; ++y)
		{
			for (int x = 0; x < recoImage.rows; ++x)
			{
				w = recoWeightsImage.at<double>(y, x);
				if (w > 0.0)
				{
					p = recoImage.at<cv::Vec3d>(y, x);
					c = objectColoredImage.at<cv::Vec3b>(y, x);
					objFile << "v " << p(0) / w << " " << p(1) / w << " " << p(2) / w << " " << c(0)/255.0 << " " << c(1) / 255.0 << " " << c(2) / 255.0 << " \n";
				}
			}
		}
		objFile.close();
	}

	return 0;
}


