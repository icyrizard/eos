#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <complex>
#include <iostream>
#include <omp.h>

typedef std::complex<double> complex;

/**
 * Calculates distance between two images, returns -1 if dims are not the same.
 *
 * @param one
 * @param two
 * @return
 */
float calculate_distance_images(cv::Mat one, cv::Mat two) {

	int width = one.cols;
	int height = one.rows;

	if(width != two.cols || height != two.rows) {
		return -1;
	}

	float distance = 0.0;

	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width -1; x++) {
			distance += fabs(one.at<uchar>(y, x) - two.at<uchar>(y, x));
		}
	}

	return distance;
}

void cpu(cv::Mat A, cv::Mat Anew) {
	int n = A.rows;
	int m = A.cols;

	float error = 1000;
	float tol = 30;
	int iter = 0;
	int iter_max = 100;

	auto t1 = std::chrono::high_resolution_clock::now();
	while (error > tol && iter < iter_max)
	{
		error = 0.0;
#pragma omp parallel num_threads(4)
		{
#pragma omp for reduction(max:error)
			for (int j = 1; j < n - 1; j++)
			{
				for (int i = 1; i < m - 1; i++)
				{
					Anew.at<uchar>(j, i) = 0.25 * (A.at<uchar>(j, i + 1) + A.at<uchar>(j, i - 1)
						+ A.at<uchar>(j - 1, i) + A.at<uchar>(j + 1, i));
					error = fmax(error, fabs(Anew.at<uchar>(j, i) - A.at<uchar>(j, i)));
				}
			}
		}

		iter++;
	}
}

void gpu(cv::Mat A, cv::Mat Anew) {
	int n = A.rows;
	int m = A.cols;

	float error = 1000;
	float tol = 30;
	int iter = 0;
	int iter_max = 100;

	auto t1 = std::chrono::high_resolution_clock::now();

	while ( error > tol && iter < iter_max )
	{
		error = 0.0;
#pragma omp target
		{
#pragma omp parallel for reduction(max:error)
			for (int j = 1; j < n - 1; j++)
			{
				for (int i = 1; i < m - 1; i++)
				{
					Anew.at<uchar>(j, i) = 0.25 * (A.at<uchar>(j, i + 1) + A.at<uchar>(j, i - 1)
						+ A.at<uchar>(j - 1, i) + A.at<uchar>(j + 1, i));
					error = fmax(error, fabs(Anew.at<uchar>(j, i) - A.at<uchar>(j, i)));
				}
			}
			iter++;
		}
	}
}

int main(int argc, char** argv)
{

	cv::Mat original = cv::imread("/data/img.jpg");
	cv::cvtColor(original, original, CV_BGR2GRAY);

	cv::Mat gpuImage = original.clone();
	cv::Mat gpuImageNew = original.clone();

	auto t1 = std::chrono::high_resolution_clock::now();
	gpu(gpuImage, gpuImageNew);
	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "gpu time: "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
			  << "ms" << std::endl;

	cv::Mat cpuImage = original.clone();
	cv::Mat cpuImageNew = original.clone();

	t1 = std::chrono::high_resolution_clock::now();
	cpu(cpuImage, cpuImageNew);
	t2 = std::chrono::high_resolution_clock::now();

	float distance = calculate_distance_images(gpuImageNew, cpuImageNew);
	std::cout << "cpu time: "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
			  << "ms" << std::endl;

	std::cout << "Distance gpu - cpu: " << distance << std::endl;

	cv::imshow("gpu", gpuImageNew);
	cv::imshow("cpu", cpuImageNew);
	cv::waitKey(0);
}
