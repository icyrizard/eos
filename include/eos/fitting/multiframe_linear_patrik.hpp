/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/multiframe_linear.hpp
 *
 * Copyright 2016 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#ifndef MULTIFRAME_LINEAR_PATRIK_HPP_
#define MULTIFRAME_LINEAR_PATRIK_HPP_

#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/fitting/linear_shape_fitting.hpp"
#include "eos/fitting/blendshape_fitting.hpp"
#include "eos/fitting/contour_correspondence.hpp"
#include "eos/fitting/nonlinear_camera_estimation.hpp"

//#include "glm/gtc/matrix_transform.hpp"

#include "opencv2/core/core.hpp"

#include <vector>
#include <iostream>

namespace eos {
	namespace fitting {

// I think this function can be split in half - the setup, and the actual solving of the system? And then reuse parts of it in _multi?
inline std::vector<float> fit_shape_to_landmarks_linear_multi(morphablemodel::MorphableModel morphable_model, std::vector<cv::Mat> affine_camera_matrix, std::vector<std::vector<cv::Vec2f>> landmarks, std::vector<std::vector<int>> vertex_ids, std::vector<cv::Mat> base_face=std::vector<cv::Mat>(), float lambda=3.0f, boost::optional<int> num_coefficients_to_fit=boost::optional<int>(), boost::optional<float> detector_standard_deviation=boost::optional<float>(), boost::optional<float> model_standard_deviation=boost::optional<float>())
{
	using cv::Mat;
	assert(affine_camera_matrix.size() == landmarks.size() && landmarks.size() == vertex_ids.size()); // same number of instances (i.e. images/frames) for each of them

	int num_coeffs_to_fit = num_coefficients_to_fit.get_value_or(morphable_model.get_shape_model().get_num_principal_components());
	int num_images = affine_camera_matrix.size();

	int total_num_landmarks_dimension = 0;
	for (auto&& l : landmarks) {
		total_num_landmarks_dimension += l.size();
	}

	// $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$ associated with the $N$ feature points
	// And we insert a row of zeros after every third row, resulting in matrix $\hat{V}_h \in R^{4N\times m-1}$:
	Mat V_hat_h = Mat::zeros(4 * total_num_landmarks_dimension, num_coeffs_to_fit, CV_32FC1);
	int V_hat_h_row_index = 0;
	// Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C (P_Affine, affine_camera_matrix) is placed on the diagonal:
	Mat P = Mat::zeros(3 * total_num_landmarks_dimension, 4 * total_num_landmarks_dimension, CV_32FC1);
	int P_index = 0;
	Mat Sigma = Mat::zeros(3 * total_num_landmarks_dimension, 3 * total_num_landmarks_dimension, CV_32FC1);
	int Sigma_index = 0; // this runs the same as P_index
	Mat Omega;
	// The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
	Mat y = Mat::ones(3 * total_num_landmarks_dimension, 1, CV_32FC1);
	int y_index = 0; // also runs the same as P_index. Should rename to "running_index"?
	// The mean, with an added homogeneous coordinate (x_1, y_1, z_1, 1, x_2, ...)^t
	Mat v_bar = Mat::ones(4 * total_num_landmarks_dimension, 1, CV_32FC1);
	int v_bar_index = 0; // also runs the same as P_index. But be careful, if I change it to be only 1 variable, only increment it once! :-)
						 // Well I think that would make it a bit messy since we need to increment inside the for (landmarks...) loop. Try to refactor some other way.

	for (int k = 0; k < num_images; ++k)
	{
		// For each image we have, set up the equations and add it to the matrices:
		assert(landmarks[k].size() == vertex_ids[k].size()); // has to be valid for each img
		
		int num_landmarks = static_cast<int>(landmarks[k].size());

		if (base_face[k].empty())
		{
			base_face[k] = morphable_model.get_shape_model().get_mean();
		}

		// $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$ associated with the $N$ feature points
		// And we insert a row of zeros after every third row, resulting in matrix $\hat{V}_h \in R^{4N\times m-1}$:
		//Mat V_hat_h = Mat::zeros(4 * num_landmarks, num_coeffs_to_fit, CV_32FC1);
		for (int i = 0; i < num_landmarks; ++i) {
			Mat basis_rows = morphable_model.get_shape_model().get_normalised_pca_basis(vertex_ids[k][i]); // In the paper, the not-normalised basis might be used? I'm not sure, check it. It's even a mess in the paper. PH 26.5.2014: I think the normalised basis is fine/better.
			//basisRows.copyTo(V_hat_h.rowRange(rowIndex, rowIndex + 3));
			basis_rows.colRange(0, num_coeffs_to_fit).copyTo(V_hat_h.rowRange(V_hat_h_row_index, V_hat_h_row_index + 3));
			V_hat_h_row_index += 4; // replace 3 rows and skip the 4th one, it has all zeros
		}
		// Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C (P_Affine, affine_camera_matrix) is placed on the diagonal:
		//Mat P = Mat::zeros(3 * num_landmarks, 4 * num_landmarks, CV_32FC1);
		for (int i = 0; i < num_landmarks; ++i) {
			Mat submatrix_to_replace = P.colRange(4 * P_index, (4 * P_index) + 4).rowRange(3 * P_index, (3 * P_index) + 3);
			affine_camera_matrix[k].copyTo(submatrix_to_replace);
			++P_index;
		}
		// The variances: Add the 2D and 3D standard deviations.
		// If the user doesn't provide them, we choose the following:
		// 2D (detector) standard deviation: In pixel, we follow [1] and choose sqrt(3) as the default value.
		// 3D (model) variance: 0.0f. It only makes sense to set it to something when we have a different variance for different vertices.
		// The 3D variance has to be projected to 2D (for details, see paper [1]) so the units do match up.
		float sigma_squared_2D = std::pow(detector_standard_deviation.get_value_or(std::sqrt(3.0f)), 2) + std::pow(model_standard_deviation.get_value_or(0.0f), 2);
		//Mat Sigma = Mat::zeros(3 * num_landmarks, 3 * num_landmarks, CV_32FC1);
		for (int i = 0; i < 3 * num_landmarks; ++i) {
			Sigma.at<float>(Sigma_index, Sigma_index) = 1.0f / std::sqrt(sigma_squared_2D); // the higher the sigma_squared_2D, the smaller the diagonal entries of Sigma will be
			++Sigma_index;
		}
		//Mat Omega = Sigma.t() * Sigma; // just squares the diagonal
		// => moved outside the loop
		
		// The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
		//Mat y = Mat::ones(3 * num_landmarks, 1, CV_32FC1);
		for (int i = 0; i < num_landmarks; ++i) {
			y.at<float>(3 * y_index, 0) = landmarks[k][i][0];
			y.at<float>((3 * y_index) + 1, 0) = landmarks[k][i][1];
			//y.at<float>((3 * i) + 2, 0) = 1; // already 1, stays (homogeneous coordinate)
			++y_index;
		}
		// The mean, with an added homogeneous coordinate (x_1, y_1, z_1, 1, x_2, ...)^t
		//Mat v_bar = Mat::ones(4 * num_landmarks, 1, CV_32FC1);
		for (int i = 0; i < num_landmarks; ++i) {
			//cv::Vec4f model_mean = morphable_model.get_shape_model().get_mean_at_point(vertex_ids[i]);
			cv::Vec4f model_mean(base_face[k].at<float>(vertex_ids[k][i] * 3), base_face[k].at<float>(vertex_ids[k][i] * 3 + 1), base_face[k].at<float>(vertex_ids[k][i] * 3 + 2), 1.0f);
			v_bar.at<float>(4 * v_bar_index, 0) = model_mean[0];
			v_bar.at<float>((4 * v_bar_index) + 1, 0) = model_mean[1];
			v_bar.at<float>((4 * v_bar_index) + 2, 0) = model_mean[2];
			//v_bar.at<float>((4 * i) + 3, 0) = 1; // already 1, stays (homogeneous coordinate)
			++v_bar_index;
			// note: now that a Vec4f is returned, we could use copyTo?
		}
	}
	Omega = Sigma.t() * Sigma; // moved outside the loop. But can do even more efficiently anyway.

	// Bring into standard regularised quadratic form with diagonal distance matrix Omega
	Mat A = P * V_hat_h; // camera matrix times the basis
	Mat b = P * v_bar - y; // camera matrix times the mean, minus the landmarks.
	//Mat c_s; // The x, we solve for this! (the variance-normalised shape parameter vector, $c_s = [a_1/sigma_{s,1} , ..., a_m-1/sigma_{s,m-1}]^t$
	//int numShapePc = morphableModel.getShapeModel().getNumberOfPrincipalComponents();
	const int num_shape_pc = num_coeffs_to_fit;
	Mat AtOmegaA = A.t() * Omega * A;
	Mat AtOmegaAReg = AtOmegaA + lambda * Mat::eye(num_shape_pc, num_shape_pc, CV_32FC1);

	// Invert (and perform some sanity checks) using Eigen:
/*	using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	Eigen::Map<RowMajorMatrixXf> AtOmegaAReg_Eigen(AtOmegaAReg.ptr<float>(), AtOmegaAReg.rows, AtOmegaAReg.cols);
	Eigen::FullPivLU<RowMajorMatrixXf> luOfAtOmegaAReg(AtOmegaAReg_Eigen); // Calculate the full-pivoting LU decomposition of the regularized AtA. Note: We could also try FullPivHouseholderQR if our system is non-minimal (i.e. there are more constraints than unknowns).
	auto rankOfAtOmegaAReg = luOfAtOmegaAReg.rank();
	bool isAtOmegaARegInvertible = luOfAtOmegaAReg.isInvertible();
	float threshold = std::abs(luOfAtOmegaAReg.maxPivot()) * luOfAtOmegaAReg.threshold(); // originaly "2 * ..." but I commented it out
	RowMajorMatrixXf AtARegInv_EigenFullLU = luOfAtOmegaAReg.inverse(); // Note: We should use ::solve() instead
	Mat AtOmegaARegInvFullLU(AtARegInv_EigenFullLU.rows(), AtARegInv_EigenFullLU.cols(), CV_32FC1, AtARegInv_EigenFullLU.data()); // create an OpenCV Mat header for the Eigen data
*/
	// Solve using OpenCV:
	Mat c_s; // Note/Todo: We get coefficients ~ N(0, sigma) I think. They are not multiplied with the eigenvalues.
	bool non_singular = cv::solve(AtOmegaAReg, -A.t() * Omega.t() * b, c_s, cv::DECOMP_SVD); // DECOMP_SVD calculates the pseudo-inverse if the matrix is not invertible.
	// Because we're using SVD, non_singular will always be true. If we were to use e.g. Cholesky, we could return an expected<T>.

/*	using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	Eigen::Map<RowMajorMatrixXf> P_Eigen(P.ptr<float>(), P.rows, P.cols);
	Eigen::Map<RowMajorMatrixXf> V_hat_h_Eigen(V_hat_h.ptr<float>(), V_hat_h.rows, V_hat_h.cols);
	Eigen::Map<RowMajorMatrixXf> v_bar_Eigen(v_bar.ptr<float>(), v_bar.rows, v_bar.cols);
	Eigen::Map<RowMajorMatrixXf> y_Eigen(y.ptr<float>(), y.rows, y.cols);
	Eigen::MatrixXf A_Eigen = P_Eigen * V_hat_h_Eigen;
	Eigen::MatrixXf res = (A_Eigen.transpose() * A_Eigen + lambda * Eigen::MatrixXf::Identity(num_coeffs_to_fit, num_coeffs_to_fit)).householderQr().solve(-A_Eigen.transpose() * (P_Eigen * v_bar_Eigen - y_Eigen));
	Mat c_s(res.rows(), res.cols(), CV_32FC1, res.data()); // Note: Could also use LDLT.
*/
	return std::vector<float>(c_s);
};

std::vector<cv::Mat> fit_shape_multi(std::vector<cv::Mat> affine_camera_matrix, eos::morphablemodel::MorphableModel morphable_model, std::vector<eos::morphablemodel::Blendshape> blendshapes, std::vector<std::vector<cv::Vec2f>> image_points, std::vector<std::vector<int>> vertex_indices, float lambda, boost::optional<int> num_coefficients_to_fit, std::vector<float>& pca_shape_coefficients, std::vector<std::vector<float>>& blendshape_coefficients)
{
	using cv::Mat;
	using std::vector;
	// asserts... copy from fit_shape_to_landmarks_linear_multi?
	int num_images = affine_camera_matrix.size();
	
	Mat blendshapes_as_basis(blendshapes[0].deformation.rows, blendshapes.size(), CV_32FC1); // assert blendshapes.size() > 0 and all of them have same number of rows, and 1 col
	for (int i = 0; i < blendshapes.size(); ++i)
	{
		blendshapes[i].deformation.copyTo(blendshapes_as_basis.col(i));
	}

	vector<vector<float>> last_blendshape_coeffs, current_blendshape_coeffs; // this might be vector<vector<float>>? One for each frame?
	vector<float> last_pca_coeffs, current_pca_coeffs; // this will stay ONE single vector
	current_blendshape_coeffs.resize(num_images);
	for (auto&& cbc : current_blendshape_coeffs) {
		cbc.resize(blendshapes.size()); // starting values t_0, all zeros
	}
	// no starting values for current_pca_coeffs required, since we start with the shape fitting, and cv::norm of an empty vector is 0.
	vector<Mat> combined_shape;
	combined_shape.resize(num_images);

	double bs_error = std::numeric_limits<double>::max();
	do // run at least once:
	{
		last_blendshape_coeffs = current_blendshape_coeffs;
		last_pca_coeffs = current_pca_coeffs;
		// Estimate the PCA shape coefficients with the current blendshape coefficients (0 in the first iteration):
		vector<Mat> means_plus_blendshapes;
		for (int img = 0; img < num_images; ++img)
		{
			Mat mean_plus_blendshapes = morphable_model.get_shape_model().get_mean() + blendshapes_as_basis * Mat(last_blendshape_coeffs[img]);
			means_plus_blendshapes.push_back(mean_plus_blendshapes);
		}
		current_pca_coeffs = fit_shape_to_landmarks_linear_multi(morphable_model, affine_camera_matrix, image_points, vertex_indices, means_plus_blendshapes, lambda, num_coefficients_to_fit);

		// Estimate the blendshape coefficients with the current PCA model estimate:
		Mat pca_model_shape = morphable_model.get_shape_model().draw_sample(current_pca_coeffs);
		// Multi: actually, the expression fitting we keep like that: Use the current global shape and estimate expression BS for each frame separately. There's no use in doing that jointly for all frames.
		for (int img = 0; img < num_images; ++img)
		{
			current_blendshape_coeffs[img] = eos::fitting::fit_blendshapes_to_landmarks_nnls(blendshapes, pca_model_shape, affine_camera_matrix[img], image_points[img], vertex_indices[img], 0.0f);
		}

		for (int img = 0; img < num_images; ++img) { // just for debug purposes here.
			combined_shape[img] = pca_model_shape + blendshapes_as_basis * Mat(current_blendshape_coeffs[img]); // Note/Todo: There's actually no need to do this in the loop here? We can just do it once, at the end?
		}

		vector<double> bs_diffs; // the abs error
		for (int img = 0; img < num_images; ++img) {
			auto nc = cv::norm(current_blendshape_coeffs[img]);
			auto nl = cv::norm(last_blendshape_coeffs[img]);
			auto d = std::abs(nc - nl);
			bs_diffs.push_back(d);
		}
		double minv = 0; double maxv = 0;
		cv::minMaxLoc(bs_diffs, &minv, &maxv);
		bs_error = maxv;
	} while (std::abs(cv::norm(current_pca_coeffs) - cv::norm(last_pca_coeffs)) >= 0.01 || bs_error >= 0.01);
	// have to change this loop: norm over all frames or something like that? (for pca_coeffs, it doesn't change - only one set!)

	pca_shape_coefficients = current_pca_coeffs;
	blendshape_coefficients = current_blendshape_coeffs;
	return combined_shape;
};

	} /* namespace fitting */
} /* namespace eos */

#endif /* MULTIFRAME_LINEAR_HPP_ */
