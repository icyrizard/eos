/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/fitting.hpp
 *
 * Copyright 2015 Patrik Huber
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

#ifndef FITTING_HPP_
#define FITTING_HPP_

#include "eos/video/Keyframe.hpp"
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/landmark_utils.hpp"
#include "eos/core/Mesh.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/morphablemodel/EdgeTopology.hpp"
#include "eos/fitting/orthographic_camera_estimation_linear.hpp"
#include "eos/fitting/linear_shape_fitting.hpp"
#include "eos/fitting/blendshape_fitting.hpp"
#include "eos/fitting/contour_correspondence.hpp"
#include "eos/fitting/closest_edge_fitting.hpp"
#include "eos/fitting/RenderingParameters.hpp"

#include "opencv2/core/core.hpp"

#include <cassert>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace eos {
	namespace fitting {

/**
 * Convenience function that fits the shape model and expression blendshapes to
 * landmarks. Makes the fitted PCA shape and blendshape coefficients accessible
 * via the out parameters \p pca_shape_coefficients and \p blendshape_coefficients.
 * It iterates PCA-shape and blendshape fitting until convergence
 * (usually it converges within 5 to 10 iterations).
 *
 * See fit_shape_model(cv::Mat, eos::morphablemodel::MorphableModel, std::vector<eos::morphablemodel::Blendshape>, std::vector<cv::Vec2f>, std::vector<int>, float lambda)
 * for a simpler overload that just returns the shape instance.
 *
 * @param[in] affine_camera_matrix The estimated pose as a 3x4 affine camera matrix that is used to fit the shape.
 * @param[in] morphable_model The 3D Morphable Model used for the shape fitting.
 * @param[in] blendshapes A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
 * @param[in] image_points 2D landmarks from an image to fit the model to.
 * @param[in] vertex_indices The vertex indices in the model that correspond to the 2D points.
 * @param[in] lambda Regularisation parameter of the PCA shape fitting.
 * @param[in] num_coefficients_to_fit How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or boost::none to fit all coefficients.
 * @param[out] pca_shape_coefficients Output parameter that will contain the resulting pca shape coefficients.
 * @param[out] blendshape_coefficients Output parameter that will contain the resulting blendshape coefficients.
 * @return The fitted model shape instance.
 */
inline Eigen::VectorXf fit_shape(cv::Mat affine_camera_matrix, const morphablemodel::MorphableModel& morphable_model, const std::vector<morphablemodel::Blendshape>& blendshapes, const std::vector<cv::Vec2f>& image_points, const std::vector<int>& vertex_indices, float lambda, boost::optional<int> num_coefficients_to_fit, std::vector<float>& pca_shape_coefficients, std::vector<float>& blendshape_coefficients)
{
	using Eigen::VectorXf;
	using Eigen::MatrixXf;

	MatrixXf blendshapes_as_basis = morphablemodel::to_matrix(blendshapes);

	std::vector<float> last_blendshape_coeffs, current_blendshape_coeffs;
	std::vector<float> last_pca_coeffs, current_pca_coeffs;
	current_blendshape_coeffs.resize(blendshapes.size()); // starting values t_0, all zeros
	// no starting values for current_pca_coeffs required, since we start with the shape fitting, and cv::norm of an empty vector is 0.

	VectorXf combined_shape;
	do // run at least once:
	{
		last_blendshape_coeffs = current_blendshape_coeffs;
		last_pca_coeffs = current_pca_coeffs;
		// Estimate the PCA shape coefficients with the current blendshape coefficients (0 in the first iteration):
		VectorXf mean_plus_blendshapes = morphable_model.get_shape_model().get_mean() + blendshapes_as_basis * Eigen::Map<const Eigen::VectorXf>(last_blendshape_coeffs.data(), last_blendshape_coeffs.size());
		current_pca_coeffs = fitting::fit_shape_to_landmarks_linear(morphable_model, affine_camera_matrix, image_points, vertex_indices, mean_plus_blendshapes, lambda, num_coefficients_to_fit);

		// Estimate the blendshape coefficients with the current PCA model estimate:
		VectorXf pca_model_shape = morphable_model.get_shape_model().draw_sample(current_pca_coeffs);
		current_blendshape_coeffs = fitting::fit_blendshapes_to_landmarks_nnls(blendshapes, pca_model_shape, affine_camera_matrix, image_points, vertex_indices);

		combined_shape = pca_model_shape + blendshapes_as_basis * Eigen::Map<const Eigen::VectorXf>(current_blendshape_coeffs.data(), current_blendshape_coeffs.size()); // Todo/Note: Could move outside the loop, not needed in here actually
	} while (std::abs(cv::norm(current_pca_coeffs) - cv::norm(last_pca_coeffs)) >= 0.01 || std::abs(cv::norm(current_blendshape_coeffs) - cv::norm(last_blendshape_coeffs)) >= 0.01);

	pca_shape_coefficients = current_pca_coeffs;
	blendshape_coefficients = current_blendshape_coeffs;

	return combined_shape;
};

/**
 * Convenience function that fits the shape model and expression blendshapes to
 * landmarks. It iterates PCA-shape and blendshape fitting until convergence
 * (usually it converges within 5 to 10 iterations).
 *
 * @param[in] affine_camera_matrix The estimated pose as a 3x4 affine camera matrix that is used to fit the shape.
 * @param[in] morphable_model The 3D Morphable Model used for the shape fitting.
 * @param[in] blendshapes A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
 * @param[in] image_points 2D landmarks from an image to fit the model to.
 * @param[in] vertex_indices The vertex indices in the model that correspond to the 2D points.
 * @param[in] lambda Regularisation parameter of the PCA shape fitting.
 * @param[in] num_coefficients_to_fit How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or boost::none to fit all coefficients.
 * @return The fitted model shape instance.
 */
inline Eigen::VectorXf fit_shape(cv::Mat affine_camera_matrix, const morphablemodel::MorphableModel& morphable_model, const std::vector<morphablemodel::Blendshape>& blendshapes, const std::vector<cv::Vec2f>& image_points, const std::vector<int>& vertex_indices, float lambda = 3.0f, boost::optional<int> num_coefficients_to_fit = boost::optional<int>())
{
	std::vector<float> unused;
	return fit_shape(affine_camera_matrix, morphable_model, blendshapes, image_points, vertex_indices, lambda, num_coefficients_to_fit, unused, unused);
};

/**
 * @brief Takes a LandmarkCollection of 2D landmarks and, using the landmark_mapper, finds the
 * corresponding 3D vertex indices and returns them, along with the coordinates of the 3D points.
 *
 * The function only returns points which the landmark mapper was able to convert, and skips all
 * points for which there is no mapping. Thus, the number of returned points might be smaller than
 * the number of input points.
 * All three output vectors have the same size and contain the points in the same order.
 * \c landmarks can be an eos::core::LandmarkCollection<cv::Vec2f> or an rcr::LandmarkCollection<cv::Vec2f>.
 *
 * Notes:
 * - Split into two functions, one which maps from 2D LMs to vtx_idx and returns a reduced vec of 2D LMs. And then the other one to go from vtx_idx to a vector<Vec4f>.
 * - Place in a potentially more appropriate header (shape-fitting?).
 * - Could move to detail namespace or forward-declare.
 * - \c landmarks has to be a collection of LMs, with size(), [] and Vec2f ::coordinates.
 *
 * @param[in] landmarks A LandmarkCollection of 2D landmarks.
 * @param[in] landmark_mapper A mapper which maps the 2D landmark identifiers to 3D model vertex indices.
 * @param[in] morphable_model Model to get the 3D point coordinates from.
 * @return A tuple of [image_points, model_points, vertex_indices].
 */
template<class T>
inline auto get_corresponding_pointset(const T& landmarks, const core::LandmarkMapper& landmark_mapper, const morphablemodel::MorphableModel& morphable_model)
{
	using cv::Mat;
	using std::vector;
	using cv::Vec2f;
	using cv::Vec4f;

	// These will be the final 2D and 3D points used for the fitting:
	vector<Vec4f> model_points; // the points in the 3D shape model
	vector<int> vertex_indices; // their vertex indices
	vector<Vec2f> image_points; // the corresponding 2D landmark points

	// Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
	for (int i = 0; i < landmarks.size(); ++i) {
		auto converted_name = landmark_mapper.convert(landmarks[i].name);
		if (!converted_name) { // no mapping defined for the current landmark
			continue;
		}
		int vertex_idx = std::stoi(converted_name.get());
		auto vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
		model_points.emplace_back(Vec4f(vertex.x(), vertex.y(), vertex.z(), 1.0f));
		vertex_indices.emplace_back(vertex_idx);
		image_points.emplace_back(landmarks[i].coordinates);
	}
	return std::make_tuple(image_points, model_points, vertex_indices);
};

/**
 * @brief Concatenates two std::vector's of the same type and returns the concatenated
 * vector. The elements of the second vector are appended after the first one.
 *
 * Note: Move to detail namespace? It's used for the contour fitting, but doesn't really belong here.
 *
 * @param[in] vec_a First vector.
 * @param[in] vec_b Second vector.
 * @return The concatenated vector.
 */
template <class T>
inline auto concat(const std::vector<T>& vec_a, const std::vector<T>& vec_b)
{
	std::vector<T> concatenated_vec;
	concatenated_vec.reserve(vec_a.size() + vec_b.size());
	concatenated_vec.insert(std::end(concatenated_vec), std::begin(vec_a), std::end(vec_a));
	concatenated_vec.insert(std::end(concatenated_vec), std::begin(vec_b), std::end(vec_b));
	return concatenated_vec;
};

/**
 * @brief Fit the pose (camera), shape model, and expression blendshapes to landmarks,
 * in an iterative way. Can fit to more than one set of landmarks, thus multiple images.
 *
 * Convenience function that fits pose (camera), the shape model, and expression blendshapes
 * to landmarks, in an iterative (alternating) way. It fits both sides of the face contour as well.
 *
 * If \p pca_shape_coefficients and/or \p blendshape_coefficients are given, they are used as
 * starting values in the fitting. When the function returns, they contain the coefficients from
 * the last iteration.
 *
 * Use render::Mesh fit_shape_and_pose(const morphablemodel::MorphableModel&, const std::vector<morphablemodel::Blendshape>&, const core::LandmarkCollection<cv::Vec2f>&, const core::LandmarkMapper&, int, int, const morphablemodel::EdgeTopology&, const fitting::ContourLandmarks&, const fitting::ModelContour&, int, boost::optional<int>, float).
 * for a simpler overload with reasonable defaults and no optional output.
 *
 * \p num_iterations: Results are good for even a single iteration. For single-image fitting and
 * for full convergence of all parameters, it can take up to 300 iterations. In tracking,
 * particularly if initialising with the previous frame, it works well with as low as 1 to 5
 * iterations.
 * \p edge_topology is used for the occluding-edge face contour fitting.
 * \p contour_landmarks and \p model_contour are used to fit the front-facing contour.
 *
 * Todo: Add a convergence criterion.
 *
 * @param[in] morphable_model The 3D Morphable Model used for the shape fitting.
 * @param[in] blendshapes A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
 * @param[in] landmarks 2D landmarks from an image to fit the model to.
 * @param[in] landmark_mapper Mapping info from the 2D landmark points to 3D vertex indices.
 * @param[in] image_width Width of the input image (needed for the camera model).
 * @param[in] image_height Height of the input image (needed for the camera model).
 * @param[in] edge_topology Precomputed edge topology of the 3D model, needed for fast edge-lookup.
 * @param[in] contour_landmarks 2D image contour ids of left or right side (for example for ibug landmarks).
 * @param[in] model_contour The model contour indices that should be considered to find the closest corresponding 3D vertex.
 * @param[in] num_iterations Number of iterations that the different fitting parts will be alternated for.
 * @param[in] num_shape_coefficients_to_fit How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or boost::none to fit all coefficients.
 * @param[in] lambda Regularisation parameter of the PCA shape fitting.
 * @param[in] initial_rendering_params Currently ignored (not used).
 * @param[in,out] pca_shape_coefficients If given, will be used as initial PCA shape coefficients to start the fitting. Will contain the final estimated coefficients.
 * @param[in,out] blendshape_coefficients If given, will be used as initial expression blendshape coefficients to start the fitting. Will contain the final estimated coefficients.
 * @param[out] fitted_image_points Debug parameter: Returns all the 2D points that have been used for the fitting.
 * @return The fitted model shape instance and the final pose.
 */
inline std::pair<std::vector<core::Mesh>, std::vector<fitting::RenderingParameters>> fit_shape_and_pose_multi(const morphablemodel::MorphableModel& morphable_model, const std::vector<morphablemodel::Blendshape>& blendshapes, const std::vector<core::LandmarkCollection<cv::Vec2f>>& landmarks, const core::LandmarkMapper& landmark_mapper, std::vector<int> image_width, std::vector<int> image_height, const morphablemodel::EdgeTopology& edge_topology, const fitting::ContourLandmarks& contour_landmarks, const fitting::ModelContour& model_contour, int num_iterations, boost::optional<int> num_shape_coefficients_to_fit, float lambda, boost::optional<fitting::RenderingParameters> initial_rendering_params, std::vector<float>& pca_shape_coefficients, std::vector<std::vector<float>>& blendshape_coefficients, std::vector<std::vector<cv::Vec2f>>& fitted_image_points)
{
    assert(blendshapes.size() > 0);
    assert(landmarks.size() > 0 && landmarks.size() == image_width.size() && image_width.size() == image_height.size());
    assert(num_iterations > 0); // Can we allow 0, for only the initial pose-fit?
    assert(pca_shape_coefficients.size() <= morphable_model.get_shape_model().get_num_principal_components());
    int num_images = static_cast<int>(landmarks.size());
    for (int j = 0; j < num_images; ++j) {
        assert(landmarks[j].size() >= 4);
        assert(image_width[j] > 0 && image_height[j] > 0);
    }
    // More asserts I forgot?

	using std::vector;
	using cv::Vec2f;
	using cv::Vec4f;
	using cv::Mat;
	using Eigen::VectorXf;
	using Eigen::MatrixXf;


	if (!num_shape_coefficients_to_fit)
	{
		num_shape_coefficients_to_fit = morphable_model.get_shape_model().get_num_principal_components();
	}

	if (pca_shape_coefficients.empty())
	{
		pca_shape_coefficients.resize(num_shape_coefficients_to_fit.get());
	}
	// Todo: This leaves the following case open: num_coeffs given is empty or defined, but the
	// pca_shape_coefficients given is != num_coeffs or the model's max-coeffs. What to do then? Handle & document!

    if (blendshape_coefficients.empty())
    {
        for (int j = 0; j < num_images; ++j) {
            std::vector<float> current_blendshape_coefficients;
            current_blendshape_coefficients.resize(blendshapes.size());
            blendshape_coefficients.push_back(current_blendshape_coefficients);
        }
    }

    MatrixXf blendshapes_as_basis = morphablemodel::to_matrix(blendshapes);

    // Current mesh - either from the given coefficients, or the mean:
    VectorXf current_pca_shape = morphable_model.get_shape_model().draw_sample(pca_shape_coefficients);
    vector<VectorXf> current_combined_shapes;
    vector<eos::core::Mesh> current_meshs;
    for (int j = 0; j < num_images; ++j) {
        VectorXf current_combined_shape = current_pca_shape + blendshapes_as_basis * Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(), blendshape_coefficients[j].size());
        current_combined_shapes.push_back(current_combined_shape);

        eos::core::Mesh current_mesh = morphablemodel::sample_to_mesh(current_combined_shape, morphable_model.get_color_model().get_mean(), morphable_model.get_shape_model().get_triangle_list(), morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
        current_meshs.push_back(current_mesh);
    }

    // The 2D and 3D point correspondences used for the fitting:
    vector<vector<Vec4f>> model_points; // the points in the 3D shape model of all frames
    vector<vector<int>> vertex_indices; // their vertex indices of all frames
    vector<vector<Vec2f>> image_points; // the corresponding 2D landmark points of all frames

    for (int j = 0; j < num_images; ++j) {
        vector<Vec4f> current_model_points;
        vector<int> current_vertex_indices;
        vector<Vec2f> current_image_points;

        // Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM),
        // and get the corresponding model points (mean if given no initial coeffs, from the computed shape otherwise):
        for (int i = 0; i < landmarks[j].size(); ++i) {
            auto converted_name = landmark_mapper.convert(landmarks[j][i].name);
            if (!converted_name) { // no mapping defined for the current landmark
                continue;
            }
            int vertex_idx = std::stoi(converted_name.get());
            Vec4f vertex(current_meshs[j].vertices[vertex_idx].x, current_meshs[j].vertices[vertex_idx].y, current_meshs[j].vertices[vertex_idx].z, current_meshs[j].vertices[vertex_idx].w);
            current_model_points.emplace_back(vertex);
            current_vertex_indices.emplace_back(vertex_idx);
            current_image_points.emplace_back(landmarks[j][i].coordinates);
        }

        model_points.push_back(current_model_points);
        vertex_indices.push_back(current_vertex_indices);
        image_points.push_back(current_image_points);
    }

    // Need to do an initial pose fit to do the contour fitting inside the loop.
    // We'll do an expression fit too, since face shapes vary quite a lot, depending on expressions.
    vector<fitting::RenderingParameters> rendering_params;
    for (int j = 0; j < num_images; ++j) {
        fitting::ScaledOrthoProjectionParameters current_pose = fitting::estimate_orthographic_projection_linear(image_points[j], model_points[j], true, image_height[j]);
        fitting::RenderingParameters current_rendering_params(current_pose, image_width[j], image_height[j]);
        rendering_params.push_back(current_rendering_params);

        cv::Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(current_rendering_params, image_width[j], image_height[j]);
        blendshape_coefficients[j] = fitting::fit_blendshapes_to_landmarks_nnls(blendshapes, current_pca_shape, affine_from_ortho, image_points[j], vertex_indices[j]);

        // Mesh with same PCA coeffs as before, but new expression fit (this is relevant if no initial blendshape coeffs have been given):
        current_combined_shapes[j] = current_pca_shape + morphablemodel::to_matrix(blendshapes) * Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(),blendshape_coefficients[j].size());
        current_meshs[j] = morphablemodel::sample_to_mesh(current_combined_shapes[j], morphable_model.get_color_model().get_mean(), morphable_model.get_shape_model().get_triangle_list(), morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
    }

    // The static (fixed) landmark correspondences which will stay the same throughout
    // the fitting (the inner face landmarks):
    vector<vector<int>> fixed_vertex_indices (vertex_indices);
    vector<vector<Vec2f>> fixed_image_points (image_points);

    for (int i = 0; i < num_iterations; ++i)
    {
        std::vector<cv::Mat> affine_from_orthos;
        std::vector<VectorXf> mean_plus_blendshapes;

        image_points = fixed_image_points;
        vertex_indices = fixed_vertex_indices;

        for (int j = 0; j < num_images; ++j) {

            // Given the current pose, find 2D-3D contour correspondences of the front-facing face contour:
            vector<Vec2f> image_points_contour;
            vector<int> vertex_indices_contour;
            auto yaw_angle = glm::degrees(glm::eulerAngles(rendering_params[j].get_rotation())[1]);
            // For each 2D contour landmark, get the corresponding 3D vertex point and vertex id:
            std::tie(image_points_contour, std::ignore, vertex_indices_contour) = fitting::get_contour_correspondences(landmarks[j], contour_landmarks, model_contour, yaw_angle, current_meshs[j], rendering_params[j].get_modelview(), rendering_params[j].get_projection(), fitting::get_opencv_viewport(image_width[j], image_height[j]));
            // Add the contour correspondences to the set of landmarks that we use for the fitting:
            vertex_indices[j] = fitting::concat(vertex_indices[j], vertex_indices_contour);
            image_points[j] = fitting::concat(image_points[j], image_points_contour);

            // Fit the occluding (away-facing) contour using the detected contour LMs:
            vector<Eigen::Vector2f> occluding_contour_landmarks;
            if (yaw_angle >= 0.0f) // positive yaw = subject looking to the left
            { // the left contour is the occluding one we want to use ("away-facing")
                auto contour_landmarks_ = core::filter(landmarks[j], contour_landmarks.left_contour); // Can do this outside of the loop
                std::for_each(begin(contour_landmarks_), end(contour_landmarks_), [&occluding_contour_landmarks](auto&& lm) { occluding_contour_landmarks.push_back({ lm.coordinates[0], lm.coordinates[1] }); });
            }
            else {
                auto contour_landmarks_ = core::filter(landmarks[j], contour_landmarks.right_contour);
                std::for_each(begin(contour_landmarks_), end(contour_landmarks_), [&occluding_contour_landmarks](auto&& lm) { occluding_contour_landmarks.push_back({ lm.coordinates[0], lm.coordinates[1] }); });
            }
            auto edge_correspondences = fitting::find_occluding_edge_correspondences(current_meshs[j], edge_topology, rendering_params[j], occluding_contour_landmarks, 180.0f);
            image_points[j] = fitting::concat(image_points[j], edge_correspondences.first);
            vertex_indices[j] = fitting::concat(vertex_indices[j], edge_correspondences.second);

            // Get the model points of the current mesh, for all correspondences that we've got:
            model_points[j].clear();
            for (const auto& v : vertex_indices[j])
            {
                model_points[j].push_back({ current_meshs[j].vertices[v][0], current_meshs[j].vertices[v][1], current_meshs[j].vertices[v][2], current_meshs[j].vertices[v][3] });
            }

            // Re-estimate the pose, using all correspondences:
            fitting::ScaledOrthoProjectionParameters current_pose = fitting::estimate_orthographic_projection_linear(image_points[j], model_points[j], true, image_height[j]);
            rendering_params[j] = fitting::RenderingParameters(current_pose, image_width[j], image_height[j]);

            cv::Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params[j], image_width[j], image_height[j]);
            affine_from_orthos.push_back(affine_from_ortho);

            // Estimate the PCA shape coefficients with the current blendshape coefficients:
            VectorXf current_mean_plus_blendshapes = morphable_model.get_shape_model().get_mean() + blendshapes_as_basis * Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(),blendshape_coefficients[j].size());
            mean_plus_blendshapes.push_back(current_mean_plus_blendshapes);
        }
        pca_shape_coefficients = fitting::fit_shape_to_landmarks_linear_multi(morphable_model, affine_from_orthos, image_points, vertex_indices, mean_plus_blendshapes, lambda, num_shape_coefficients_to_fit);

        // Estimate the blendshape coefficients with the current PCA model estimate:
        current_pca_shape = morphable_model.get_shape_model().draw_sample(pca_shape_coefficients);

        for (int j = 0; j < num_images; ++j) {
            blendshape_coefficients[j] = fitting::fit_blendshapes_to_landmarks_nnls(blendshapes, current_pca_shape, affine_from_orthos[j], image_points[j], vertex_indices[j]);

            current_combined_shapes[j] = current_pca_shape + blendshapes_as_basis * Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(),blendshape_coefficients[j].size());
            current_meshs[j] = morphablemodel::sample_to_mesh(current_combined_shapes[j], morphable_model.get_color_model().get_mean(), morphable_model.get_shape_model().get_triangle_list(), morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
        }
    }

    fitted_image_points = image_points;
    return { current_meshs, rendering_params }; // I think we could also work with a Mat face_instance in this function instead of a Mesh, but it would convolute the code more (i.e. more complicated to access vertices).
};

/**
 * @brief Fit the pose (camera), shape model, and expression blendshapes to landmarks,
 * in an iterative way.  Can fit to more than one set of landmarks, thus multiple images.
 *
 * Convenience function that fits pose (camera), the shape model, and expression blendshapes
 * to landmarks, in an iterative (alternating) way. It fits both sides of the face contour as well.
 *
 * If you want to access the values of shape or blendshape coefficients, or want to set starting
 * values for them, use the following overload to this function:
 * std::pair<render::Mesh, fitting::RenderingParameters> fit_shape_and_pose(const morphablemodel::MorphableModel&, const std::vector<morphablemodel::Blendshape>&, const core::LandmarkCollection<cv::Vec2f>&, const core::LandmarkMapper&, int, int, const morphablemodel::EdgeTopology&, const fitting::ContourLandmarks&, const fitting::ModelContour&, int, boost::optional<int>, float, boost::optional<fitting::RenderingParameters>, std::vector<float>&, std::vector<float>&, std::vector<cv::Vec2f>&)
 *
 * Todo: Add a convergence criterion.
 *
 * \p num_iterations: Results are good for even a single iteration. For single-image fitting and
 * for full convergence of all parameters, it can take up to 300 iterations. In tracking,
 * particularly if initialising with the previous frame, it works well with as low as 1 to 5
 * iterations.
 * \p edge_topology is used for the occluding-edge face contour fitting.
 * \p contour_landmarks and \p model_contour are used to fit the front-facing contour.
 *
 * @param[in] morphable_model The 3D Morphable Model used for the shape fitting.
 * @param[in] blendshapes A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
 * @param[in] landmarks 2D landmarks from an image to fit the model to.
 * @param[in] landmark_mapper Mapping info from the 2D landmark points to 3D vertex indices.
 * @param[in] image_width Width of the input image (needed for the camera model).
 * @param[in] image_height Height of the input image (needed for the camera model).
 * @param[in] edge_topology Precomputed edge topology of the 3D model, needed for fast edge-lookup.
 * @param[in] contour_landmarks 2D image contour ids of left or right side (for example for ibug landmarks).
 * @param[in] model_contour The model contour indices that should be considered to find the closest corresponding 3D vertex.
 * @param[in] num_iterations Number of iterations that the different fitting parts will be alternated for.
 * @param[in] num_shape_coefficients_to_fit How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or boost::none to fit all coefficients.
 * @param[in] lambda Regularisation parameter of the PCA shape fitting.
 * @return The fitted model shape instance and the final pose.
 */
inline std::pair<std::vector<core::Mesh>, std::vector<fitting::RenderingParameters>> fit_shape_and_pose_multi(const morphablemodel::MorphableModel& morphable_model, const std::vector<morphablemodel::Blendshape>& blendshapes, const std::vector<core::LandmarkCollection<cv::Vec2f>>& landmarks, const core::LandmarkMapper& landmark_mapper, std::vector<int> image_width, std::vector<int> image_height, const morphablemodel::EdgeTopology& edge_topology, const fitting::ContourLandmarks& contour_landmarks, const fitting::ModelContour& model_contour, int num_iterations = 5, boost::optional<int> num_shape_coefficients_to_fit = boost::none, float lambda = 30.0f)
{
    std::vector<float> pca_shape_coefficients;
    std::vector<std::vector<float>> blendshape_coefficients;
    std::vector<std::vector<cv::Vec2f>> fitted_image_points;

    return fit_shape_and_pose_multi(morphable_model, blendshapes, landmarks, landmark_mapper, image_width, image_height, edge_topology, contour_landmarks, model_contour, num_iterations, num_shape_coefficients_to_fit, lambda, boost::none, pca_shape_coefficients, blendshape_coefficients, fitted_image_points);
};


/**
 * @brief Fit the pose (camera), shape model, and expression blendshapes to landmarks,
 * in an iterative way.
 *
 * Convenience function that fits pose (camera), the shape model, and expression blendshapes
 * to landmarks, in an iterative (alternating) way. It fits both sides of the face contour as well.
 *
 * If \p pca_shape_coefficients and/or \p blendshape_coefficients are given, they are used as
 * starting values in the fitting. When the function returns, they contain the coefficients from
 * the last iteration.
 *
 * Use render::Mesh fit_shape_and_pose(const morphablemodel::MorphableModel&, const std::vector<morphablemodel::Blendshape>&, const core::LandmarkCollection<cv::Vec2f>&, const core::LandmarkMapper&, int, int, const morphablemodel::EdgeTopology&, const fitting::ContourLandmarks&, const fitting::ModelContour&, int, boost::optional<int>, float).
 * for a simpler overload with reasonable defaults and no optional output.
 *
 * \p num_iterations: Results are good for even a single iteration. For single-image fitting and
 * for full convergence of all parameters, it can take up to 300 iterations. In tracking,
 * particularly if initialising with the previous frame, it works well with as low as 1 to 5
 * iterations.
 * \p edge_topology is used for the occluding-edge face contour fitting.
 * \p contour_landmarks and \p model_contour are used to fit the front-facing contour.
 *
 * Todo: Add a convergence criterion.
 *
 * @param[in] morphable_model The 3D Morphable Model used for the shape fitting.
 * @param[in] blendshapes A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
 * @param[in] landmarks 2D landmarks from an image to fit the model to.
 * @param[in] landmark_mapper Mapping info from the 2D landmark points to 3D vertex indices.
 * @param[in] image_width Width of the input image (needed for the camera model).
 * @param[in] image_height Height of the input image (needed for the camera model).
 * @param[in] edge_topology Precomputed edge topology of the 3D model, needed for fast edge-lookup.
 * @param[in] contour_landmarks 2D image contour ids of left or right side (for example for ibug landmarks).
 * @param[in] model_contour The model contour indices that should be considered to find the closest corresponding 3D vertex.
 * @param[in] num_iterations Number of iterations that the different fitting parts will be alternated for.
 * @param[in] num_shape_coefficients_to_fit How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or boost::none to fit all coefficients.
 * @param[in] lambda Regularisation parameter of the PCA shape fitting.
 * @param[in] initial_rendering_params Currently ignored (not used).
 * @param[in,out] pca_shape_coefficients If given, will be used as initial PCA shape coefficients to start the fitting. Will contain the final estimated coefficients.
 * @param[in,out] blendshape_coefficients If given, will be used as initial expression blendshape coefficients to start the fitting. Will contain the final estimated coefficients.
 * @param[out] fitted_image_points Debug parameter: Returns all the 2D points that have been used for the fitting.
 * @return The fitted model shape instance and the final pose.
 */
inline std::pair<core::Mesh, fitting::RenderingParameters> fit_shape_and_pose(const morphablemodel::MorphableModel& morphable_model, const std::vector<morphablemodel::Blendshape>& blendshapes, const core::LandmarkCollection<cv::Vec2f>& landmarks, const core::LandmarkMapper& landmark_mapper, int image_width, int image_height, const morphablemodel::EdgeTopology& edge_topology, const fitting::ContourLandmarks& contour_landmarks, const fitting::ModelContour& model_contour, int num_iterations, boost::optional<int> num_shape_coefficients_to_fit, float lambda, boost::optional<fitting::RenderingParameters> initial_rendering_params, std::vector<float>& pca_shape_coefficients, std::vector<float>& blendshape_coefficients, std::vector<cv::Vec2f>& fitted_image_points)
{
    //we have to create a new vector here if the blendshape_coefficients are empty, as otherwise the new vector is not empty anymore and contains one element
    std::vector<std::vector<float>> all_blendshape_coefficients;
    if (!blendshape_coefficients.empty())
    {
        all_blendshape_coefficients = {blendshape_coefficients};
    }
    std::vector<std::vector<cv::Vec2f>> all_fitted_image_points;
    if (!fitted_image_points.empty())
    {
        all_fitted_image_points = {fitted_image_points};
    }
    std::pair<std::vector<core::Mesh>, std::vector<fitting::RenderingParameters>>  all_meshs_and_params = fit_shape_and_pose_multi( morphable_model, blendshapes, { landmarks }, landmark_mapper, { image_width }, { image_height }, edge_topology, contour_landmarks, model_contour, num_iterations, num_shape_coefficients_to_fit, lambda, initial_rendering_params, pca_shape_coefficients, all_blendshape_coefficients, all_fitted_image_points);
    blendshape_coefficients = all_blendshape_coefficients[0];
    fitted_image_points = all_fitted_image_points[0];

    return {all_meshs_and_params.first[0], all_meshs_and_params.second[0] };
}

/**
 * @brief Fit the pose (camera), shape model, and expression blendshapes to landmarks,
 * in an iterative way.
 *
 * Convenience function that fits pose (camera), the shape model, and expression blendshapes
 * to landmarks, in an iterative (alternating) way. It fits both sides of the face contour as well.
 *
 * If you want to access the values of shape or blendshape coefficients, or want to set starting
 * values for them, use the following overload to this function:
 * std::pair<render::Mesh, fitting::RenderingParameters> fit_shape_and_pose(const morphablemodel::MorphableModel&, const std::vector<morphablemodel::Blendshape>&, const core::LandmarkCollection<cv::Vec2f>&, const core::LandmarkMapper&, int, int, const morphablemodel::EdgeTopology&, const fitting::ContourLandmarks&, const fitting::ModelContour&, int, boost::optional<int>, float, boost::optional<fitting::RenderingParameters>, std::vector<float>&, std::vector<float>&, std::vector<cv::Vec2f>&)
 *
 * Todo: Add a convergence criterion.
 *
 * \p num_iterations: Results are good for even a single iteration. For single-image fitting and
 * for full convergence of all parameters, it can take up to 300 iterations. In tracking,
 * particularly if initialising with the previous frame, it works well with as low as 1 to 5
 * iterations.
 * \p edge_topology is used for the occluding-edge face contour fitting.
 * \p contour_landmarks and \p model_contour are used to fit the front-facing contour.
 *
 * @param[in] morphable_model The 3D Morphable Model used for the shape fitting.
 * @param[in] blendshapes A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
 * @param[in] landmarks 2D landmarks from an image to fit the model to.
 * @param[in] landmark_mapper Mapping info from the 2D landmark points to 3D vertex indices.
 * @param[in] image_width Width of the input image (needed for the camera model).
 * @param[in] image_height Height of the input image (needed for the camera model).
 * @param[in] edge_topology Precomputed edge topology of the 3D model, needed for fast edge-lookup.
 * @param[in] contour_landmarks 2D image contour ids of left or right side (for example for ibug landmarks).
 * @param[in] model_contour The model contour indices that should be considered to find the closest corresponding 3D vertex.
 * @param[in] num_iterations Number of iterations that the different fitting parts will be alternated for.
 * @param[in] num_shape_coefficients_to_fit How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or boost::none to fit all coefficients.
 * @param[in] lambda Regularisation parameter of the PCA shape fitting.
 * @return The fitted model shape instance and the final pose.
 */
inline std::pair<core::Mesh, fitting::RenderingParameters> fit_shape_and_pose(const morphablemodel::MorphableModel& morphable_model, const std::vector<morphablemodel::Blendshape>& blendshapes, const core::LandmarkCollection<cv::Vec2f>& landmarks, const core::LandmarkMapper& landmark_mapper, int image_width, int image_height, const morphablemodel::EdgeTopology& edge_topology, const fitting::ContourLandmarks& contour_landmarks, const fitting::ModelContour& model_contour, int num_iterations = 5, boost::optional<int> num_shape_coefficients_to_fit = boost::none, float lambda = 50.0f)
{
	std::vector<float> pca_coeffs;
	std::vector<float> blendshape_coeffs;
	std::vector<cv::Vec2f> fitted_image_points;
	return fit_shape_and_pose(morphable_model, blendshapes, landmarks, landmark_mapper, image_width, image_height, edge_topology, contour_landmarks, model_contour, num_iterations, num_shape_coefficients_to_fit, lambda, boost::none, pca_coeffs, blendshape_coeffs, fitted_image_points);
};

/**
* @brief Fit the pose (camera), shape model, and expression blendshapes to landmarks,
* in an iterative way. Can fit to more than one set of landmarks, thus multiple images.
*
* Convenience function that fits pose (camera), the shape model, and expression blendshapes
* to landmarks, in an iterative (alternating) way. It fits both sides of the face contour as well.
*
* If \p pca_shape_coefficients and/or \p blendshape_coefficients are given, they are used as
* starting values in the fitting. When the function returns, they contain the coefficients from
* the last iteration.
*
* Use render::Mesh fit_shape_and_pose(const morphablemodel::MorphableModel&, const std::vector<morphablemodel::Blendshape>&, const core::LandmarkCollection<cv::Vec2f>&, const core::LandmarkMapper&, int, int, const morphablemodel::EdgeTopology&, const fitting::ContourLandmarks&, const fitting::ModelContour&, int, boost::optional<int>, float).
* for a simpler overload with reasonable defaults and no optional output.
*
* \p num_iterations: Results are good for even a single iteration. For single-image fitting and
* for full convergence of all parameters, it can take up to 300 iterations. In tracking,
* particularly if initialising with the previous frame, it works well with as low as 1 to 5
* iterations.
* \p edge_topology is used for the occluding-edge face contour fitting.
* \p contour_landmarks and \p model_contour are used to fit the front-facing contour.
*
* Todo: Add a convergence criterion.
*
* @param[in] morphable_model The 3D Morphable Model used for the shape fitting.
* @param[in] blendshapes A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
* @param[in] landmarks 2D landmarks from an image to fit the model to.
* @param[in] landmark_mapper Mapping info from the 2D landmark points to 3D vertex indices.
* @param[in] image_width Width of the input image (needed for the camera model).
* @param[in] image_height Height of the input image (needed for the camera model).
* @param[in] edge_topology Precomputed edge topology of the 3D model, needed for fast edge-lookup.
* @param[in] contour_landmarks 2D image contour ids of left or right side (for example for ibug landmarks).
* @param[in] model_contour The model contour indices that should be considered to find the closest corresponding 3D vertex.
* @param[in] num_iterations Number of iterations that the different fitting parts will be alternated for.
* @param[in] num_shape_coefficients_to_fit How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or boost::none to fit all coefficients.
* @param[in] lambda Regularisation parameter of the PCA shape fitting.
* @param[in] initial_rendering_params Currently ignored (not used).
* @param[in,out] pca_shape_coefficients If given, will be used as initial PCA shape coefficients to start the fitting. Will contain the final estimated coefficients.
* @param[in,out] blendshape_coefficients If given, will be used as initial expression blendshape coefficients to start the fitting. Will contain the final estimated coefficients.
* @param[out] fitted_image_points Debug parameter: Returns all the 2D points that have been used for the fitting.
* @return The fitted model shape instance and the final pose.
*/
inline std::pair<std::vector<core::Mesh>, std::vector<fitting::RenderingParameters>> fit_shape_and_pose_multi(
				const morphablemodel::MorphableModel& morphable_model,
				const std::vector<morphablemodel::Blendshape>& blendshapes,
				const std::vector<core::LandmarkCollection<cv::Vec2f>>& landmarks,
				const core::LandmarkMapper& landmark_mapper,
				int image_width,
				int image_height,
				int num_images,
				const morphablemodel::EdgeTopology& edge_topology,
				const fitting::ContourLandmarks& contour_landmarks,
				const fitting::ModelContour& model_contour,
				int num_iterations,
				boost::optional<int> num_shape_coefficients_to_fit,
				float lambda,
				boost::optional<fitting::RenderingParameters> initial_rendering_params,
				std::vector<float>& pca_shape_coefficients,
				std::vector<std::vector<float>>& blendshape_coefficients,
				std::vector<std::vector<cv::Vec2f>>& fitted_image_points) {

	assert(blendshapes.size() > 0);
	assert(num_iterations > 0); // Can we allow 0, for only the initial pose-fit?
	assert(pca_shape_coefficients.size() <= morphable_model.get_shape_model().get_num_principal_components());

	using std::vector;
	using cv::Vec2f;
	using cv::Vec4f;
	using cv::Mat;
	using Eigen::VectorXf;
	using Eigen::MatrixXf;

	if (!num_shape_coefficients_to_fit) {
		num_shape_coefficients_to_fit = morphable_model.get_shape_model().get_num_principal_components();
	}

	if (pca_shape_coefficients.empty()) {
		pca_shape_coefficients.resize(num_shape_coefficients_to_fit.get());
	}

	// TODO: This leaves the following case open: num_coeffs given is empty or defined, but the
	// pca_shape_coefficients given is != num_coeffs or the model's max-coeffs. What to do then? Handle & document!

	if (blendshape_coefficients.size() < num_images) {
		for (int j = 0; j < num_images; ++j) {
			std::vector<float> current_blendshape_coefficients;
			current_blendshape_coefficients.resize(blendshapes.size());
			blendshape_coefficients.push_back(current_blendshape_coefficients);
		}
	}

	MatrixXf blendshapes_as_basis = morphablemodel::to_matrix(blendshapes);

	// Current mesh - either from the given coefficients, or the mean:
	VectorXf current_pca_shape = morphable_model.get_shape_model().draw_sample(pca_shape_coefficients);
	vector<VectorXf> current_combined_shapes;
	vector<eos::core::Mesh> current_meshs;

	for (int j = 0; j < num_images; ++j) {
		VectorXf current_combined_shape = current_pca_shape + blendshapes_as_basis * Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(), blendshape_coefficients[j].size());
		current_combined_shapes.push_back(current_combined_shape);

		eos::core::Mesh current_mesh = morphablemodel::sample_to_mesh(current_combined_shape, morphable_model.get_color_model().get_mean(), morphable_model.get_shape_model().get_triangle_list(), morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
		current_meshs.push_back(current_mesh);
	}

	// The 2D and 3D point correspondences used for the fitting:
	vector<vector<Vec4f>> model_points; // the points in the 3D shape model of all frames
	vector<vector<int>> vertex_indices; // their vertex indices of all frames
	vector<vector<Vec2f>> image_points; // the corresponding 2D landmark points of all frames

	for (int j = 0; j < num_images; ++j) {
		vector<Vec4f> current_model_points;
		vector<int> current_vertex_indices;
		vector<Vec2f> current_image_points;

		// Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM),
		// and get the corresponding model points (mean if given no initial coeffs, from the computed shape otherwise):
		for (int i = 0; i < landmarks[j].size(); ++i) {
			auto converted_name = landmark_mapper.convert(landmarks[j][i].name);
			if (!converted_name) { // no mapping defined for the current landmark
				continue;
			}
			int vertex_idx = std::stoi(converted_name.get());
			Vec4f vertex(current_meshs[j].vertices[vertex_idx].x, current_meshs[j].vertices[vertex_idx].y, current_meshs[j].vertices[vertex_idx].z, current_meshs[j].vertices[vertex_idx].w);
			current_model_points.emplace_back(vertex);
			current_vertex_indices.emplace_back(vertex_idx);
			current_image_points.emplace_back(landmarks[j][i].coordinates);
		}

		model_points.push_back(current_model_points);
		vertex_indices.push_back(current_vertex_indices);
		image_points.push_back(current_image_points);
	}

	// Need to do an initial pose fit to do the contour fitting inside the loop.
	// We'll do an expression fit too, since face shapes vary quite a lot, depending on expressions.
	vector<fitting::RenderingParameters> rendering_params;
	for (int j = 0; j < num_images; ++j) {
		fitting::ScaledOrthoProjectionParameters current_pose = fitting::estimate_orthographic_projection_linear(image_points[j], model_points[j], true, image_height);
		fitting::RenderingParameters current_rendering_params(current_pose, image_width, image_height);
		rendering_params.push_back(current_rendering_params);

		cv::Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(current_rendering_params, image_width, image_height);
		blendshape_coefficients[j] = fitting::fit_blendshapes_to_landmarks_nnls(blendshapes, current_pca_shape, affine_from_ortho, image_points[j], vertex_indices[j]);

		// Mesh with same PCA coeffs as before, but new expression fit (this is relevant if no initial blendshape coeffs have been given):
		current_combined_shapes[j] = current_pca_shape + morphablemodel::to_matrix(blendshapes) * Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(),blendshape_coefficients[j].size());
		current_meshs[j] = morphablemodel::sample_to_mesh(current_combined_shapes[j], morphable_model.get_color_model().get_mean(), morphable_model.get_shape_model().get_triangle_list(), morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
	}

	// The static (fixed) landmark correspondences which will stay the same throughout
	// the fitting (the inner face landmarks):
	vector<vector<int>> fixed_vertex_indices (vertex_indices);
	vector<vector<Vec2f>> fixed_image_points (image_points);

	std::vector<cv::Mat> affine_from_orthos;
	std::vector<VectorXf> mean_plus_blendshapes;

	image_points = fixed_image_points;
	vertex_indices = fixed_vertex_indices;

	for (int j = 0; j < num_images; ++j) {
		// Given the current pose, find 2D-3D contour correspondences of the front-facing face contour:
		vector<Vec2f> image_points_contour;
		vector<int> vertex_indices_contour;
		auto yaw_angle = glm::degrees(glm::eulerAngles(rendering_params[j].get_rotation())[1]);
		// For each 2D contour landmark, get the corresponding 3D vertex point and vertex id:
		std::tie(image_points_contour, std::ignore, vertex_indices_contour) = fitting::get_contour_correspondences(landmarks[j], contour_landmarks, model_contour, yaw_angle, current_meshs[j], rendering_params[j].get_modelview(), rendering_params[j].get_projection(), fitting::get_opencv_viewport(image_width, image_height));
		// Add the contour correspondences to the set of landmarks that we use for the fitting:
		vertex_indices[j] = fitting::concat(vertex_indices[j], vertex_indices_contour);
		image_points[j] = fitting::concat(image_points[j], image_points_contour);

		// Fit the occluding (away-facing) contour using the detected contour LMs:
		vector<Eigen::Vector2f> occluding_contour_landmarks;
		if (yaw_angle >= 0.0f) // positive yaw = subject looking to the left
		{ // the left contour is the occluding one we want to use ("away-facing")
			auto contour_landmarks_ = core::filter(landmarks[j], contour_landmarks.left_contour); // Can do this outside of the loop
			std::for_each(begin(contour_landmarks_), end(contour_landmarks_), [&occluding_contour_landmarks](auto&& lm) { occluding_contour_landmarks.push_back({ lm.coordinates[0], lm.coordinates[1] }); });
		}
		else {
			auto contour_landmarks_ = core::filter(landmarks[j], contour_landmarks.right_contour);
			std::for_each(begin(contour_landmarks_), end(contour_landmarks_), [&occluding_contour_landmarks](auto&& lm) { occluding_contour_landmarks.push_back({ lm.coordinates[0], lm.coordinates[1] }); });
		}
		auto edge_correspondences = fitting::find_occluding_edge_correspondences(current_meshs[j], edge_topology, rendering_params[j], occluding_contour_landmarks, 180.0f);
		image_points[j] = fitting::concat(image_points[j], edge_correspondences.first);
		vertex_indices[j] = fitting::concat(vertex_indices[j], edge_correspondences.second);

		// Get the model points of the current mesh, for all correspondences that we've got:
		model_points[j].clear();
		for (const auto& v : vertex_indices[j])
		{
			model_points[j].push_back({ current_meshs[j].vertices[v][0], current_meshs[j].vertices[v][1], current_meshs[j].vertices[v][2], current_meshs[j].vertices[v][3] });
		}

		// Re-estimate the pose, using all correspondences:
		fitting::ScaledOrthoProjectionParameters current_pose = fitting::estimate_orthographic_projection_linear(image_points[j], model_points[j], true, image_height);
		rendering_params[j] = fitting::RenderingParameters(current_pose, image_width, image_height);

		cv::Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params[j], image_width, image_height);
		affine_from_orthos.push_back(affine_from_ortho);

		// Estimate the PCA shape coefficients with the current blendshape coefficients:
		VectorXf current_mean_plus_blendshapes = morphable_model.get_shape_model().get_mean() + blendshapes_as_basis * Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(),blendshape_coefficients[j].size());
		mean_plus_blendshapes.push_back(current_mean_plus_blendshapes);
	}

	pca_shape_coefficients = fitting::fit_shape_to_landmarks_linear_multi(morphable_model, affine_from_orthos, image_points, vertex_indices, mean_plus_blendshapes, lambda, num_shape_coefficients_to_fit);

	// Estimate the blendshape coefficients with the current PCA model estimate:
	current_pca_shape = morphable_model.get_shape_model().draw_sample(pca_shape_coefficients);

	for (int j = 0; j < num_images; ++j) {
		blendshape_coefficients[j] = fitting::fit_blendshapes_to_landmarks_nnls(blendshapes, current_pca_shape, affine_from_orthos[j], image_points[j], vertex_indices[j]);

		current_combined_shapes[j] = current_pca_shape + blendshapes_as_basis * Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(),blendshape_coefficients[j].size());
		current_meshs[j] = morphablemodel::sample_to_mesh(current_combined_shapes[j], morphable_model.get_color_model().get_mean(), morphable_model.get_shape_model().get_triangle_list(), morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
	}

	fitted_image_points = image_points;
	return { current_meshs, rendering_params }; // I think we could also work with a Mat face_instance in this function instead of a Mesh, but it would convolute the code more (i.e. more complicated to access vertices).
};

/**
* @brief Fit the pose (camera), shape model, and expression blendshapes to landmarks,
* in an iterative way.  Can fit to more than one set of landmarks, thus multiple images.
*
* Convenience function that fits pose (camera), the shape model, and expression blendshapes
* to landmarks, in an iterative (alternating) way. It fits both sides of the face contour as well.
*
* If you want to access the values of shape or blendshape coefficients, or want to set starting
* values for them, use the following overload to this function:
* std::pair<render::Mesh, fitting::RenderingParameters> fit_shape_and_pose(const morphablemodel::MorphableModel&, const std::vector<morphablemodel::Blendshape>&, const core::LandmarkCollection<cv::Vec2f>&, const core::LandmarkMapper&, int, int, const morphablemodel::EdgeTopology&, const fitting::ContourLandmarks&, const fitting::ModelContour&, int, boost::optional<int>, float, boost::optional<fitting::RenderingParameters>, std::vector<float>&, std::vector<float>&, std::vector<cv::Vec2f>&)
*
* Todo: Add a convergence criterion.
*
* \p num_iterations: Results are good for even a single iteration. For single-image fitting and
* for full convergence of all parameters, it can take up to 300 iterations. In tracking,
* particularly if initialising with the previous frame, it works well with as low as 1 to 5
* iterations.
* \p edge_topology is used for the occluding-edge face contour fitting.
* \p contour_landmarks and \p model_contour are used to fit the front-facing contour.
*
* @param[in] morphable_model The 3D Morphable Model used for the shape fitting.
* @param[in] blendshapes A vector of blendshapes that are being fit to the landmarks in addition to the PCA model.
* @param[in] landmarks 2D landmarks from an image to fit the model to.
* @param[in] landmark_mapper Mapping info from the 2D landmark points to 3D vertex indices.
* @param[in] image_width Width of the input image (needed for the camera model).
* @param[in] image_height Height of the input image (needed for the camera model).
* @param[in] edge_topology Precomputed edge topology of the 3D model, needed for fast edge-lookup.
* @param[in] contour_landmarks 2D image contour ids of left or right side (for example for ibug landmarks).
* @param[in] model_contour The model contour indices that should be considered to find the closest corresponding 3D vertex.
* @param[in] num_iterations Number of iterations that the different fitting parts will be alternated for.
* @param[in] num_shape_coefficients_to_fit How many shape-coefficients to fit (all others will stay 0). Should be bigger than zero, or boost::none to fit all coefficients.
* @param[in] lambda Regularisation parameter of the PCA shape fitting.
* @return The fitted model shape instance and the final pose.
*/
inline std::pair<std::vector<core::Mesh>, std::vector<fitting::RenderingParameters>> fit_shape_and_pose_multi(
				const morphablemodel::MorphableModel& morphable_model, const std::vector<morphablemodel::Blendshape>& blendshapes, const std::vector<core::LandmarkCollection<cv::Vec2f>>& landmarks,
				const core::LandmarkMapper& landmark_mapper, int image_width, int image_height, const morphablemodel::EdgeTopology& edge_topology, const fitting::ContourLandmarks& contour_landmarks,
				const fitting::ModelContour& model_contour, int num_iterations = 5, boost::optional<int> num_shape_coefficients_to_fit = boost::none, float lambda = 30.0f)
{
	std::vector<float> pca_shape_coefficients;
	std::vector<std::vector<float>> blend_shape_coefficients;
	std::vector<std::vector<cv::Vec2f>> fitted_image_points;

	return fit_shape_and_pose_multi(
			morphable_model,
			blendshapes,
			landmarks,
			landmark_mapper,
			image_width,
			image_height,
			static_cast<int>(landmarks.size()),
			edge_topology,
			contour_landmarks,
			model_contour,
			num_iterations,
			num_shape_coefficients_to_fit,
			lambda,
			boost::none,
			pca_shape_coefficients,
			blend_shape_coefficients,
			fitted_image_points
	);
};

/**
 * Generate a new mesh given pca_coefficients and blend_shape_coefficients.
 * @param pca_shape_coefficients
 * @param blend_shape_coefficients
 * @param num_shape_coefficients_to_fit
 * @return
 */
inline eos::core::Mesh generate_new_mesh(
		morphablemodel::MorphableModel& morphable_model,
		std::vector<morphablemodel::Blendshape>& blendshapes,
		std::vector<float>& pca_shape_coefficients,
		std::vector<float>& blendshape_coefficients,
		unsigned int num_shape_coefficients_to_fit = 50) {

	eos::core::Mesh mesh;
	Eigen::MatrixXf blendshapes_as_basis = morphablemodel::to_matrix(blendshapes);
	Eigen::VectorXf pca_shape = morphable_model.get_shape_model().draw_sample(pca_shape_coefficients);

	// create new set of blendshape coefficients if empty:
	if(blendshape_coefficients.empty()) {
		blendshape_coefficients.resize(blendshapes.size());
	}

	Eigen::VectorXf combined_shape = pca_shape +
		blendshapes_as_basis * Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients.data(), blendshape_coefficients.size());

	mesh = morphablemodel::sample_to_mesh(
		combined_shape,
		morphable_model.get_color_model().get_mean(),
		morphable_model.get_shape_model().get_triangle_list(),
		morphable_model.get_color_model().get_triangle_list(),
		morphable_model.get_texture_coordinates());

	return mesh;
}


inline std::pair<std::vector<core::Mesh>, std::vector<fitting::RenderingParameters>> fit_shape_and_pose_multi_2(
	const morphablemodel::MorphableModel& morphable_model,
	const std::vector<morphablemodel::Blendshape>& blendshapes,
	std::vector<eos::video::Keyframe>& keyframes,
	const core::LandmarkMapper& landmark_mapper,
	int image_width,
	int image_height,
	int num_images,
	const morphablemodel::EdgeTopology& edge_topology,
	const fitting::ContourLandmarks& contour_landmarks,
	const fitting::ModelContour& model_contour,
	int num_iterations,
	boost::optional<int> num_shape_coefficients_to_fit,
	float lambda,
	boost::optional<fitting::RenderingParameters> initial_rendering_params,
	std::vector<float>& pca_shape_coefficients,
	std::vector<std::vector<float>>& blendshape_coefficients,
	std::vector<std::vector<cv::Vec2f>>& fitted_image_points) {

	assert(blendshapes.size() > 0);
	assert(num_iterations > 0); // Can we allow 0, for only the initial pose-fit?
	assert(pca_shape_coefficients.size() <= morphable_model.get_shape_model().get_num_principal_components());

	using std::vector;
	using cv::Vec2f;
	using cv::Vec4f;
	using cv::Mat;
	using Eigen::VectorXf;
	using Eigen::MatrixXf;

	if (!num_shape_coefficients_to_fit) {
		num_shape_coefficients_to_fit = morphable_model.get_shape_model().get_num_principal_components();
	}

	if (pca_shape_coefficients.empty()) {
		pca_shape_coefficients.resize(num_shape_coefficients_to_fit.get());
	}

	// TODO: This leaves the following case open: num_coeffs given is empty or defined, but the
	// pca_shape_coefficients given is != num_coeffs or the model's max-coeffs. What to do then? Handle & document!
	if (blendshape_coefficients.size() < num_images) {
		for (int j = 0; j < num_images; ++j) {
			std::vector<float> current_blendshape_coefficients;
			current_blendshape_coefficients.resize(blendshapes.size());
			blendshape_coefficients.push_back(current_blendshape_coefficients);
		}
	}

	MatrixXf blendshapes_as_basis = morphablemodel::to_matrix(blendshapes);

	// Current mesh - either from the given coefficients, or the mean:
	VectorXf current_pca_shape = morphable_model.get_shape_model().draw_sample(pca_shape_coefficients);
	vector<VectorXf> current_combined_shapes;
	vector<eos::core::Mesh> current_meshs;

	// The 2D and 3D point correspondences used for the fitting:
	vector<vector<Vec4f>> model_points; // the points in the 3D shape model of all frames.
	vector<vector<int>> vertex_indices; // their vertex indices of all frames.
	vector<vector<Vec2f>> image_points; // the corresponding 2D landmark points of all frames.

	vector<fitting::RenderingParameters> rendering_params; // list of rendering params for all frames.

	// Generate meshes from current shapes.
	for (int j = 0; j < num_images; ++j) {
		VectorXf current_combined_shape = current_pca_shape +
			blendshapes_as_basis *
				Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(), blendshape_coefficients[j].size());

		eos::core::Mesh current_mesh = morphablemodel::sample_to_mesh(
			current_combined_shape,
			morphable_model.get_color_model().get_mean(),
			morphable_model.get_shape_model().get_triangle_list(),
			morphable_model.get_color_model().get_triangle_list(),
			morphable_model.get_texture_coordinates());

		// Get the locations of the model locations of the meshes, vertex_indices and image points
		// (equal to landmark coordinates), for every image / mesh.

//		std::tie(curr_model_points, curr_vertex_indices, curr_image_points) = eos::core::get_landmark_coordinates<Vec2f, Vec4f>(
//			keyframes[j].fitting_result.landmarks, landmark_mapper, current_mesh);
		eos::core::get_landmark_coordinates_inline<Vec2f, Vec4f>(
			keyframes[j].fitting_result.landmarks,
			landmark_mapper,
			current_mesh,
			// output parameters
			model_points,
			vertex_indices,
			image_points);

		// Start constructing a list of rendering parameters needed for reconstruction.
		// Get the current points from the last added image points and model points
		auto current_pose = fitting::estimate_orthographic_projection_linear(
			image_points.back(), model_points.back(), true, image_height);

		fitting::RenderingParameters current_rendering_params(current_pose, image_width, image_height);
		rendering_params.push_back(current_rendering_params);

		// update key frame rendering params
		keyframes[j].fitting_result.rendering_parameters = current_rendering_params;

		Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(current_rendering_params, image_width, image_height);

		blendshape_coefficients[j] = fitting::fit_blendshapes_to_landmarks_nnls(
			blendshapes, current_pca_shape, affine_from_ortho, image_points.back(), vertex_indices.back()
		);

		// Mesh with same PCA coeffs as before, but new expression fit (this is relevant if no initial blendshape coeffs have been given):
		current_combined_shape = current_pca_shape +
			morphablemodel::to_matrix(blendshapes) *
				Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(), blendshape_coefficients[j].size()
				);

		current_combined_shapes.push_back(current_combined_shape);

		current_mesh = morphablemodel::sample_to_mesh(
			current_combined_shape, morphable_model.get_color_model().get_mean(),
			morphable_model.get_shape_model().get_triangle_list(),
			morphable_model.get_color_model().get_triangle_list(),
			morphable_model.get_texture_coordinates()
		);

		current_meshs.push_back(current_mesh);
	}

	// The static (fixed) landmark correspondences which will stay the same throughout
	// the fitting (the inner face landmarks):
	vector<vector<int>> fixed_vertex_indices (vertex_indices);
	vector<vector<Vec2f>> fixed_image_points (image_points);

	std::vector<cv::Mat> affine_from_orthos;
	std::vector<VectorXf> mean_plus_blendshapes;

	image_points = fixed_image_points;
	vertex_indices = fixed_vertex_indices;

	for (int j = 0; j < num_images; ++j) {
		// Given the current pose, find 2D-3D contour correspondences of the front-facing face contour:
		vector<Vec2f> image_points_contour;
		vector<int> vertex_indices_contour;

		auto curr_keyframe = keyframes[j];
		auto landmarks = curr_keyframe.fitting_result.landmarks;
		auto yaw_angle = glm::degrees(glm::eulerAngles(rendering_params[j].get_rotation())[1]);

		// For each 2D contour landmark, get the corresponding 3D vertex point and vertex id:
		std::tie(image_points_contour, std::ignore, vertex_indices_contour) = fitting::get_contour_correspondences(
			landmarks,
			contour_landmarks,
			model_contour,
			yaw_angle,
			current_meshs[j],
			rendering_params[j].get_modelview(),
			rendering_params[j].get_projection(),
			fitting::get_opencv_viewport(image_width, image_height)
		);

		// Add the contour correspondences to the set of landmarks that we use for the fitting:
		vertex_indices[j] = fitting::concat(vertex_indices[j], vertex_indices_contour);
		image_points[j] = fitting::concat(image_points[j], image_points_contour);

		// Fit the occluding (away-facing) contour using the detected contour LMs:
		vector<Eigen::Vector2f> occluding_contour_landmarks;

		// positive yaw = subject looking to the left
		if (yaw_angle >= 0.0f) {
			// the left contour is the occluding one we want to use ("away-facing")
			auto contour_landmarks_ = core::filter(landmarks, contour_landmarks.left_contour); // Can do this outside of the loop
			std::for_each(begin(contour_landmarks_), end(contour_landmarks_), [&occluding_contour_landmarks](auto&& lm) {
				occluding_contour_landmarks.push_back({ lm.coordinates[0], lm.coordinates[1] });
			});
		} else {
			auto contour_landmarks_ = core::filter(landmarks, contour_landmarks.right_contour);
			std::for_each(begin(contour_landmarks_), end(contour_landmarks_), [&occluding_contour_landmarks](auto&& lm) {
				occluding_contour_landmarks.push_back({ lm.coordinates[0], lm.coordinates[1] });
			});
		}

		auto edge_correspondences = fitting::find_occluding_edge_correspondences(
			current_meshs[j], edge_topology, rendering_params[j], occluding_contour_landmarks, 180.0f
		);

		image_points[j] = fitting::concat(image_points[j], edge_correspondences.first);
		vertex_indices[j] = fitting::concat(vertex_indices[j], edge_correspondences.second);

		// Get the model points of the current mesh, for all correspondences that we've got:
		model_points[j].clear();
		for (const auto& v : vertex_indices[j]) {
			model_points[j].push_back(
				{
					current_meshs[j].vertices[v][0],
					current_meshs[j].vertices[v][1],
					current_meshs[j].vertices[v][2],
					current_meshs[j].vertices[v][3]
				});
		}

		// Re-estimate the pose, using all correspondences:
		auto current_pose = fitting::estimate_orthographic_projection_linear(image_points[j], model_points[j], true, image_height);
		rendering_params[j] = fitting::RenderingParameters(current_pose, image_width, image_height);
		Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params[j], image_width, image_height);

		affine_from_orthos.push_back(affine_from_ortho);

		// Estimate the PCA shape coefficients with the current blendshape coefficients:
		VectorXf current_mean_plus_blendshapes = morphable_model.get_shape_model().get_mean() +
			blendshapes_as_basis * Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(),blendshape_coefficients[j].size());
		mean_plus_blendshapes.push_back(current_mean_plus_blendshapes);
	}

	pca_shape_coefficients = fitting::fit_shape_to_landmarks_linear_multi_parallel(
		morphable_model,
		affine_from_orthos,
		image_points,
		vertex_indices,
		mean_plus_blendshapes,
		lambda,
		num_shape_coefficients_to_fit
	);

	// Estimate the blendshape coefficients with the current PCA model estimate:
	current_pca_shape = morphable_model.get_shape_model().draw_sample(pca_shape_coefficients);

	for (int j = 0; j < num_images; ++j) {
		blendshape_coefficients[j] = fitting::fit_blendshapes_to_landmarks_nnls(
			blendshapes, current_pca_shape, affine_from_orthos[j], image_points[j], vertex_indices[j]
		);
		current_combined_shapes[j] = current_pca_shape +
			blendshapes_as_basis * Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(), blendshape_coefficients[j].size());

		current_meshs[j] = morphablemodel::sample_to_mesh(
			current_combined_shapes[j],
			morphable_model.get_color_model().get_mean(),
			morphable_model.get_shape_model().get_triangle_list(),
			morphable_model.get_color_model().get_triangle_list(),
			morphable_model.get_texture_coordinates()
		);
	}

	fitted_image_points = image_points;

	return { current_meshs, rendering_params }; // I think we could also work with a Mat face_instance in this function instead of a Mesh, but it would convolute the code more (i.e. more complicated to access vertices).
};

#ifdef _OPENMP
inline std::pair<std::vector<core::Mesh>, std::vector<fitting::RenderingParameters>> fit_shape_and_pose_multi_parallel(
	const morphablemodel::MorphableModel& morphable_model,
	const std::vector<morphablemodel::Blendshape>& blendshapes,
	std::vector<eos::video::Keyframe>& keyframes,
	const core::LandmarkMapper& landmark_mapper,
	int image_width,
	int image_height,
	int num_images,
	const morphablemodel::EdgeTopology& edge_topology,
	const fitting::ContourLandmarks& contour_landmarks,
	const fitting::ModelContour& model_contour,
	int num_iterations,
	boost::optional<int> num_shape_coefficients_to_fit,
	float lambda,
	boost::optional<fitting::RenderingParameters> initial_rendering_params,
	std::vector<float>& pca_shape_coefficients,
	std::vector<std::vector<float>>& blendshape_coefficients,
	std::vector<std::vector<cv::Vec2f>>& fitted_image_points,
	boost::property_tree::ptree settings) {

	assert(blendshapes.size() > 0);
	assert(num_iterations > 0); // Can we allow 0, for only the initial pose-fit?
	assert(pca_shape_coefficients.size() <= morphable_model.get_shape_model().get_num_principal_components());

	using std::vector;
	using cv::Vec2f;
	using cv::Vec4f;
	using cv::Mat;
	using Eigen::VectorXf;
	using Eigen::MatrixXf;

	// read settings and trade-offs:
	int NUM_THREADS = settings.get<int>("reconstruction.num_threads", 1);
	bool use_contours = settings.get<bool>("reconstruction.use_contours", true);

	if (!num_shape_coefficients_to_fit) {
		num_shape_coefficients_to_fit = morphable_model.get_shape_model().get_num_principal_components();
	}

	if (pca_shape_coefficients.empty()) {
		pca_shape_coefficients.resize(num_shape_coefficients_to_fit.get());
	}

	// TODO: This leaves the following case open: num_coeffs given is empty or defined, but the
	// pca_shape_coefficients given is != num_coeffs or the model's max-coeffs. What to do then? Handle & document!
	if (blendshape_coefficients.size() < num_images) {
		for (int j = 0; j < num_images; ++j) {
			std::vector<float> current_blendshape_coefficients;
			current_blendshape_coefficients.resize(blendshapes.size());
			blendshape_coefficients.push_back(current_blendshape_coefficients);
		}
	}

	MatrixXf blendshapes_as_basis = morphablemodel::to_matrix(blendshapes);

	// Current mesh - either from the given coefficients, or the mean:
	VectorXf current_pca_shape = morphable_model.get_shape_model().draw_sample(pca_shape_coefficients);
	vector<VectorXf> current_combined_shapes(num_images);
	vector<eos::core::Mesh> current_meshs(num_images);

	// The 2D and 3D point correspondences used for the fitting:
	vector<vector<Vec4f>> model_points(num_images); // the points in the 3D shape model of all frames.
	vector<vector<int>> vertex_indices(num_images); // their vertex indices of all frames.
	vector<vector<Vec2f>> image_points(num_images); // the corresponding 2D landmark points of all frames.

	vector<fitting::RenderingParameters> rendering_params(num_images); // list of rendering params for all frames.
	std::vector<cv::Mat> affine_from_orthos(num_images);
	std::vector<VectorXf> mean_plus_blendshapes(num_images);

#pragma omp parallel num_threads(NUM_THREADS)
	{
#pragma omp for
		for (int j = 0; j < num_images; ++j) {
			VectorXf current_combined_shape = current_pca_shape +
				blendshapes_as_basis *
					Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(), blendshape_coefficients[j].size());

			eos::core::Mesh current_mesh = morphablemodel::sample_to_mesh(
				current_combined_shape,
				morphable_model.get_color_model().get_mean(),
				morphable_model.get_shape_model().get_triangle_list(),
				morphable_model.get_color_model().get_triangle_list(),
				morphable_model.get_texture_coordinates());

			vector<Vec4f> curr_model_points;
			vector<int> curr_vertex_indices;
			vector<Vec2f> curr_image_points;

			// Get the locations of the model locations of the meshes, vertex_indices and image points
			// (equal to landmark coordinates), for every image / mesh.
			std::tie(curr_model_points, curr_vertex_indices, curr_image_points) = eos::core::get_landmark_coordinates<Vec2f, Vec4f>(
				keyframes[j].fitting_result.landmarks, landmark_mapper, current_mesh);

			// Start constructing a list of rendering parameters needed for reconstruction.
			// Get the current points from the last added image points and model points
			auto current_pose = fitting::estimate_orthographic_projection_linear(
				curr_image_points, curr_model_points, true, image_height);

			fitting::RenderingParameters current_rendering_params(current_pose, image_width, image_height);
			rendering_params[j] = current_rendering_params;

			// update key frame rendering params
			keyframes[j].fitting_result.rendering_parameters = current_rendering_params;

			Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(current_rendering_params, image_width, image_height);

			// if no contour
			affine_from_orthos[j] = affine_from_ortho;

			blendshape_coefficients[j] = fitting::fit_blendshapes_to_landmarks_nnls(
				blendshapes, current_pca_shape, affine_from_ortho, curr_image_points, curr_vertex_indices);

			// Mesh with same PCA coeffs as before, but new expression fit (this is relevant if no initial blendshape coeffs have been given):
			current_combined_shape = current_pca_shape +
				morphablemodel::to_matrix(blendshapes) *
					Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(), blendshape_coefficients[j].size()
					);

			current_combined_shapes[j] = current_combined_shape;

			current_mesh = morphablemodel::sample_to_mesh(
				current_combined_shape, morphable_model.get_color_model().get_mean(),
				morphable_model.get_shape_model().get_triangle_list(),
				morphable_model.get_color_model().get_triangle_list(),
				morphable_model.get_texture_coordinates()
			);

			current_meshs[j] = current_mesh;
			model_points[j] = curr_model_points;
			vertex_indices[j] = curr_vertex_indices;
			image_points[j] = curr_image_points;

			// Estimate the PCA shape coefficients with the current blendshape coefficients:
			VectorXf current_mean_plus_blendshapes = morphable_model.get_shape_model().get_mean() +
				blendshapes_as_basis * Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(),
																		 blendshape_coefficients[j].size());
			mean_plus_blendshapes[j] = current_mean_plus_blendshapes;
		}
	}
	// The static (fixed) landmark correspondences which will stay the same throughout
	// the fitting (the inner face landmarks):
	vector<vector<int>> fixed_vertex_indices (vertex_indices);
	vector<vector<Vec2f>> fixed_image_points (image_points);

	image_points = fixed_image_points;
	vertex_indices = fixed_vertex_indices;

	if (use_contours) {

		for (int j = 0; j < num_images; ++j)
		{
			// Given the current pose, find 2D-3D contour correspondences of the front-facing face contour:
			vector<Vec2f> image_points_contour;
			vector<int> vertex_indices_contour;

			auto curr_keyframe = keyframes[j];
			auto landmarks = curr_keyframe.fitting_result.landmarks;
			auto yaw_angle = glm::degrees(glm::eulerAngles(rendering_params[j].get_rotation())[1]);

			// For each 2D contour landmark, get the corresponding 3D vertex point and vertex id:
			std::tie(image_points_contour, std::ignore, vertex_indices_contour) =
				fitting::get_contour_correspondences(
					landmarks,
					contour_landmarks,
					model_contour,
					yaw_angle,
					current_meshs[j],
					rendering_params[j].get_modelview(),
					rendering_params[j].get_projection(),
					fitting::get_opencv_viewport(image_width, image_height)
				);

			// Add the contour correspondences to the set of landmarks that we use for the fitting:
			vertex_indices[j] = fitting::concat(vertex_indices[j], vertex_indices_contour);
			image_points[j] = fitting::concat(image_points[j], image_points_contour);

			// Fit the occluding (away-facing) contour using the detected contour LMs:
			vector<Eigen::Vector2f> occluding_contour_landmarks;

			// positive yaw = subject looking to the left
			if (yaw_angle >= 0.0f)
			{
				// the left contour is the occluding one we want to use ("away-facing")
				auto contour_landmarks_ =
					core::filter(landmarks, contour_landmarks.left_contour); // Can do this outside of the loop
				std::for_each(begin(contour_landmarks_),
							  end(contour_landmarks_),
							  [&occluding_contour_landmarks](auto &&lm)
							  {
								  occluding_contour_landmarks.push_back({lm.coordinates[0], lm.coordinates[1]});
							  });
			}
			else
			{
				auto contour_landmarks_ = core::filter(landmarks, contour_landmarks.right_contour);
				std::for_each(begin(contour_landmarks_),
							  end(contour_landmarks_),
							  [&occluding_contour_landmarks](auto &&lm)
							  {
								  occluding_contour_landmarks.push_back({lm.coordinates[0], lm.coordinates[1]});
							  });
			}

			auto edge_correspondences = fitting::find_occluding_edge_correspondences_parallel(
				current_meshs[j], edge_topology, rendering_params[j], occluding_contour_landmarks, 180.0f
			);

			image_points[j] = fitting::concat(image_points[j], edge_correspondences.first);
			vertex_indices[j] = fitting::concat(vertex_indices[j], edge_correspondences.second);

			// Get the model points of the current mesh, for all correspondences that we've got:
			model_points[j].clear();

			for (const auto &v : vertex_indices[j])
			{
				model_points[j].push_back(
					{
						current_meshs[j].vertices[v][0],
						current_meshs[j].vertices[v][1],
						current_meshs[j].vertices[v][2],
						current_meshs[j].vertices[v][3]
					});
			}

			// Re-estimate the pose, using all correspondences:
			auto current_pose = fitting::estimate_orthographic_projection_linear(image_points[j],
																				 model_points[j],
																				 true,
																				 image_height);
			rendering_params[j] = fitting::RenderingParameters(current_pose, image_width, image_height);

			Mat affine_from_ortho =
				fitting::get_3x4_affine_camera_matrix(rendering_params[j], image_width, image_height);
			affine_from_orthos[j] = affine_from_ortho;

			// Estimate the PCA shape coefficients with the current blendshape coefficients:
			VectorXf current_mean_plus_blendshapes = morphable_model.get_shape_model().get_mean() +
				blendshapes_as_basis * Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(),
																		 blendshape_coefficients[j].size());
			mean_plus_blendshapes[j] = current_mean_plus_blendshapes;
		}
	}

	pca_shape_coefficients = fitting::fit_shape_to_landmarks_linear_multi_parallel(
		morphable_model,
		affine_from_orthos,
		image_points,
		vertex_indices,
		mean_plus_blendshapes,
		lambda,
		num_shape_coefficients_to_fit
	);

	// Estimate the blendshape coefficients with the current PCA model estimate:
	current_pca_shape = morphable_model.get_shape_model().draw_sample(pca_shape_coefficients);

#pragma omp parallel num_threads(NUM_THREADS)
	{
#pragma omp for
		for (int j = 0; j < num_images; ++j) {
			blendshape_coefficients[j] = fitting::fit_blendshapes_to_landmarks_nnls(
				blendshapes, current_pca_shape, affine_from_orthos[j], image_points[j], vertex_indices[j]
			);
			current_combined_shapes[j] = current_pca_shape +
				blendshapes_as_basis * Eigen::Map<const Eigen::VectorXf>(blendshape_coefficients[j].data(),
																		 blendshape_coefficients[j].size());

			current_meshs[j] = morphablemodel::sample_to_mesh(
				current_combined_shapes[j],
				morphable_model.get_color_model().get_mean(),
				morphable_model.get_shape_model().get_triangle_list(),
				morphable_model.get_color_model().get_triangle_list(),
				morphable_model.get_texture_coordinates()
			);

			// save it to the keyframe, we might need it for showing the reconstruction.
			// we could make it optional
			keyframes[j].fitting_result.mesh = current_meshs[j];
			keyframes[j].fitting_result.rendering_parameters = rendering_params[j];
		}

	}
	fitted_image_points = image_points;

	return {current_meshs, rendering_params}; // I think we could also work with a Mat face_instance in this function instead of a Mesh, but it would convolute the code more (i.e. more complicated to access vertices).
};
#endif // if OpenMP

	} /* namespace fitting */
} /* namespace eos */

#endif /* FITTING_HPP_ */
