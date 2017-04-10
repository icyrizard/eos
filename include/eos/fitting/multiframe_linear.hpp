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
		/**
		 *
		 * @param morphable_model
		 * @param affine_camera_matrix
		 * @param landmarks
		 * @param vertex_ids
		 * @param base_face
		 * @param lambda
		 * @param num_coefficients_to_fit
		 * @param detector_standard_deviation
		 * @param model_standard_deviation
		 *
		 * @return
		 */
		inline std::vector<float> fit_shape_to_landmarks_linear_multi(
			morphablemodel::MorphableModel morphable_model,
			std::vector <cv::Mat> affine_camera_matrix,
			std::vector <std::vector<cv::Vec2f>> landmarks,
			std::vector <std::vector<int>> vertex_ids,
			std::vector <cv::Mat> base_face = std::vector<cv::Mat>(),
			float lambda = 3.0f,
			boost::optional<int> num_coefficients_to_fit = boost::optional<int>(),
			boost::optional<float> detector_standard_deviation = boost::optional<float>(),
			boost::optional<float> model_standard_deviation = boost::optional<float>()) {
		}
	}
}
