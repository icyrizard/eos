/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/fitting/FittingResult.hpp
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

#ifndef FITTINGRESULT_HPP_
#define FITTINGRESULT_HPP_

#include "eos/fitting/RenderingParameters.hpp"

#include <vector>

namespace eos {
	namespace fitting {

/**
 * @brief A.
 *
 * A.
 * Note: Any of them can be empty (=default constructed?).
 * For example when we fit to a video, and want to save the
 * final result, which will only have PCA shape coeffs, without the rest.
 * What happens in serialisation?
 * Or in that case, we should actually just use save_coefficients?
 */
struct FittingResult
{
	RenderingParameters rendering_parameters;
	std::vector<float> pca_shape_coefficients;
	std::vector<float> blendshape_coefficients;
	core::LandmarkCollection<cv::Vec2f> landmarks;
	core::Mesh mesh;
	std::vector<Vec4f> model_points; // the points in the 3D shape model
	std::vector<int> vertex_indices; // their vertex indices
	std::vector<Vec2f> image_points; // the corresponding 2D landmark points
};

	} /* namespace fitting */
} /* namespace eos */

#endif /* FITTINGRESULT_HPP_ */
