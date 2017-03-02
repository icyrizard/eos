#include "catch.hpp"
#include "test_helper.hpp"

#include "glm/ext.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/transform.hpp"

#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/fitting/orthographic_camera_estimation_linear.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/fitting/linear_shape_fitting.hpp"
#include "eos/render/utils.hpp"
#include "eos/render/texture_extraction.hpp"
#include "eos/render/render.hpp"
#include "eos/render/detail/render_detail.hpp"


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <tuple>
#include <random>
#include <cmath>
#include <numeric>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::tuple;

/**
 * Reads an ibug .pts landmark file and returns an ordered vector with
 * the 68 2D landmark coordinates.
 *
 * @param[in] filename Path to a .pts file.
 * @return An ordered vector with the 68 ibug landmarks.
 */
LandmarkCollection<cv::Vec2f> read_pts_landmarks(std::string filename)
{
	using std::getline;
	using cv::Vec2f;
	using std::string;
	LandmarkCollection<Vec2f> landmarks;
	landmarks.reserve(68);

	std::ifstream file(filename);
	if (!file.is_open()) {
		throw std::runtime_error(string("Could not open landmark file: " + filename));
	}

	string line;
	// Skip the first 3 lines, they're header lines:
	getline(file, line); // 'version: 1'
	getline(file, line); // 'n_points : 68'
	getline(file, line); // '{'

	int ibugId = 1;
	while (getline(file, line))
	{
		if (line == "}") { // end of the file
			break;
		}
		std::stringstream lineStream(line);

		Landmark<Vec2f> landmark;
		landmark.name = std::to_string(ibugId);
		if (!(lineStream >> landmark.coordinates[0] >> landmark.coordinates[1])) {
			throw std::runtime_error(string("Landmark format error while parsing the line: " + line));
		}
		// From the iBug website:
		// "Please note that the re-annotated data for this challenge are saved in the Matlab convention of 1 being
		// the first index, i.e. the coordinates of the top left pixel in an image are x=1, y=1."
		// ==> So we shift every point by 1:
		landmark.coordinates[0] -= 1.0f;
		landmark.coordinates[1] -= 1.0f;
		landmarks.emplace_back(landmark);
		++ibugId;
	}
	return landmarks;
};

/**
 * Loads test data. Returns
 * @param landmarks
 * @param landmark_mapper
 * @return
 */
std::tuple<morphablemodel::MorphableModel, vector<Vec4f>, vector<int>, vector<Vec2f>> loadTestData(
		LandmarkCollection<cv::Vec2f> landmarks, core::LandmarkMapper landmark_mapper) {
	morphablemodel::MorphableModel morphable_model = loadTestModel();

	// These will be the final 2D and 3D points used for the fitting:
	vector<Vec4f> model_points; // the points in the 3D shape model
	vector<int> vertex_indices; // their vertex indices
	vector<Vec2f> image_points; // the corresponding 2D landmark points

	// Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
	for (int i = 0; i < landmarks.size(); ++i) {
		auto converted_name = landmark_mapper.convert(landmarks[i].name);

		// no mapping defined for the current landmark
		if (!converted_name) {
			continue;
		}

		int vertex_idx = std::stoi(converted_name.get());

		Vec4f vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
		model_points.emplace_back(vertex);
		vertex_indices.emplace_back(vertex_idx);
		image_points.emplace_back(landmarks[i].coordinates);
	}

	return std::make_tuple(morphable_model, model_points, vertex_indices, image_points);
}

/**
 * Helper function to calculate the euclidean distance between the landmark and a projected
 * point. Nothing more than Pythogas.
 *
 * @param landmark
 * @param vertex_screen_coords
 * @return
 */
inline float euclidean_distance(cv::Vec2f landmark, cv::Mat vertex_screen_coords) {
	float screen_x = vertex_screen_coords.at<float>(0, 0);
	float screen_y = vertex_screen_coords.at<float>(1, 0);

	// Calc squared differences, ready for use in Pythagoras.
	float landmark_diff_x_sq = std::fabs(landmark[0] - screen_x) * std::fabs(landmark[0] - screen_x);
	float landmark_diff_y_sq = std::fabs(landmark[1] - screen_y) * std::fabs(landmark[0] - screen_x);

	return std::sqrt(landmark_diff_x_sq + landmark_diff_y_sq);
}

TEST_CASE("Test ortographic projection", "[projection]" ){
	// ======== begin setup ============
	Mat image = cv::imread("data/image_0010.png");

	LandmarkCollection<cv::Vec2f> landmarks;
	landmarks = read_pts_landmarks("data/image_0010.pts");
	core::LandmarkMapper landmark_mapper = core::LandmarkMapper("../share/ibug2did.txt");

	vector<Vec4f> model_points; // the points in the 3D shape model
	vector<int> vertex_indices; // their vertex indices
	vector<Vec2f> image_points; // the corresponding 2D landmark points
	morphablemodel::MorphableModel morphable_model;
	std::tie(morphable_model, model_points, vertex_indices, image_points) = loadTestData(landmarks, landmark_mapper);

	// Estimate the camera (pose) from the 2D - 3D point correspondences
	fitting::ScaledOrthoProjectionParameters pose = fitting::estimate_orthographic_projection_linear(
		image_points, model_points, true, image.rows
	);

	fitting::RenderingParameters rendering_params(pose, image.cols, image.rows);

	// Estimate the shape coefficients by fitting the shape to the landmarks:
	Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params, image.cols, image.rows);

	vector<float> fitted_coeffs = fitting::fit_shape_to_landmarks_linear(
		morphable_model, affine_from_ortho, image_points, vertex_indices
	);

	// Obtain the full mesh with the estimated coefficients:
	render::Mesh mesh = morphable_model.draw_sample(fitted_coeffs, vector<float>());

	// ======== end setup ============

	SECTION("Landmark projection error") {
		vector<float> total_error;

		// Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
		for (int i = 0; i < landmarks.size(); ++i) {
			auto converted_name = landmark_mapper.convert(landmarks[i].name);

			// no mapping defined for the current landmark
			if (!converted_name) {
				continue;
			}

			int vertex_idx = std::stoi(converted_name.get());

			// The vertex_idx should be around the value of the original coordinates after we have
			// projected it with the affine_from_ortho that is obtained earlier.
			Mat vertex_screen_coords = affine_from_ortho *
			   Mat(cv::Vec4f(
					   mesh.vertices[vertex_idx].x,
					   mesh.vertices[vertex_idx].y,
					   mesh.vertices[vertex_idx].z,
					   mesh.vertices[vertex_idx].w
			       )
			   );

			// using euclidean distance here, but should look at other ways too.
			float dist = euclidean_distance(landmarks[i].coordinates, vertex_screen_coords);
			total_error.push_back(dist);
		}

		// Calculate mean error and stddev.
		float accum = 0.0;
		float mean_error = std::accumulate(total_error.begin(), total_error.end(), 0) / landmarks.size();

		// cacl. standard deviation
		std::for_each (std::begin(total_error), std::end(total_error), [&](const float d) {
			accum += (d - mean_error) * (d - mean_error);
		});

		float stddev = std::sqrt(accum / (total_error.size() - 1));

		CAPTURE(mean_error);
		CAPTURE(stddev);

		// TODO: make better requirements / tests for these values.
		// These are just based on the current output of the tests, it however make sure that we do
		// not go over these values while altering eos code.
		REQUIRE(mean_error < 4.0f);
		REQUIRE(stddev < 5.0f);
	}
}
