/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: utils/json-to-cereal-binary.cpp
 *
 * Copyright 2016 Patrik Huber *
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
#include "eos/core/landmark_utils.hpp"
#include "eos/render/render.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/render/utils.hpp"
#include "eos/render/texture_extraction.hpp"
#include "eos/video/Keyframe.hpp"
#include "eos/fitting/ReconstructionData.hpp"
#include "eos/video/BufferedVideoIterator.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include "glm/gtx/string_cast.hpp"

#include <vector>
#include <iostream>
#include <fstream>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using eos::video::BufferedVideoIterator;
using eos::video::WeightedIsomapAveraging;
using eos::video::ReconstructionVideoWriter;
using eos::video::Keyframe;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::vector;
using std::string;

using namespace cv;

typedef unsigned int uint;

std::mt19937 gen(1);

/**
 * Draws the given mesh as wireframe into the image.
 *
 * It does backface culling, i.e. draws only vertices in CCW order.
 *
 * @param[in] image An image to draw into.
 * @param[in] mesh The mesh to draw.
 * @param[in] modelview Model-view matrix to draw the mesh.
 * @param[in] projection Projection matrix to draw the mesh.
 * @param[in] viewport Viewport to draw the mesh.
 * @param[in] colour Colour of the mesh to be drawn.
 */
void draw_wireframe(cv::Mat image, const eos::core::Mesh& mesh, glm::mat4x4 modelview, glm::mat4x4 projection, glm::vec4 viewport, cv::Scalar colour = cv::Scalar(0, 255, 0, 255))
{
	for (const auto& triangle : mesh.tvi)
	{
		const auto p1 = glm::project({ mesh.vertices[triangle[0]][0], mesh.vertices[triangle[0]][1], mesh.vertices[triangle[0]][2] }, modelview, projection, viewport);
		const auto p2 = glm::project({ mesh.vertices[triangle[1]][0], mesh.vertices[triangle[1]][1], mesh.vertices[triangle[1]][2] }, modelview, projection, viewport);
		const auto p3 = glm::project({ mesh.vertices[triangle[2]][0], mesh.vertices[triangle[2]][1], mesh.vertices[triangle[2]][2] }, modelview, projection, viewport);
		if (eos::render::detail::are_vertices_ccw_in_screen_space(glm::vec2(p1), glm::vec2(p2), glm::vec2(p3))) {
			cv::line(image, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), colour);
			cv::line(image, cv::Point(p2.x, p2.y), cv::Point(p3.x, p3.y), colour);
			cv::line(image, cv::Point(p3.x, p3.y), cv::Point(p1.x, p1.y), colour);
		}
	}
};

/**
 * Helper function to calculate the euclidean distance between the landmark and a projected
 * point. Nothing more than Pythagoras.
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

/**
 *
 * @param landmarks
 * @param landmark_mapper
 * @param mesh
 * @param affine_from_ortho
 */

std::pair<float, float> calc_reprojection_error(
	core::LandmarkCollection<cv::Vec2f> landmarks,
	core::LandmarkMapper landmark_mapper,
	eos::core::Mesh& mesh,
	Mat affine_from_ortho,
	boost::property_tree::ptree settings) {

	vector<float> total_error;

	// Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
	for (uint i = 0; i < landmarks.size(); ++i) {
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
	float mean_error = static_cast<float>(std::accumulate(total_error.begin(), total_error.end(), 0)) / landmarks.size();

	for(auto &d: total_error) {
		accum += (d - mean_error) * (d - mean_error);
	}

	float stddev = std::sqrt(accum / (total_error.size() - 1));

	return std::make_pair(mean_error, stddev);

}

inline std::string array_to_string(std::vector<std::string> input) {
	std::string output;

	for(int i = 0; i < input.size(); i++) {
		output += input[i] + ((i < input.size() - 1) ? "," : "");
	}

	return output;
}

/**
 *
 *
 * @param keyframes
 * @param rendering_paramss
 * @param landmark_list
 * @param morphable_model
 * @param blendshapes
 * @param meshs
 * @param pca_shape_coefficients
 * @param blendshape_coefficients
 * @param fitted_image_points
 * @param annotations
 * @param landmark_mapper
 * @param settings
 * @param n_iter
 */
void eval(std::vector<std::shared_ptr<eos::video::Keyframe>> keyframes,
					  vector<core::Mesh> meshs,
					  std::vector<fitting::RenderingParameters> rendering_paramss,
					  std::vector<float> pca_shape_coefficients,
					  std::vector<std::vector<cv::Vec2f>> fitted_image_points,
					  std::vector<std::string> annotations,
					  fitting::ReconstructionData reconstruction_data,
					  boost::property_tree::ptree settings,
					  long process_time,
					  std::string videofile,
					  std::string reconstruction_config,
					  int n_iter) {

	std::string output_path = settings.get<std::string>("output.output_path", "/tmp");
	float merge_isomap_face_angle = 90.0; //settings.get<float>("output.merge_isomap_face_angle", 60.f);
	bool use_pose_binning = settings.get<bool>("frames.use_pose_binning", true);
	bool show_video = settings.get<bool>("output.show_video", false);

	Mat merged_isomap;
	fs::path outputfilebase = reconstruction_config;
	WeightedIsomapAveraging isomap_averaging(merge_isomap_face_angle); // merge all triangles that are facing <60° towards the camera

	auto landmark_mapper = reconstruction_data.landmark_mapper;
	auto blendshapes = morphablemodel::to_matrix(reconstruction_data.blendshapes);
	auto morphable_model = reconstruction_data.morphable_model;

	// The 3D head pose can be recovered as follows:
	for (uint i = 0; i < keyframes.size(); ++i) {
		Mat frame = keyframes[i].get()->frame;

		int frame_width = frame.cols;
		int frame_height = frame.rows;

		auto landmarks = keyframes[i].get()->fitting_result.landmarks;
		auto rendering_params = rendering_paramss[i];
		auto yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));

		// and similarly for pitch and roll.
		// Extract the texture from the image using given mesh and camera parameters:
		Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params, frame_width, frame_height);

		Mat isomap = render::extract_texture(
			meshs[i],
			affine_from_ortho,
			frame,
			true,
			render::TextureInterpolation::NearestNeighbour,
			512);

		// Merge the isomaps - add the current one to the already merged ones:
		merged_isomap = isomap_averaging.add_and_merge(isomap);
	}

	// gather csv output here
	std::vector<std::string> output_csv;

	size_t last_frame_index = keyframes.size() - 1;
	auto keyframe = keyframes[last_frame_index].get();
	Mat frame = keyframe->frame;

	int frame_width = frame.cols;
	int frame_height = frame.rows;

	int frame_number = keyframe->frame_number;
	auto landmarks = keyframe->fitting_result.landmarks;
	auto rendering_params = rendering_paramss[last_frame_index];
	auto yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));
	auto mesh = meshs[last_frame_index];

	Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params, frame_width, frame_height);

	float reprojection_error;
	float stddev;
	std::tie(reprojection_error, stddev) = calc_reprojection_error(
		landmarks, landmark_mapper, mesh, affine_from_ortho, settings);

	output_csv.push_back(std::to_string(keyframe->frame_number));
	output_csv.push_back(std::to_string(keyframes.size()));
	output_csv.push_back(std::to_string(reprojection_error));
	output_csv.push_back(std::to_string(stddev));
	output_csv.push_back(std::to_string(yaw_angle));
	output_csv.push_back(std::to_string(settings.get<int>("frames.frames_per_bin")));
	output_csv.push_back(use_pose_binning ? "true" : "false");
	output_csv.push_back("true");
	output_csv.push_back(std::to_string(process_time));
	output_csv.push_back(std::to_string(frame_width));
	output_csv.push_back(std::to_string(frame_height));
	output_csv.push_back(videofile);
	output_csv.push_back(reconstruction_config);

	if (show_video) {
		Mat outimg = frame.clone();

		std::cout << "fitted points size: " << fitted_image_points.size() << " -- " << landmarks.size() << std::endl;
		// Draw the fitted mesh as wireframe, and save the image:
		if (settings.get<bool>("output.landmarks", true))
		{
			for (auto &&f: fitted_image_points[last_frame_index]) {
				cv::rectangle(
					outimg, cv::Point2f(f[0] - 2.0f, f[1] - 2.0f),
					cv::Point2f(f[0], f[1] + 2.0f), {255, 0, 0}
				);

			}
		}

		if (settings.get<bool>("output.wireframe", true)) {
			draw_wireframe(
				outimg,
				mesh,
				rendering_params.get_modelview(),
				rendering_params.get_projection(),
				fitting::get_opencv_viewport(frame_width, frame_height));
		}

		cv::imshow("Reconstruction", outimg);
		cv::waitKey(20);
	}

	if (settings.get<bool>("output.build_obj", true)) {
		// save the merged isomap:
		std::string iter = "_" + std::to_string(n_iter);
		fs::path path = (fs::path(reconstruction_config).parent_path()
			/ fs::path(reconstruction_config).replace_extension("").filename()).string();
		fs::create_directory(path);
		std::string outputfile = (path / fs::path(reconstruction_config).replace_extension("").filename()).string();

		outputfile += use_pose_binning ? "_true" : "_false";

		// sharpen isomap
		Mat sharpen = isomap_averaging.sharpen(merged_isomap);
		cv::imwrite(outputfile + iter + ".isomap.png", sharpen);

		// save the frontal rendering with merged isomap:
		Mat frontal_rendering;
		glm::mat4 modelview_frontal = glm::mat4(1.0);

		std::tie(frontal_rendering, std::ignore) = render::render(
			mesh,
			modelview_frontal,
			glm::ortho(-130.0f, 130.0f, -130.0f, 130.0f),
			512, 512,
			render::create_mipmapped_texture(merged_isomap),
			true,
			false,
			false
		);

		cv::imwrite(outputfile + iter + ".merged_frontal.png", frontal_rendering);

		output_csv.push_back(outputfile + iter + ".obj");

		// Save the mesh as textured obj:
		core::write_textured_obj(mesh, outputfile + iter + ".obj");
		std::cout << "Finished fitting and wrote result mesh and isomap to files with basename "
				  << outputfile + iter + ".obj" << std::endl;
	}

	// add all alpha's
	for (auto &alpha: pca_shape_coefficients) {
		output_csv.push_back(std::to_string(alpha));
	}

	std::cerr << array_to_string(output_csv) << std::endl;
}

/**
 *
 * @param keyframes
 * @param rendering_paramss
 * @param landmark_list
 * @param morphable_model
 * @param blendshapes
 * @param meshs
 * @param pca_shape_coefficients
 * @param blendshape_coefficients
 * @param fitted_image_points
 * @param annotations
 * @param landmark_mapper
 * @param settings
 * @param n_iter
 */
void evaluate_results(std::vector<std::shared_ptr<eos::video::Keyframe>> keyframes,
		  std::vector<float> pca_shape_coefficients,
		  std::vector<std::vector<cv::Vec2f>> fitted_image_points,
		  std::vector<std::string> annotations,
		  fitting::ReconstructionData reconstruction_data,
		  boost::property_tree::ptree settings,
		  long process_time,
		  std::string videofile,
		  std::string reconstruction_config,
		  int n_iter) {

	std::string output_path = settings.get<std::string>("output.output_path", "/tmp");
	float merge_isomap_face_angle = settings.get<float>("output.merge_isomap_face_angle", 60.f);
	bool use_pose_binning = settings.get<bool>("frames.use_pose_binning", true);

	Mat merged_isomap;
	Mat isomap;
	fs::path outputfilebase = reconstruction_config;
	WeightedIsomapAveraging isomap_averaging(merge_isomap_face_angle); // merge all triangles that are facing <60° towards the camera

	auto landmark_mapper = reconstruction_data.landmark_mapper;
	auto blendshapes = morphablemodel::to_matrix(reconstruction_data.blendshapes);
	auto morphable_model = reconstruction_data.morphable_model;

	bool show_video = settings.get<bool>("output.show_video", false);

	// The 3D head pose can be recovered as follows:
	for (uint i = 0; i < keyframes.size(); ++i) {
		Mat frame = keyframes[i].get()->frame;

		int frame_width = frame.cols;
		int frame_height = frame.rows;

		int frame_number = keyframes[i].get()->frame_number;
		auto landmarks = keyframes[i].get()->fitting_result.landmarks;
		auto rendering_params = keyframes[i].get()->fitting_result.rendering_parameters;
		auto yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));

		// and similarly for pitch and roll.
		// Extract the texture from the image using given mesh and camera parameters:
		Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params, frame_width, frame_height);

		// Draw the loaded landmarks:
		isomap = render::extract_texture(keyframes[i].get()->fitting_result.mesh, affine_from_ortho, frame);

		// merge the isomaps:
		merged_isomap = isomap_averaging.add_and_merge(isomap);
	}

	// gather csv output here
	std::vector<std::string> output_csv;

	size_t last_frame_index = keyframes.size() - 1;
	auto keyframe = keyframes[last_frame_index].get();
	Mat frame = keyframe->frame;

	int frame_width = frame.cols;
	int frame_height = frame.rows;

	int frame_number = keyframe->frame_number;
	auto landmarks = keyframe->fitting_result.landmarks;
	auto rendering_params = keyframe->fitting_result.rendering_parameters;
	auto yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));
	auto mesh = keyframes[last_frame_index].get()->fitting_result.mesh;

	Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(rendering_params, frame_width, frame_height);

	float reprojection_error;
	float stddev;
//	std::tie(reprojection_error, stddev) = calc_reprojection_error(
//		landmarks, landmark_mapper, mesh, affine_from_ortho, settings);

//	output_csv.push_back(std::to_string(keyframe->frame_number));
//	output_csv.push_back(std::to_string(keyframes.size()));
//	output_csv.push_back(std::to_string(reprojection_error));
//	output_csv.push_back(std::to_string(stddev));
//	output_csv.push_back(std::to_string(yaw_angle));
//	output_csv.push_back(std::to_string(settings.get<int>("frames.frames_per_bin")));
//	output_csv.push_back(use_pose_binning ? "true": "false");
//	output_csv.push_back(std::to_string(process_time));
//	output_csv.push_back(std::to_string(frame_width));
//	output_csv.push_back(std::to_string(frame_height));
//	output_csv.push_back(videofile);
//	output_csv.push_back(reconstruction_config);

//	for (auto &alpha: pca_shape_coefficients) {
//		output_csv.push_back(std::to_string(alpha));
//	}
//
//	std::cerr << array_to_string(output_csv) << std::endl;
//
//	if (show_video) {
//		Mat outimg = frame.clone();
//
//		// Draw the fitted mesh as wireframe, and save the image:
//		if (settings.get<bool>("output.landmarks", true)) {
//			for (auto &&lm : landmarks) {
//				cv::rectangle(
//					outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f),
//					cv::Point2f(lm.coordinates[0], lm.coordinates[1] + 2.0f), {255, 0, 0}
//				);
//			}
//		}
//
//		if (settings.get<bool>("output.wireframe", true)) {
//			draw_wireframe(
//				outimg,
//				mesh,
//				rendering_params.get_modelview(),
//				rendering_params.get_projection(),
//				fitting::get_opencv_viewport(frame_width, frame_height));
//		}
//
//		cv::imshow("Reconstruction", outimg);
//		cv::waitKey(20);
//	}
//
//	if (settings.get<bool>("output.build_obj", true)) {
//		// save the merged isomap:
//		std::string iter = "_" + std::to_string(n_iter);
//		fs::path path = (fs::path(reconstruction_config).parent_path() / fs::path(reconstruction_config).replace_extension("").filename()).string();
//		fs::create_directory(path);
//		std::string outputfile = (path / fs::path(reconstruction_config).replace_extension("").filename()).string();
//
//		outputfile += use_pose_binning ? "_true" : "_false";
//
//		// sharpen isomap
//		Mat sharpen = isomap_averaging.sharpen(merged_isomap);
//		cv::imwrite(outputfile + iter + ".isomap.png", sharpen);
//
//		// save the frontal rendering with merged isomap:
//		Mat frontal_rendering;
//		glm::mat4 modelview_frontal = glm::mat4(1.0);
//
//		std::tie(frontal_rendering, std::ignore) = render::render(
//			mesh,
//			modelview_frontal,
//			glm::ortho(-130.0f, 130.0f, -130.0f, 130.0f),
//			512, 512,
//			render::create_mipmapped_texture(merged_isomap),
//			true,
//			false,
//			false
//		);
//
//		cv::imwrite(outputfile + iter + ".merged_frontal.png", frontal_rendering);
//
//		// Save the mesh as textured obj:
//		core::write_textured_obj(mesh, outputfile + iter + ".obj");
//		std::cout << "Finished fitting and wrote result mesh and isomap to files with basename "
//				  << outputfile + iter + ".obj" << std::endl;
//	}
}

/**
 * Parse config file
 * @param filename
 */
boost::property_tree::ptree get_reconstruction_config(std::string filename) {
	boost::property_tree::ptree pt;
	boost::property_tree::ini_parser::read_ini(filename, pt);

	return pt;
}

/**
 *
 * Return a list of landmarks based on the keyframe's frame_number. Such that the frame and the landmarks
 * are aligned. The VideoIterator is able to skip frames based on certain conditions, skipping frames
 * causes un-alignment of the total landmarks list and the list of frames. This samples the correct landmark
 * annotations with the based on a given keyframe list.
 *
 * @param keyframes
 * @param landmarks
 * @return
 */
core::LandmarkCollection<cv::Vec2f> add_noise(core::LandmarkCollection<cv::Vec2f> landmarks, int noise_x, int noise_y) {
	std::uniform_real_distribution<> dis_x(0, noise_x);
	std::uniform_real_distribution<> dis_y(0, noise_y);

	core::LandmarkCollection<cv::Vec2f> sublist;
	vector<cv::Vec2f> noise;

	for (auto &lm : landmarks) {
		cv::Vec2f offset(static_cast<float>(dis_x(gen)), dis_y(gen));
		lm.coordinates += offset;

		noise.push_back(offset);
		sublist.push_back(lm);
	}

	return sublist; } /**
 *
 * Return a list of landmarks based on the keyframe's frame_number. Such that the frame and the landmarks
 * are aligned. The VideoIterator is able to skip frames based on certain conditions, skipping frames
 * causes un-alignment of the total landmarks list and the list of frames. This samples the correct landmark
 * annotations with the based on a given keyframe list.
 *
 * @param keyframes
 * @param landmarks
 * @return
 */
vector<core::LandmarkCollection<cv::Vec2f>> sample_landmarks(
	std::vector<std::shared_ptr<eos::video::Keyframe>> keyframes, vector<core::LandmarkCollection<cv::Vec2f>> landmarks, int noise_x, int noise_y) {
	vector<core::LandmarkCollection<cv::Vec2f>> sublist;

	for (auto& f : keyframes) {
		auto orig_landmarks = landmarks[f.get()->frame_number];
		auto noisy_landmarks = add_noise(orig_landmarks, noise_x, noise_y);

		sublist.push_back(noisy_landmarks);
	}

	return sublist;
}

/**
 * This app demonstrates estimation of the camera and fitting of the shape
 * model of a 3D Morphable Model from an ibug LFPW image with its landmarks.
 * In addition to fit-model-simple, this example uses blendshapes, contour-
 * fitting, and can iterate the fitting.
 *
 * 68 ibug landmarks are loaded from the .pts file and converted
 * to vertex indices using the LandmarkMapper.
 */
int main(int argc, char *argv[])
{
	fs::path modelfile, isomapfile, videofile, landmarksfile, mappingsfile, contourfile, edgetopologyfile,
		blendshapesfile, outputfile, reconstruction_config;
	std::vector<std::string> annotations;

	// get annotations from one file
	bool get_annotations = false;

	try
	{
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
			 "display the help message")
			("model,m", po::value<fs::path>(&modelfile)->required()->default_value("../share/sfm_shape_3448.bin"),
			 "a Morphable Model stored as cereal BinaryArchive")
			("video,v", po::value<fs::path>(&videofile)->required(),
			 "an input image")
			("config,f",
			 po::value<fs::path>(&reconstruction_config)->default_value("../share/default_reconstruction_config.ini"),
			 "configuration file for the reconstruction")
			("get_annotations,g", po::bool_switch(&get_annotations)->default_value(false),
			 "read .pts annotation file locations from one file, put one file path on each line")
			("annotations,l", po::value<vector<std::string>>(&annotations)->multitoken(),
			 ".pts annotation files per frame of video")
			("mapping,p", po::value<fs::path>(&mappingsfile)->required()->default_value("../share/ibug2did.txt"),
			 "2D landmarks for the image, in ibug .pts format")
			("model-contour,c",
			 po::value<fs::path>(&contourfile)->required()->default_value("../share/model_contours.json"),
			 "file with model contour indices")
			("edge-topology,e", po::value<fs::path>(&edgetopologyfile)->required()->default_value(
				"../share/sfm_3448_edge_topology.json"),
			 "file with model's precomputed edge topology")
			("blendshapes,b", po::value<fs::path>(&blendshapesfile)->required()->default_value(
				"../share/expression_blendshapes_3448.bin"),
			 "file with blendshapes")
			("output,o", po::value<fs::path>(&outputfile)->required()->default_value("out"),
			 "basename for the output rendering and obj files");
		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
		if (vm.count("help"))
		{
			std::cout << "Usage: fit-model [options]" << std::endl;
			std::cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
	}
	catch (const po::error &e)
	{
		std::cout << "Error while parsing command-line arguments: " << e.what() << std::endl;
		std::cout << "Use --help to display a list of options." << std::endl;
		return EXIT_SUCCESS;
	}

	// Start loading prerequisites:
	morphablemodel::MorphableModel morphable_model;
	std::vector<float> pca_shape_coefficients;

	// List of blendshapes coefficients:
	std::vector<std::vector<float>> blendshape_coefficients;
	std::vector<std::vector<cv::Vec2f>> fitted_image_points;

	// These will be the final 2D and 3D points used for the fitting:
	vector<cv::Vec3f> model_points; // the points in the 3D shape model
	vector<int> vertex_indices; // their vertex indices
	vector<core::Mesh> meshs;
	vector<fitting::RenderingParameters> rendering_paramss;

	// Load landmarks, LandmarkMapper and the Morphable Model:
	auto landmark_mapper = core::LandmarkMapper(mappingsfile);

	// Load all annotation files into lists of landmarks:
	vector<core::LandmarkCollection<cv::Vec2f>> landmark_list;
	try {
		morphable_model = morphablemodel::load_model(modelfile.string());
	}
	catch (const std::runtime_error &e) {
		std::cout << "Error loading the Morphable Model: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	// Load all annotation using input parameters:
	try {
		std::tie(landmark_list, annotations) = eos::core::load_annotations<cv::Vec2f>(
			annotations,
			landmark_mapper,
			morphable_model,
			get_annotations);
	}
	catch (const std::runtime_error &e) {
		std::cout << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	// The expression blendshapes:
	auto blendshapes = morphablemodel::load_blendshapes(blendshapesfile.string());

	// These two are used to fit the front-facing contour to the ibug contour landmarks:
	auto model_contour =
		contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile.string());
	auto ibug_contour = fitting::ContourLandmarks::load(mappingsfile.string());
	// The edge topology is used to speed up computation of the occluding face contour fitting:
	auto edge_topology = morphablemodel::load_edge_topology(edgetopologyfile.string());

	// Read .ini file to accommodate.
	auto settings = get_reconstruction_config(reconstruction_config.string());
	auto reconstruction_data = eos::fitting::ReconstructionData{
		morphable_model, blendshapes, landmark_mapper, landmark_list, model_contour, ibug_contour, edge_topology};

	bool use_pose_binning = settings.get<bool>("frames.use_pose_binning", true);
	float threshold = settings.get<float>("frames.threshold", 0.1);
	bool evaluate = settings.get<bool>("eval.evaluate", true);
	int num_iterations = settings.get<int>("reconstruction.num_iterations", 10);
	int max_frames = settings.get<int>("eval.max_frames", 0);
	int drop_frames = settings.get<int>("frames.drop_frames", 0);
	int noise_x = settings.get<int>("reconstruction.noise_x", 0);
	int noise_y = settings.get<int>("reconstruction.noise_y", 0);

	// Start with the video play and get video file:
	BufferedVideoIterator vid_iterator;

	try {
		vid_iterator = BufferedVideoIterator(videofile.string(), reconstruction_data, settings);
	} catch (std::runtime_error &e) {
		std::cout << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	// TODO: determine if this following one should only be used in other programs:
	// Count the amount of iterations:
	int n_iter = 0;

	int frames_dropped = 0;

	std::vector<std::shared_ptr<Keyframe>> keyframes;
	std::uniform_real_distribution<> keyframe_add_random(0, 1);

	// global random generator with given seed
	std::random_device rd;
	std::mt19937 local_gen(rd());

	// Get frames randomly until video is done playing OR we've reached max_frames
	while (vid_iterator.is_playing()) {
		bool new_frame = vid_iterator.next();
		if (new_frame == false) {
			break;
		}

		double add_frame = keyframe_add_random(local_gen);

		if (add_frame < threshold) {
			keyframes.push_back(std::make_shared<Keyframe>(vid_iterator.get_last_keyframe()));
			if (keyframes.size() >= max_frames) {
				break;
			}
		}

		n_iter++;
	}

	vid_iterator.__stop();

	// Generate a sublist of the total available landmark list:
	auto landmark_sublist = sample_landmarks(keyframes, landmark_list, noise_x, noise_y);
	std::cout << "Going to reconstruct with " << keyframes.size() <<
				   " images, " << num_iterations << " iterations: " << std::endl;

	// Fit shape and pose:
	auto t1 = std::chrono::high_resolution_clock::now();
	std::tie(meshs, rendering_paramss) = fitting::fit_shape_and_pose_multi(
		morphable_model,
		blendshapes,
		landmark_sublist,
		landmark_mapper,
		keyframes[0].get()->frame.cols,
		keyframes[0].get()->frame.rows,
		static_cast<int>(keyframes.size()),
		edge_topology,
		ibug_contour,
		model_contour,
		num_iterations,
		boost::none,
		30.0f,
		boost::none,
		pca_shape_coefficients,
		blendshape_coefficients,
		fitted_image_points
	);

	std::cout << vid_iterator.to_string() << std::endl;
	auto t2 = std::chrono::high_resolution_clock::now();

	std::cout << "Reconstruction took "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
			  << "ms, mean(" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
				  / (keyframes.size() * num_iterations)
			  << "ms)" << std::endl;

	if (settings.get<bool>("eval.evaluate", false)) {
		eval(
			keyframes,
			meshs,
			rendering_paramss,
			pca_shape_coefficients,
			fitted_image_points,
			annotations,
			reconstruction_data,
			settings,
			std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count(),
			videofile.string(),
			reconstruction_config.string(),
			n_iter
		);
	}

	return EXIT_SUCCESS;
}
