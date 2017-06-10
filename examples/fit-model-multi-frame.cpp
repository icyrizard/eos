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
void output_obj(std::vector<std::shared_ptr<eos::video::Keyframe>> keyframes,
					  std::vector<float> pca_shape_coefficients,
					  fitting::ReconstructionData reconstruction_data,
					  boost::property_tree::ptree settings) {

	Mat isomap;
	Mat merged_isomap;
	float merge_isomap_face_angle = settings.get<float>("output.merge_isomap_face_angle", 60.f);
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

	// save an obj + current merged isomap to the disk:
	core::Mesh neutral_expression = morphablemodel::sample_to_mesh(
		morphable_model.get_shape_model().draw_sample(pca_shape_coefficients),
		morphable_model.get_color_model().get_mean(),
		morphable_model.get_shape_model().get_triangle_list(),
		morphable_model.get_color_model().get_triangle_list(),
		morphable_model.get_texture_coordinates());

	core::write_textured_obj(neutral_expression, "current_merged.obj");
	cv::imwrite("current_merged.isomap.png", merged_isomap);
	std::cout << "written current_merged.obj and current_merged.isomap" << std::endl;
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
vector<core::LandmarkCollection<cv::Vec2f>> sample_landmarks(
	std::vector<std::shared_ptr<eos::video::Keyframe>> keyframes, vector<core::LandmarkCollection<cv::Vec2f>> landmarks) {

	vector<core::LandmarkCollection<cv::Vec2f>> sublist;

	for (auto& f : keyframes) {
		sublist.push_back(landmarks[f.get()->frame_number]);
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
int main(int argc, char *argv[]) {
	fs::path modelfile, isomapfile, videofile, landmarksfile, mappingsfile, contourfile, edgetopologyfile, blendshapesfile, outputfile, reconstruction_config;
	std::vector<std::string> annotations;

	// get annotations from one file
	bool get_annotations = false;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h",
			 "display the help message")
			("model,m", po::value<fs::path>(&modelfile)->required()->default_value("../share/sfm_shape_3448.bin"),
			 "a Morphable Model stored as cereal BinaryArchive")
			("video,v", po::value<fs::path>(&videofile)->required(),
			 "an input image")
			("config,f", po::value<fs::path>(&reconstruction_config)->default_value("../share/default_reconstruction_config.ini"),
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
		if (vm.count("help")) {
			std::cout << "Usage: fit-model [options]" << std::endl;
			std::cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
	}
	catch (const po::error &e) {
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
	} catch (const std::runtime_error &e) {
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
	} catch(const std::runtime_error &e) {
		std::cout << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	// The expression blendshapes:
	auto blendshapes = morphablemodel::load_blendshapes(blendshapesfile.string());

	// These two are used to fit the front-facing contour to the ibug contour landmarks:
	auto model_contour = contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile.string());
	auto ibug_contour = fitting::ContourLandmarks::load(mappingsfile.string());
	// The edge topology is used to speed up computation of the occluding face contour fitting:
	auto edge_topology = morphablemodel::load_edge_topology(edgetopologyfile.string());

	// Read .ini file to accommodate.
	auto settings = get_reconstruction_config(reconstruction_config.string());
	auto reconstruction_data = eos::fitting::ReconstructionData{
		morphable_model, blendshapes, landmark_mapper, landmark_list, model_contour, ibug_contour, edge_topology};

	bool use_pose_binning = settings.get<bool>("frames.use_pose_binning", true);
	bool evaluate = settings.get<bool>("eval.evaluate", true);
	int num_iterations = settings.get<int>("reconstruction.num_iterations", 10);
	int max_frames = settings.get<int>("eval.max_frames", 0);
	int drop_frames = settings.get<int>("frames.drop_frames", 0);

	// Start with the video play and get video file:
	BufferedVideoIterator vid_iterator;

	try {
		vid_iterator = BufferedVideoIterator(videofile.string(), reconstruction_data, settings);
	} catch(std::runtime_error &e) {
		std::cout << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	// start a thread to search for the best frames
	vid_iterator.start();

	// Waiting for the iterator to find the best keyframes, we will reconstruct when done, note that this is
	// best for short videos. Although the vid iterator has a maximum capacity therefore memory leaks are
	// not a problem, still short videos will demonstrate its usefulness best.
	while(vid_iterator.is_playing()) { }

	auto keyframes = vid_iterator.get_keyframes();

	// Generate a sublist of the total available landmark list:
	std::cout << "Going to reconstruct with " << keyframes.size() << " images, "<< num_iterations << " iterations:"<< std::endl;

	// Fit shape and pose:
	auto t1 = std::chrono::high_resolution_clock::now();
	std::tie(meshs, rendering_paramss) = fitting::fit_shape_and_pose_multi_parallel(
		morphable_model,
		blendshapes,
		keyframes,
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
		fitted_image_points,
		settings
	);

	std::cout << pca_shape_coefficients.size() << std::endl;
	std::cout << vid_iterator.to_string() << std::endl;

	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "Reconstruction took "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
			  << "ms, mean(" << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() / (keyframes.size() * num_iterations)
			  << "ms)" << std::endl;

	output_obj(
		keyframes,
		pca_shape_coefficients,
		reconstruction_data,
		settings
	);

	return EXIT_SUCCESS;
}

