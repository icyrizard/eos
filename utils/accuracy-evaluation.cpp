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
void evaluate_accuracy(
		core::LandmarkCollection<cv::Vec2f> landmarks,
		core::LandmarkMapper landmark_mapper,
		const eos::core::Mesh& mesh,
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
	float mean_error = std::accumulate(total_error.begin(), total_error.end(), 0) / landmarks.size();

	for(auto &d: total_error) {
		accum += (d - mean_error) * (d - mean_error);
	}

	float stddev = std::sqrt(accum / (total_error.size() - 1));

	std::cout << "stddev/mean_error: " << stddev << " " <<  mean_error << std::endl;
}

/**
 *
 * @param key_frames
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
void render_output(
		std::vector<eos::video::Keyframe> key_frames,
		std::vector<fitting::RenderingParameters> rendering_paramss,
		std::vector<core::LandmarkCollection<cv::Vec2f>> landmark_list,
		vector<core::Mesh> meshs,
		std::vector<float> pca_shape_coefficients,
		std::vector<std::vector<float>> blendshape_coefficients,
		std::vector<std::vector<cv::Vec2f>> fitted_image_points,
		std::vector<std::string> annotations,
		fitting::ReconstructionData reconstruction_data,
		Mat last_frame,
		int last_frame_number,
		boost::property_tree::ptree settings,
		int n_iter) {

	std::string output_path = settings.get<std::string>("output.output_path", "/tmp");
	float merge_isomap_face_angle = settings.get<float>("output.merge_isomap_face_angle", 60.f);
	bool make_video = settings.get<bool>("output.save_video", false);
	bool show_video = settings.get<bool>("output.show_video", false);

	Mat merged_isomap;
	WeightedIsomapAveraging isomap_averaging(merge_isomap_face_angle); // merge all triangles that are facing <60° towards the camera
	eos::video::PcaCoefficientMerging pca_shape_merging;

	auto landmark_mapper = reconstruction_data.landmark_mapper;
	auto blendshapes = reconstruction_data.blendshapes;
	auto morphable_model = reconstruction_data.morphable_model;
	auto landmarks = reconstruction_data.landmark_list[last_frame_number];

	auto outputfilebase = annotations[0];

	for (uint i = 0; i < key_frames.size(); ++i) {
		Mat frame = key_frames[i].frame;

		auto unmodified_frame = frame.clone();
		int frame_number = key_frames[i].frame_number;
		float yaw_angle = key_frames[i].yaw_angle;// glm::degrees(glm::yaw(rendering_paramss[i].get_rotation()));

		cv::imwrite("/tmp/eos/" + std::to_string(frame_number) + "_" + std::to_string(yaw_angle) + ".png", frame);

		// Extract the texture using the fitted mesh from this frame:
		Mat affine_cam = fitting::get_3x4_affine_camera_matrix(rendering_paramss[i], frame.cols, frame.rows);
		Mat isomap = render::extract_texture(
				meshs[i],
				affine_cam,
				unmodified_frame,
				true,
				render::TextureInterpolation::NearestNeighbour,
				512);

		// Merge the isomaps - add the current one to the already merged ones:
		merged_isomap = isomap_averaging.add_and_merge(isomap);
	}

	// Get last frame (that's the newest one).
	Mat frame = last_frame.clone();
	int frame_width = frame.cols;
	int frame_height = frame.rows;

	vector<cv::Vec4f> model_points;
	vector<int> vertex_indices;
	vector<cv::Vec2f> image_points;

	// make a new one
	std::vector<float> blend_shape_coefficients;

	auto mesh = fitting::generate_new_mesh(
		morphable_model,
		blendshapes,
		pca_shape_coefficients, // current pca_coeff will be the mean for the first iterations.
		blend_shape_coefficients);

	// Will yield model_points, vertex_indices and image_points
	// todo: should this function not come from mesh?
	core::get_mesh_coordinates(landmarks, landmark_mapper, mesh, model_points, vertex_indices, image_points);

	auto current_pose = fitting::estimate_orthographic_projection_linear(
		image_points, model_points, true, frame_height);

	fitting::RenderingParameters rendering_params(current_pose, frame_width, frame_height);

	// Same for the shape:
	pca_shape_coefficients = pca_shape_merging.add_and_merge(pca_shape_coefficients);

	auto blendshapes_as_basis = morphablemodel::to_matrix(blendshapes);
	auto merged_shape = morphable_model.get_shape_model().draw_sample(pca_shape_coefficients) +
						blendshapes_as_basis *
						Eigen::Map<const Eigen::VectorXf>(blend_shape_coefficients.data(),
														  blend_shape_coefficients.size());

	auto merged_mesh = morphablemodel::sample_to_mesh(
			merged_shape,
			morphable_model.get_color_model().get_mean(),
			morphable_model.get_shape_model().get_triangle_list(),
			morphable_model.get_color_model().get_triangle_list(),
			morphable_model.get_texture_coordinates()
	);


	// Render the model in a separate window using the estimated pose, shape and merged texture:
	Mat rendering;

	// render needs 4 channels 8 bits image, needs a two step conversion.
	cvtColor(frame, rendering, CV_BGR2BGRA);

	// make sure the image is CV_8UC4, maybe do check first?
	rendering.convertTo(rendering, CV_8UC4);

	Mat affine_cam = fitting::get_3x4_affine_camera_matrix(rendering_params, frame_width, frame_height);
	Mat isomap = render::extract_texture(
		merged_mesh,
		affine_cam,
		last_frame,
		true,
		render::TextureInterpolation::NearestNeighbour,
		512);

	std::tie(rendering, std::ignore) = render::render(
			merged_mesh,
			rendering_params.get_modelview(),
			rendering_params.get_projection(),
			frame_width,
			frame_height,
			render::create_mipmapped_texture(isomap),
			true,
			false,
			false,
			rendering);

//	if(save_video) {
//		cv::imshow("render", rendering);
//		cv::waitKey(1);
//	}
//
//	if(show_video) {
//		cv::imshow("render", rendering);
//		cv::waitKey(1);
//	}
//
//	eos::video::ReconstructionVideoWriter outputVideo;
//	Size S = Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH),    //Acquire input size
//				  (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
//	outputVideo.open(NAME , ex, inputVideo.get(CV_CAP_PROP_FPS),S, true);
}

/**
 *
 * @param key_frames
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
void evaluate_results(
		std::vector<eos::video::Keyframe> key_frames,
		std::vector<fitting::RenderingParameters> rendering_paramss,
		vector<core::Mesh> meshs,
		std::vector<float> pca_shape_coefficients,
		std::vector<std::vector<float>> blendshape_coefficients,
		std::vector<std::vector<cv::Vec2f>> fitted_image_points,
		std::vector<std::string> annotations,
		fitting::ReconstructionData reconstruction_data,
		boost::property_tree::ptree settings,
		int n_iter) {

	std::string output_path = settings.get<std::string>("output.output_path", "/tmp");
	float merge_isomap_face_angle = settings.get<float>("output.merge_isomap_face_angle", 60.f);

	WeightedIsomapAveraging isomap_averaging(merge_isomap_face_angle); // merge all triangles that are facing <60° towards the camera
	Mat merged_isomap;
	fs::path outputfilebase = annotations[0];

	auto landmark_mapper = reconstruction_data.landmark_mapper;
	auto blendshapes = reconstruction_data.blendshapes;
	auto morphable_model = reconstruction_data.morphable_model;


	// The 3D head pose can be recovered as follows:
	for (uint i = 0; i < key_frames.size(); ++i) {
		int frame_number = key_frames[i].frame_number;
		auto landmarks = key_frames[i].fitting_result.landmarks;
		auto rendering_params = key_frames[i].fitting_result.rendering_parameters;
		float yaw_angle = key_frames[i].yaw_angle;

		Mat frame = key_frames[i].frame;

		int frame_width = frame.cols;
		int frame_height = frame.rows;

		Mat outimg = frame.clone();

		// and similarly for pitch and roll.
		// Extract the texture from the image using given mesh and camera parameters:
		Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(
			rendering_params, frame_width, frame_height);

		for (auto &&lm : landmarks) {
			cv::rectangle(
					outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f),
					cv::Point2f(lm.coordinates[0], lm.coordinates[1] + 2.0f), {255, 0, 0}
			);
		}
		// Draw the fitted mesh as wireframe, and save the image:
		draw_wireframe(
				outimg,
				meshs[i],
				rendering_params.get_modelview(),
				rendering_params.get_projection(),
				fitting::get_opencv_viewport(frame_width, frame_height));

//		bool eyes_open = isomap_averaging.has_eyes_open(frame, landmarks);
//
//		if (!eyes_open) {
//			continue;
//		}

//		cv::imshow("Img", outimg);
//		cv::waitKey(0);

		// Draw the loaded landmarks:
		Mat isomap = render::extract_texture(meshs[i], affine_from_ortho, frame);

		// merge the isomaps:
		merged_isomap = isomap_averaging.add_and_merge(isomap);

//		cv::imshow("isomap", isomap);
//		cv::imshow("merged_isomap", merged_isomap);
//
//		Mat outimg = frame.clone();
//
//		for (auto &&lm : landmarks) {
//			cv::rectangle(
//					outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f),
//					cv::Point2f(lm.coordinates[0], lm.coordinates[1] + 2.0f), {255, 0, 0}
//			);
//		}
//		// Draw the fitted mesh as wireframe, and save the image:
//		draw_wireframe(
//				outimg,
//				meshs[i],
//				rendering_params.get_modelview(),
//				rendering_params.get_projection(),
//				fitting::get_opencv_viewport(frame_width, frame_height));
//
//		cv::imshow("Img", outimg);
//		cv::waitKey(0);

//		fs::path path = (fs::path(annotations[frame_number]).parent_path() / "eval");
//		std::string outputfile = (path / fs::path(annotations[frame_number]).replace_extension("").filename()).string();
//		std::string iter = "_" + std::to_string(n_iter) + "_" + std::to_string(i);
//		cv::imwrite(outputfile + iter + ".annotated.png", outimg);
//
//		// save frontal rendering with texture:
//		glm::mat4 modelview_frontal = glm::mat4( 1.0 );
//		core::Mesh neutral_expression = morphablemodel::sample_to_mesh(
//				morphable_model.get_shape_model().draw_sample(pca_shape_coefficients),
//				morphable_model.get_color_model().get_mean(),
//				morphable_model.get_shape_model().get_triangle_list(),
//				morphable_model.get_color_model().get_triangle_list(),
//				morphable_model.get_texture_coordinates()
//		);

//		cv::imwrite(outputfile + iter + ".isomap.png", isomap);

//		Mat frontal_rendering;
//		std::tie(frontal_rendering, std::ignore) = eos::render::render(
//				neutral_expression,
//				rendering_params.get_modelview(),
//				rendering_params.get_projection(),
//				frame_width,
//				frame_height,
//				render::create_mipmapped_texture(merged_isomap),
//				true,
//				false,
//				false
//		);

//		cv::imwrite(outputfile + iter + ".frontal.png", frontal_rendering);
//		evaluate_accuracy(
//			landmarks,
//			landmark_mapper,
//			meshs[i],
//			affine_from_ortho,
//			settings
//		);
	}

	// save the merged isomap:
	std::string iter = "_" + std::to_string(n_iter);
	fs::path path = (fs::path(annotations[n_iter]).parent_path() / "eval");
	fs::create_directory(path);
	std::string outputfile = (path / fs::path(annotations[n_iter]).replace_extension("").filename()).string();

	// sharpen isomap
	Mat sharpen = isomap_averaging.sharpen(merged_isomap);

	cv::imwrite(outputfile + iter + ".isomap.png", sharpen);

	// save the frontal rendering with merged isomap:
	Mat frontal_rendering;
	glm::mat4 modelview_frontal = glm::mat4( 1.0 );
	core::Mesh neutral_expression = morphablemodel::sample_to_mesh(
			morphable_model.get_shape_model().draw_sample(pca_shape_coefficients),
			morphable_model.get_color_model().get_mean(),
			morphable_model.get_shape_model().get_triangle_list(),
			morphable_model.get_color_model().get_triangle_list(),
			morphable_model.get_texture_coordinates()
	);

//	std::tie(frontal_rendering, std::ignore) = render::render(
//			neutral_expression,
//			modelview_frontal,
//			glm::ortho(-130.0f, 130.0f, -130.0f, 130.0f),
//			512, 512,
//			render::create_mipmapped_texture(merged_isomap),
//			true,
//			false,
//			false
//	);
//	cv::imwrite(outputfile + iter + ".merged_frontal.png", frontal_rendering);

	// Save the mesh as textured obj:
	core::write_textured_obj(neutral_expression, outputfile + iter + ".obj");
	std::cout << "Finished fitting and wrote result mesh and isomap to files with basename " << outputfile + iter + ".obj" << std::endl;
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
 * @param key_frames
 * @param landmarks
 * @return
 */
vector<core::LandmarkCollection<cv::Vec2f>> sample_landmarks(
		std::vector<eos::video::Keyframe> key_frames, vector<core::LandmarkCollection<cv::Vec2f>> landmarks) {
	vector<core::LandmarkCollection<cv::Vec2f>> sublist;

	for (auto& f : key_frames) {
		sublist.push_back(landmarks[f.frame_number]);
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

	// Start with the video play and get video file:
	BufferedVideoIterator vid_iterator;

	try {
		vid_iterator = BufferedVideoIterator(videofile.string(), reconstruction_data, settings);
	} catch(std::runtime_error &e) {
		std::cout << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	// Start getting video frames:
	vid_iterator.start();
//	vid_writer.start();

	// Count the amount of iterations:
	int n_iter = 0;

	while(vid_iterator.is_playing()) {
		auto key_frames = vid_iterator.get_keyframes();

		// Generate a sublist of the total available landmark list:
		auto landmark_sublist = sample_landmarks(key_frames, landmark_list);

		// it makes no sense to update pca_coeff if nothing in the buffer has changed:
		if (vid_iterator.has_changed()) {
			std::cout << "Going to reconstruct with " << key_frames.size() << " images."<< std::endl;

			// Fit shape and pose:
			auto t1 = std::chrono::high_resolution_clock::now();
			std::tie(meshs, rendering_paramss) = fitting::fit_shape_and_pose_multi_parallel(
				morphable_model,
				blendshapes,
				key_frames,
				landmark_mapper,
				key_frames[0].frame.cols,
				key_frames[0].frame.rows,
				static_cast<int>(key_frames.size()),
				edge_topology,
				ibug_contour,
				model_contour,
				50,
				boost::none,
				30.0f,
				boost::none,
				pca_shape_coefficients,
				blendshape_coefficients,
				fitted_image_points,
				settings
			);

			auto t2 = std::chrono::high_resolution_clock::now();
			std::cout << "Reconstruction took "
					  << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
					  << "ms, mean(" << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() / key_frames.size()
					  << "ms)" << std::endl;


//			evaluate_results(
//				key_frames,
//				rendering_paramss,
//				meshs,
//				pca_shape_coefficients,
//				blendshape_coefficients,
//				fitted_image_points,
//				annotations,
//				reconstruction_data,
//				settings,
//				n_iter
//			);

		} else {
//			std::cout << "No reconstruction - buffer did not change" << std::endl;
		}

		// Get new frames:
		n_iter++;
	}

//	vid_writer.__stop();
	if(settings.get<bool>("output.make_video", false)) {
		ReconstructionVideoWriter vid_writer;
		try {
			vid_writer = ReconstructionVideoWriter(videofile.string(), reconstruction_data, settings);
		} catch(std::runtime_error &e) {
			std::cout << e.what() << std::endl;
			return EXIT_FAILURE;
		}
		// Render output:
		std::cout << "Waiting for video to be completed..." << std::endl;
		vid_writer.update_reconstruction_coeff(pca_shape_coefficients);

		while (vid_writer.next()) {
			printf("%d/%d\r", vid_writer.get_frame_number(), vid_iterator.get_frame_number());
		}

	}

//	auto key_frames = vid_iterator.get_keyframes();
//	std::cout << "Going to reconstruct with " << key_frames.size() << " images."<< std::endl;
//
//	evaluate_results(
//		key_frames,
//		rendering_paramss,
//		meshs,
//		pca_shape_coefficients,
//		blendshape_coefficients,
//		fitted_image_points,
//		annotations,
//		reconstruction_data,
//		settings,
//		n_iter
//	);

	//todo: we could build our final obj here?

	return EXIT_SUCCESS;
}
