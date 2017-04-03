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
#include "eos/morphablemodel/morphablemodel.hpp"
#include "eos/morphablemodel/blendshape.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/render/utils.hpp"
#include "eos/render/texture_extraction.hpp"
#include "eos/video/Keyframe.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
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
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::vector;
using std::string;

using namespace cv;

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
		if (eos::render::detail::are_vertices_ccw_in_screen_space(glm::vec2(p1), glm::vec2(p2), glm::vec2(p3)))
		{
			cv::line(image, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), colour);
			cv::line(image, cv::Point(p2.x, p2.y), cv::Point(p3.x, p3.y), colour);
			cv::line(image, cv::Point(p3.x, p3.y), cv::Point(p1.x, p1.y), colour);
		}
	}
};

void evaluate_results(
		std::deque<eos::video::Keyframe> key_frames,
		std::vector<fitting::RenderingParameters> rendering_paramss,
		std::vector<core::LandmarkCollection<cv::Vec2f>> landmark_list,
		morphablemodel::MorphableModel morphable_model,
		vector<core::Mesh> meshs,
		std::vector<float> pca_shape_coefficients,
		std::vector<std::vector<float>> blend_shape_coefficients,
		std::vector<std::vector<cv::Vec2f>> fitted_image_points,
		std::vector<std::string> annotations, int n_iter) {

	WeightedIsomapAveraging isomap_averaging(60.f); // merge all triangles that are facing <60° towards the camera
	Mat merged_isomap;

	fs::path outputfilebase = annotations[0];

	// The 3D head pose can be recovered as follows:
	for (uint i = 0; i < key_frames.size(); ++i) {
		float yaw_angle = glm::degrees(glm::yaw(rendering_paramss[i].get_rotation()));
		int frame_number = key_frames[i].frame_number;
		Mat frame = key_frames[i].frame;

		int frame_width = frame.cols;
		int frame_height = frame.rows;

		// and similarly for pitch and roll.
		// Extract the texture from the image using given mesh and camera parameters:
		Mat affine_from_ortho = fitting::get_3x4_affine_camera_matrix(
			rendering_paramss[i], frame_width, frame_height
		);

		// Draw the loaded landmarks:
		Mat isomap = render::extract_texture(meshs[i], affine_from_ortho, frame);
		Mat outimg = frame.clone();

		std::cout << "processing:" << frame_number << std::endl;
		for (auto&& lm : landmark_list[frame_number]) {
			cv::rectangle(
					outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f),
							cv::Point2f(lm.coordinates[0], lm.coordinates[1] + 2.0f), { 255, 0, 0 }
			);
		}

		// Draw the fitted mesh as wireframe, and save the image:
		draw_wireframe(
				outimg,
				meshs[i],
				rendering_paramss[i].get_modelview(),
				rendering_paramss[i].get_projection(),
				fitting::get_opencv_viewport(frame_width, frame_height)
		);

		fs::path outputfile = annotations[i];
		outputfile.replace_extension(".annotated.png");
		cv::imwrite(outputfile.string(), outimg);

		// save frontal rendering with texture:
		Mat frontal_rendering;
		glm::mat4 modelview_frontal = glm::mat4( 1.0 );
		core::Mesh neutral_expression = morphablemodel::sample_to_mesh(
				morphable_model.get_shape_model().draw_sample(pca_shape_coefficients),
				morphable_model.get_color_model().get_mean(),
				morphable_model.get_shape_model().get_triangle_list(),
				morphable_model.get_color_model().get_triangle_list(),
				morphable_model.get_texture_coordinates()
		);

		std::tie(frontal_rendering, std::ignore) = eos::render::render(neutral_expression, modelview_frontal, glm::ortho(-130.0f, 130.0f, -130.0f, 130.0f), 256, 256, render::create_mipmapped_texture(isomap), true, false, false);
		cv::imshow("frame", isomap);
		outputfile.replace_extension(".frontal.png");
		cv::imwrite(outputfile.string(), frontal_rendering);
		outputfile.replace_extension("");

		// And save the isomap:
		if (!isomap.empty()) {
			// And save the isomap:
			outputfile.replace_extension(".isomap.png");
			cv::imwrite(outputfile.string(), isomap);

			// merge the isomaps:
			merged_isomap = isomap_averaging.add_and_merge(isomap);
			cv::imwrite(outputfile.string(), isomap);

		}
	}

	// save the merged isomap:
	fs::path outputfile = outputfilebase;
	outputfile += fs::path("merged.isomap.png");
	cv::imwrite(outputfile.string(), merged_isomap);
	outputfile.replace_extension("");

	// save the frontal rendering with merged isomap:
	Mat frontal_rendering;
	glm::mat4 modelview_frontal = glm::mat4( 1.0 );
	core::Mesh neutral_expression = morphablemodel::sample_to_mesh(
			morphable_model.get_shape_model().draw_sample(pca_shape_coefficients),
			morphable_model.get_color_model().get_mean(),
			morphable_model.get_shape_model().get_triangle_list(), morphable_model.get_color_model().get_triangle_list(), morphable_model.get_texture_coordinates());
	std::tie(frontal_rendering, std::ignore) = render::render(neutral_expression, modelview_frontal, glm::ortho(-130.0f, 130.0f, -130.0f, 130.0f), 512, 512, render::create_mipmapped_texture(merged_isomap), true, false, false);
	outputfile.replace_extension(".frontal.png");
	cv::imwrite(outputfile.string(), frontal_rendering);
	outputfile.replace_extension("");

	// Save the mesh as textured obj:
	std::string obj_filename = "_" + std::to_string(n_iter) + ".obj";
	outputfile.replace_extension(obj_filename.c_str());
	core::write_textured_obj(morphable_model.draw_sample(
			pca_shape_coefficients, std::vector<float>()), outputfile.string());

	std::cout << "Finished fitting and wrote result mesh and isomap to files with basename " << outputfilebase << "." << std::endl;
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
	fs::path modelfile, isomapfile, videofile, landmarksfile, mappingsfile, contourfile, edgetopologyfile, blendshapesfile, outputfile;
	std::vector<std::string> annotations;

	// get annotaitions from one file
	bool get_annotations = false;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
				("help,h",
				 "display the help message")
				("model,m", po::value<fs::path>(&modelfile)->required()->default_value("../share/sfm_shape_3448.bin"),
				 "a Morphable Model stored as cereal BinaryArchive")
				("video,i", po::value<fs::path>(&videofile)->required(),
				 "an input image")
				("get_annotations,g", po::bool_switch(&get_annotations)->default_value(false),
				 "read .pts annotation file locations from one file, one file path per line")
				("annotations,l", po::value<vector<std::string>>(&annotations)->multitoken()->default_value("../bin/data/vid_annot_pts_files.txt"),
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

	// start loading prerequisites
	morphablemodel::MorphableModel morphable_model;
	try {
		morphable_model = morphablemodel::load_model(modelfile.string());
	} catch (const std::runtime_error &e) {
		std::cout << "Error loading the Morphable Model: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	// Load landmarks, LandmarkMapper and the Morphable Model:
	core::LandmarkMapper landmark_mapper = core::LandmarkMapper(mappingsfile);

	// load all annotation files into lists of landmarks
	vector<core::LandmarkCollection<cv::Vec2f>> landmark_list;
	try {
		std::tie(landmark_list, annotations) = eos::core::load_annotations<cv::Vec2f>(annotations, landmark_mapper, morphable_model, get_annotations);
	} catch(const std::runtime_error &e) {
		std::cout << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	// The expression blendshapes:
	vector<morphablemodel::Blendshape> blend_shapes = morphablemodel::load_blendshapes(blendshapesfile.string());

	// These two are used to fit the front-facing contour to the ibug contour landmarks:
	fitting::ModelContour model_contour = contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile.string());
	fitting::ContourLandmarks ibug_contour = fitting::ContourLandmarks::load(mappingsfile.string());

	// The edge topology is used to speed up computation of the occluding face contour fitting:
	morphablemodel::EdgeTopology edge_topology = morphablemodel::load_edge_topology(edgetopologyfile.string());

	// These will be the final 2D and 3D points used for the fitting:
	vector<cv::Vec3f> model_points; // the points in the 3D shape model
	vector<int> vertex_indices; // their vertex indices
	vector<core::Mesh> meshs;
	vector<fitting::RenderingParameters> rendering_paramss;

	BufferedVideoIterator vid_iterator;

	try {
		vid_iterator = BufferedVideoIterator(videofile.string());
	} catch(std::runtime_error &e) {
		std::cout << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	// iteration count
	int frame_width = vid_iterator.width;
	int frame_height = vid_iterator.height;

	// test with loading 10 frames subsequently.
	std::deque<eos::video::Keyframe> key_frames = vid_iterator.next();

	std::vector<float> pca_shape_coefficients;
	std::vector<std::vector<float>> blend_shape_coefficients;
	std::vector<std::vector<cv::Vec2f>> fitted_image_points;

	int n_iter = 0;

	while(!(key_frames.empty())) {
		if (n_iter == 10) {
			break;
		}

		std::tie(meshs, rendering_paramss) = fitting::fit_shape_and_pose_multi(
				morphable_model,
				blend_shapes,
				landmark_list,
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
				blend_shape_coefficients,
				fitted_image_points
		);

		evaluate_results(
				key_frames,
				rendering_paramss,
				landmark_list,
				morphable_model,
				meshs,
				pca_shape_coefficients,
				blend_shape_coefficients,
				fitted_image_points,
				annotations,
				n_iter
		);

		key_frames = vid_iterator.next();
		n_iter++;
	}

	return EXIT_SUCCESS;

}

//		// iterator through all frames, not needed persee.
//		for (std::deque<eos::video::Keyframe>::iterator it = frames.begin(); it!=frames.end(); ++it) {
//			std::cout << it->score << " ";
//			frame_count++;
//		}

