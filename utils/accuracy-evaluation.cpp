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

#include <vector>
#include <iostream>
#include <fstream>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using eos::video::BufferedVideoIterator;
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

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
				("help,h",
				 "display the help message")
				("model,m", po::value<fs::path>(&modelfile)->required()->default_value("../share/sfm_shape_3448.bin"),
				 "a Morphable Model stored as cereal BinaryArchive")
				("video,i", po::value<fs::path>(&videofile)->required(),
				 "an input image")
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

	// start loading prerequisites
	morphablemodel::MorphableModel morphable_model;
	try {
		morphable_model = morphablemodel::load_model(modelfile.string());
	} catch (const std::runtime_error &e) {
		std::cout << "Error loading the Morphable Model: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	// load all annotation files into lists of landmarks
	vector<core::LandmarkCollection<cv::Vec2f>> landmark_list;
	try {
		landmark_list = eos::core::load_annotations<cv::Vec2f, cv::Vec3f>(annotations, mappingsfile);
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

	// Load landmarks, LandmarkMapper and the Morphable Model:
	LandmarkCollection <cv::Vec2f> landmarks;
	core::LandmarkMapper landmark_mapper = core::LandmarkMapper(mappingsfile);

	// These will be the final 2D and 3D points used for the fitting:
	vector<cv::Vec3f> model_points; // the points in the 3D shape model
	vector<int> vertex_indices; // their vertex indices

	vector<core::Mesh> meshs;
	vector<fitting::RenderingParameters> rendering_paramss;

//	std::tie(model_points, vertex_indices) = eos::core::load_model_data<cv::Vec2f, cv::Vec3f>(landmarks, morphable_model, landmark_mapper);

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
	std::deque<eos::video::Keyframe> frames = vid_iterator.next();

	std::vector<float> pca_shape_coefficients;
	std::vector<std::vector<float>> blend_shape_coefficients;
	std::vector<std::vector<cv::Vec2f>> fitted_image_points;

	int n_frames = static_cast<int>(frames.size());
	while(!(frames.empty())) {
		if (n_frames == 100) {
			break;
		}

		int frame_count = 0;
		std::cout << std::endl << "frames in buffer: " << frames.size() << std::endl;
		std::tie(meshs, rendering_paramss) = fitting::fit_shape_and_pose_multi(
				morphable_model,
				blend_shapes,
				landmark_list,
				landmark_mapper,
				frame_width,
				frame_height,
				static_cast<int>(frames.size()),
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

		frames = vid_iterator.next();

		n_frames++;

//		// iterator through all frames, not needed persee.
//		for (std::deque<eos::video::Keyframe>::iterator it = frames.begin(); it!=frames.end(); ++it) {
//			std::cout << it->score << " ";
//			frame_count++;
//		}
	}
}

