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
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/landmark_utils.hpp"
#include "eos/core/BufferedVideoIterator.hpp"
#include "eos/morphablemodel/morphablemodel.hpp"
#include "eos/morphablemodel/blendshape.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/render/utils.hpp"
#include "eos/render/texture_extraction.hpp"

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
using eos::core::BufferedVideoIterator;
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
void draw_wireframe(cv::Mat image, const eos::render::Mesh& mesh, glm::mat4x4 modelview, glm::mat4x4 projection, glm::vec4 viewport, cv::Scalar colour = cv::Scalar(0, 255, 0, 255))
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
			cout << "Usage: fit-model [options]" << endl;
			cout << desc;
			return EXIT_SUCCESS;
		}
		po::notify(vm);
	}
	catch (const po::error &e) {
		cout << "Error while parsing command-line arguments: " << e.what() << endl;
		cout << "Use --help to display a list of options." << endl;
		return EXIT_SUCCESS;
	}

	try {
		vector <vector<Vec2f>> multi_frame_points = eos::core::load_annotations(annotations, mappingsfile);
	} catch(const std::runtime_error &e) {
		std::cout << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	// Load landmarks, LandmarkMapper and the Morphable Model:
	LandmarkCollection <cv::Vec2f> landmarks;
	core::LandmarkMapper landmark_mapper = core::LandmarkMapper(mappingsfile);

	try {
		landmarks = eos::core::read_pts_landmarks(annotations[0]);
	}
	catch (const std::runtime_error &e) {
		cout << "Error reading the landmarks: " << e.what() << endl;
		return EXIT_FAILURE;
	}

	morphablemodel::MorphableModel morphable_model;

	try {
		morphable_model = morphablemodel::load_model(modelfile.string());
	} catch (const std::runtime_error &e) {
		std::cout << "Error loading the Morphable Model: " << e.what() << std::endl;
			return EXIT_FAILURE;
	}

	// These will be the final 2D and 3D points used for the fitting:
	vector<Vec4f> model_points; // the points in the 3D shape model
	vector<int> vertex_indices; // their vertex indices
	std::tie(model_points, vertex_indices) = eos::core::load_model_data(landmarks, morphable_model, landmark_mapper);

	BufferedVideoIterator<cv::Mat> vid_iterator;
	std::vector <std::vector<cv::Vec2f>> landmark_annotation_list = eos::core::load_annotations(annotations, mappingsfile);

	try {
		vid_iterator = bufferedvideoiterator<cv::mat>(videofile.string(), landmark_annotation_list);
	} catch(std::runtime_error &e) {
		cout << e.what() << endl;
		return exit_failure;
	}


	// todo: expand this to really perform some reconstruction, and move this to a test file.
	// test with loading 10 frames subsequently.
	// vid_iterator.next() will return a number of frames, depending on
	std::deque<cv::mat> frames = vid_iterator.next();
	int count = 0;
	while(!(frames.empty())) {
		if (count == 10) {
			break;
		}
		int frame_count = 0;
		for (std::deque<cv::mat>::iterator it = frames.begin(); it!=frames.end(); ++it) {
			//std::cout << ' ' << *it;
			std::cout << frame_count << " ";
			frame_count++;
		}

		std::cout << std::endl << "frames processed: " << count * frame_count << std::endl;

		frames = vid_iterator.next();
		usleep(10);
		count++;
	}
}

