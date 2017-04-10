//
// Created by RA Torenvliet on 07/02/2017.
//

#ifndef EOS_LANDMARK_UTILS_H
#define EOS_LANDMARK_UTILS_H

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

#include "eos/morphablemodel/morphablemodel.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/Landmark.hpp"

#include "boost/filesystem.hpp"

#include <vector>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::vector;
using std::string;

namespace fs = boost::filesystem;

/**
 * Reads an ibug .pts landmark file and returns an ordered vector with
 * the 68 2D landmark coordinates.
 *
 * @param[in] filename Path to a .pts file.
 * @return An ordered vector with the 68 ibug landmarks.
 */
namespace eos {
	namespace core {
		template<typename vec2f>
		eos::core::LandmarkCollection <vec2f> read_pts_landmarks(std::string filename) {
			using std::getline;

			LandmarkCollection<vec2f> landmarks;
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
			while (getline(file, line)) {
				if (line[0] == '}') { // end of the file
					break;
				}

				std::stringstream lineStream(line);
				Landmark <vec2f> landmark;
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
		}

		/**
		 * Read strings from a given file and return them as a vector of strings.
		 *
		 * @param filename
		 * @return
		 */
		std::vector<std::string> file_to_string_vector(std::string filename) {
			std::ifstream file(filename);
			if (!file.is_open()) {
				throw std::runtime_error(string("Could not open annotation list file: " + filename));
			}

			string line;
			std::vector<std::string> output;

			while (getline(file, line)) {
				std::cout << line << std::endl;
				output.push_back(line);
			}

			return output;
		}

		/**
		 * Helper function, gathers matching model_points with given landmarks and LandmarkMapper.
		 *
		 * @param landmarks
		 * @param morphable_model
		 * @param landmark_mapper
		 * @return std::pair<std::vector<Vec4f>, std::vector<int> model_points and vertex_indices.
		 */
		template<typename vec2f, typename vec3f>
		std::pair <std::vector<vec2f>, std::vector<int>>
		load_model_data(eos::core::LandmarkCollection<vec2f> landmarks,
		                morphablemodel::MorphableModel morphable_model, eos::core::LandmarkMapper landmark_mapper) {
			std::vector <vec3f> model_points;
			std::vector<int> vertex_indices;

			// Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
			for (int i = 0; i < landmarks.size(); ++i) {
				auto converted_name = landmark_mapper.convert(landmarks[i].name);
				if (!converted_name) { // no mapping defined for the current landmark
					continue;
				}
				int vertex_idx = std::stoi(converted_name.get());
				vec3f vertex = morphable_model.get_shape_model().get_mean_at_point(vertex_idx);
				model_points.emplace_back(vertex);
				vertex_indices.emplace_back(vertex_idx);
			}

			return std::make_pair(model_points, vertex_indices);
		}

		/**
		* Load annotations, return all annotations as a vector of LandmarkCollection (vectors of Vec2f).
		* Supports either a given template, any cv::Vec2f / Eigen::Vector2f will work.
		*
		* Returns the annotations as a list of files, useful for later use if read_from_file flag is set to true.
		* This flag will read the filepaths from the given annotation file, probably depends on the command line args.
		*
		* @tparam vec2f
		* @param annotations
		* @param mapper
		* @param morphable_model
		* @param read_from_file
		* @return
		*/
		template<typename vec2f>
		std::pair <std::vector<core::LandmarkCollection<vec2f>>, std::vector<std::string>>
		load_annotations(
				std::vector<std::string>annotations,
				core::LandmarkMapper mapper, morphablemodel::MorphableModel morphable_model, bool read_from_file = false) {
			std::vector<core::LandmarkCollection<vec2f>> landmark_collection_list;

			// load file names from one file.
			std::vector<std::string> annotation_files;

			if (read_from_file) {
				annotations = file_to_string_vector(annotations[0]);
			}

			for (int i = 0; i < annotations.size(); i++) {
				eos::core::LandmarkCollection <vec2f> landmarks;

				try {
					landmarks = read_pts_landmarks<vec2f>(annotations[i]);
				} catch (const std::runtime_error &e) {
					std::cout << e.what() << std::endl;
					throw std::runtime_error("Error reading the landmarks:  " + annotations[i]);
				}

				core::LandmarkCollection<vec2f> image_points;

				// Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
				for (int i = 0; i < landmarks.size(); ++i) {
					image_points.emplace_back(landmarks[i]);
				}

				landmark_collection_list.push_back(image_points);
			}

			return std::make_pair(landmark_collection_list, annotations);
		}
	}
}

#endif //EOS_LANDMARK_UTILS_H