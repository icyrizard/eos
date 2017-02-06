/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/core/BufferedVideoIterator.hpp
 *
 * Copyright 2014, 2015 Patrik Huber
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

#ifndef BUFFERED_VIDEO_ITERATOR_HPP_
#define BUFFERED_VIDEO_ITERATOR_HPP_

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vector>
#include <string>

using eos::core::Landmark;
using eos::core::LandmarkCollection;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::vector;
using std::string;

using namespace cv;

namespace eos {
	namespace core {

		template<typename T>
		class BufferedVideoIterator {
		public:
			BufferedVideoIterator() {};

			BufferedVideoIterator(std::string videoFilePath) {
				std::cout << "capturing:" << videoFilePath << std::endl;;

				std::ifstream file(videoFilePath, std::ios::binary);
				if (file.fail()) {
					throw std::runtime_error("Error opening given file: " + videoFilePath);
				}

				cv::VideoCapture tmp_cap(videoFilePath); // open the default camera

				if (!tmp_cap.isOpened()) { // check if we succeeded
					throw std::runtime_error("Could not play video");
				}

				cap = tmp_cap;
			}

			std::deque <T> next() {
				int frame_buffer_length = frame_buffer.size();
				int landmark_landmark_length = landmark_buffer.size();

				// Get a new frame from camera.
				Mat frame;
				cap >> frame;

				if (n_frames >= max_frames) {
					frame_buffer.pop_front();
					landmark_buffer.pop_front();
				}

				std::cout << "frame_buff" << frame.empty() << std::endl;
				if (frame.empty() == 0) {
					std::cout << "derpio" << std::endl;
					frame_buffer.push_back(n_frames);
					landmark_buffer.push_back(n_frames);
				}

				std::cout << "frame_buff" << frame_buffer.empty() << std::endl;

				n_frames++;


				return frame_buffer;
			}

			std::deque <T> get_frame_buffer() {
				return frame_buffer;
			}

			std::deque <T> get_landmark_buffer() {
				return landmark_buffer;
			}

		private:
			cv::VideoCapture cap;
			std::deque <T> frame_buffer;
			std::deque <T> landmark_buffer;

			int n_frames = 0;
			int max_frames = 5;
		};
	}
}

#endif
