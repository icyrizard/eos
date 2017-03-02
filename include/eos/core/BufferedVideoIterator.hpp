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

using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::vector;
using std::string;

namespace fs = boost::filesystem;
using namespace cv;
using namespace std;

namespace eos {
	namespace core {
		/**
		 * BufferedVideo Iterator will keep a buffer of the last seen n_frames. By calling .next() it will load a new
		 * frame from the given video and you will get a pointer to the front of the buffer (which has n_frames).
		 *
		 * Just imagine a sliding window accross the video, this is what we aim to implement here.
		 *
		 * Example:
		 *		vid_iterator = bufferedvideoiterator<cv::mat>(videofile.string(), landmark_annotation_list);
		 *
		 *		std::deque<cv::mat> frames = vid_iterator.next();
		 *		while(!(frames.empty())) {
		 *			for (std::deque<cv::mat>::iterator it = frames.begin(); it!=frames.end(); ++it) {
		 *				std::cout << ' ' << *it;
		 *			}
		 *
		 *			frames = vid_iterator.next();
		 *		}
		 *
		 * @tparam T
		 */
		// Note for this template: later we can use other templates for easy testing (not using cv:Mat but <int> for example).
		template<typename T>
		class BufferedVideoIterator {
		public:
			BufferedVideoIterator() {};

			// TODO: build support for setting the amount of max_frames in the buffer.
			BufferedVideoIterator(std::string videoFilePath, std::vector<std::vector<cv::Vec2f>> landmarks) {
				std::ifstream file(videoFilePath);

				if (!file.is_open()) {
					throw std::runtime_error("Error opening given file: " + videoFilePath);
				}

				cv::VideoCapture tmp_cap(videoFilePath); // open video file

				if (!tmp_cap.isOpened()) { // check if we succeeded
					throw std::runtime_error("Could not play video");
				}

				cap = tmp_cap;
				this->landmarks = landmarks;
			}

			/**
			 * Set next frame and return frame_buffer.
			 *
			 * @return dequeue<Mat> frame buffer.
			 *
			 * TODO: build support for returning landmarks AND frames.
			 */
			std::deque <Mat> next() {
				long frame_buffer_length = frame_buffer.size();
				long landmark_landmark_length = landmark_buffer.size();

				// Get a new frame from camera.
				Mat frame;
				cap >> frame;

				// Pop if we exceeded max_frames.
				if (n_frames > max_frames) {
					frame_buffer.pop_front();
				}

				if (frame.empty() == 0) {
					frame_buffer.push_back(frame);
				}

				n_frames++;

				return frame_buffer;
			}

			std::deque <Mat> get_frame_buffer() {
				return frame_buffer;
			}

			std::deque <Mat> get_landmark_buffer() {
				return landmark_buffer;
			}

		private:
			cv::VideoCapture cap;
			std::deque <Mat> frame_buffer;
			std::deque <Mat> landmark_buffer;
			std::vector<std::vector<cv::Vec2f>> landmarks;

			// TODO: make set-able
			// load n_frames at once into the buffer.
			long n_frames = 1;
			// keep max_frames into the buffer.
			long max_frames = 5;
		};
	}
}

#endif
