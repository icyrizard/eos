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
#include "eos/video/Keyframe.hpp"

#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <unistd.h>

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
	namespace video {
		// TODO: maybe move video iterator here.. or remove fil.
/**
 * @brief Computes the variance of laplacian of the given image or patch.
 *
 * This should compute the variance of the laplacian of a given image or patch, according to the 'LAPV'
 * algorithm of Pech 2000.
 * It is used as a focus or blurriness measure, i.e. to assess the quality of the given patch.
 *
 * @param[in] image Input image or patch.
 * @return The computed variance of laplacian score.
 */
double variance_of_laplacian(const cv::Mat& image)
{
	cv::Mat laplacian;
	cv::Laplacian(image, laplacian, CV_64F);

	cv::Scalar mu, sigma;
	cv::meanStdDev(laplacian, mu, sigma);

	double focus_measure = sigma.val[0] * sigma.val[0];
	return focus_measure;
};

/**
 *
 *
 * @tparam T
 */
class BufferedVideoIterator
{
public:
	int width;
	int height;
	Mat last_frame;
	int last_frame_number;

	bool reached_eof;

	std::unique_ptr<std::thread> frame_buffer_worker;

	BufferedVideoIterator() {};

	// TODO: build support for setting the amount of max_frames in the buffer.
	BufferedVideoIterator(std::string videoFilePath, fitting::ReconstructionData reconstruction_data, boost::property_tree::ptree settings) {
		std::ifstream file(videoFilePath);
		std::cout << "Opening video: " << videoFilePath << std::endl;

		if (!file.is_open()) {
			throw std::runtime_error("Error opening given file: " + videoFilePath);
		}

		cv::VideoCapture tmp_cap(videoFilePath); // open video file
		if (!tmp_cap.isOpened()) {
			throw std::runtime_error("Could not play video");
		}

		cap = tmp_cap;

		this->reconstruction_data = reconstruction_data;

		// copy settings from gathered from a .ini file
		min_frames = settings.get<int>("video.min_frames", 5);
		drop_frames = settings.get<int>("video.drop_frames", 0);
		skip_frames = settings.get<int>("video.skip_frames", 0);
		frames_per_bin = settings.get<unsigned int>("video.frames_per_bin", 2);

		unsigned int num_shape_coeff = reconstruction_data.morphable_model.get_shape_model().get_num_principal_components();

		this->num_shape_coefficients_to_fit = settings.get<unsigned int>(
			"video.num_shape_coefficients_to_fit", num_shape_coeff);

		// initialize bins
		bins.resize(num_yaw_bins);

		// reset frame count
		n_frames = 0;
		total_frames = 0;

		std::cout << "Settings: " << std::endl <<
			"min_frames: " << min_frames << std::endl <<
		    "drop_frames: " << drop_frames << std::endl <<
		    "frames_per_bin: " << frames_per_bin << std::endl <<
		    "num_shape_coefficients_to_fit: " << num_shape_coefficients_to_fit << std::endl;

		std::cout << "total frames in video: " << cap.get(CV_CAP_PROP_FRAME_COUNT) << std::endl;
	}

	bool is_playing() {
		return cap.isOpened();
	}

	/**
	 * Generate a new keyframe containing information about pose and landmarks
	 * These are needed to determine if we want the image in the first place.
	 *
	 * @param frame
	 * @return Keyframe
	 */
	Keyframe generate_new_keyframe(cv::Mat frame) {
		int frame_height = frame.rows;
		int frame_width = frame.cols;

		// Reached the end of the landmarks (only applicable for annotated videos):
		if (reconstruction_data.landmark_list.size() <= total_frames) {
			std::cout << "frame empty? " << frame.empty() << std::endl;
			std::cout << "Reached end of landmarks(" <<
				 reconstruction_data.landmark_list.size() << "/" << total_frames << ")" << std::endl;

			return Keyframe();
		}

		// Get the necessary information for reconstruction.
		auto landmarks = reconstruction_data.landmark_list[total_frames];
		auto landmark_mapper = reconstruction_data.landmark_mapper;
		auto blendshapes = reconstruction_data.blendshapes;
		auto morphable_model = reconstruction_data.morphable_model;

		vector<cv::Vec4f> model_points;
		vector<int> vertex_indices;
		vector<cv::Vec2f> image_points;

		// make a new one
		std::vector<float> blend_shape_coefficients;

		if (pca_shape_coefficients.empty()) {
			pca_shape_coefficients.resize(num_shape_coefficients_to_fit);
		}

		auto mesh = fitting::generate_new_mesh(
			morphable_model,
			blendshapes,
			pca_shape_coefficients, // current pca_coeff will be the mean for the first iterations.
			blend_shape_coefficients
		);

		// Will yield model_points, vertex_indices and frame_points
		// todo: should this function not come from mesh?
		core::get_mesh_coordinates(landmarks, landmark_mapper, mesh, model_points, vertex_indices, image_points);

		auto current_pose = fitting::estimate_orthographic_projection_linear(
			image_points, model_points, true, frame_height
		);

		fitting::RenderingParameters rendering_params(current_pose, frame_width, frame_height);
		fitting::FittingResult fitting_result;
		fitting_result.rendering_parameters = rendering_params;
		fitting_result.landmarks = landmarks;

		cv::Rect face_roi = core::get_face_roi(image_points, frame_width, frame_height);

		float frame_laplacian_score = static_cast<float>(variance_of_laplacian(frame(face_roi)));
		return Keyframe(frame_laplacian_score, frame, fitting_result, total_frames);
	}

	/**
	 * Try to add a new key frame. Look at the laplacian score and the yaw_angle. The yaw_angle will
	 * determine the bin in which it will be added. The score will be compared to the other frames in the
	 * same bin, if a bitter score is found, the frames are sorted on the score and cut-off on max frames.
	 *
	 * Todo: make sure this this cutting-off is done correct, we don't want faulty frames.
	 *
	 * @param keyframe
	 * @return
	 */
	bool try_add(Keyframe &keyframe) {
		// Determine whether to add or not:
		auto yaw_angle = glm::degrees(glm::yaw(keyframe.fitting_result.rendering_parameters.get_rotation()));
		auto idx = angle_to_index(yaw_angle);
		bool add_frame = false;

		keyframe.yaw_angle = yaw_angle;

		// Score is 0 for total black frames, we don't want those:
		if (keyframe.score == 0) {
			return false;
		}

		// always add when we don't have enough frames
		if (bins[idx].size() < frames_per_bin) {
			add_frame = true; // definitely adding - we wouldn't have to go through the for-loop on the next line.
		}

		for (auto&& f : bins[idx]) {
			if (keyframe.score > f.score) {
				add_frame = true;
			}
		}

		if (!add_frame) {
			return false;
		}

		// Add the keyframe:
		bins[idx].push_back(keyframe);
		if (bins[idx].size() > frames_per_bin) {
			n_frames--;
			// need to remove the lowest one:
			std::sort(std::begin(bins[idx]), std::end(bins[idx]),
					  [](const auto& lhs, const auto& rhs) { return lhs.score > rhs.score; });
			bins[idx].resize(frames_per_bin);
		}

		return true;
	}

	/**
	 * Builds string from current state of the buffer as informative as possible.
	 * TODO: string representation can be better, maybe add a max precision for floats.
	 *
	 * @return
	 */
	std::string to_string() const {
		std::string output = std::to_string(n_frames) + ": ";

		for (auto&& b : bins) {
			output.append("[");

			for (auto&& f : b) {
				std::size_t idx = angle_to_index(f.yaw_angle);
				output.append("(" + std::to_string(f.score));
				output.append("," + std::to_string(f.yaw_angle));
				output.append("), ");
			}

			output.append("],");
		}

		return output;
	}

	/**
	 * Get keyframes from all bins as a flattened list of keyframes.
	 *
	 * @return std::vector<Keyframe>
	 */
	std::vector<Keyframe> get_keyframes() const {
		std::vector<Keyframe> keyframes;
		for (auto&& b : bins) {
			for (auto&& f : b) {
				keyframes.push_back(f);
			}
		}

		return keyframes;
	}

	/**
	 * Start filling the buffer with frames and start a worker thread to keep doing so.
	 *
	 * @return
	 */
	std::vector<Keyframe> start() {
		// blocking calling next for the first time, stop if empty frame.
		if (!next()) {
			__stop();
		}

		// start a thread to keep getting new frames into the buffer from video:
		frame_buffer_worker = std::make_unique<std::thread>(&BufferedVideoIterator::video_iterator, this);
		frame_buffer_worker.get()->detach();

		return get_keyframes();
	}

	/**
	 * Fill the buffer by iterating through the video until the very last frame.
	 */
	void video_iterator() {
		// Go and fill the buffer with more frames while we are reconstructing.
		while (next()) { }

		// stop the video
		__stop();

		std::cout << "Video stopped at:" << total_frames << " frames - in buff " << n_frames <<  std::endl;
	}

	 /**
	  *
	  * Set next frame and return frame_buffer. Returns true or false if the next frame is empty.
	  * Empty frames mean that we reached the end of the video stream (i.e., file or camera).
	  *
	  * @return bool if
	  */
	bool next() {
		Mat frame = __get_new_frame();

		if (frame.empty()) {
			return false;
		}

		// keep the last frame here, so we can play a video subsequently:
		last_frame_number = total_frames;
		last_frame = frame;

		// TODO: only calculate lapscore within the bounding box of the face.
		auto keyframe = generate_new_keyframe(frame);
		bool frame_added = try_add(keyframe);

		if(frame_added) {
			n_frames++;

			// Setting that the buffer has changed:
			frame_buffer_changed = true;
			std::cout << "frame added(" << total_frames << "): " << keyframe.score << ", " << keyframe.yaw_angle << std::endl;
		}

		total_frames++;

		// fill up the buffer until we hit the minimum frames we want in the buffer.
		if(n_frames < min_frames && total_frames < 30) {
			std::cout << "not enough frames yet: " << n_frames << "/" << min_frames << std::endl;
			return next();
		}

		return true;
	}

	/**
	 * Update pca shape coeff. Probably we need to make something with a mutex, for updating and reading
	 * the pca_shape_coeff in the generate_new_keyframe functionality.
	 *
	 * @param pca_shape_coefficients
	 */
	void update_reconstruction_coeff(std::vector<float> pca_shape_coefficients) {
		std::cout << "-- Update pca coeff -- " << std::endl;
		this->pca_shape_coefficients = pca_shape_coefficients;
	}

	/**
	 *
	 * Converts a given yaw angle to an index in the internal bins vector.
	 * Assumes 9 bins and 20� intervals.
	 *
	 * @param yaw_angle
	 * @return
	 */
	static std::size_t angle_to_index(float yaw_angle)
	{
		if (yaw_angle <= -70.f)
			return 0;
		if (yaw_angle <= -50.f)
			return 1;
		if (yaw_angle <= -30.f)
			return 2;
		if (yaw_angle <= -10.f)
			return 3;
		if (yaw_angle <= 10.f)
			return 4;
		if (yaw_angle <= 30.f)
			return 5;
		if (yaw_angle <= 50.f)
			return 6;
		if (yaw_angle <= 70.f)
			return 7;
		return 8;
	};

	/**
	 * Only by calling this function, we return if the buffer has changed since last check.
	 *
	 * @return
	 */
	bool has_changed() {
		bool has_changed = frame_buffer_changed;

		// record the event if the buffer is changed since last time:
		frame_buffer_changed = false;

		return has_changed;
	}

	/**
	 * Get a new frame from the video source.
	 *
	 * @return cv::Mat
	 */
	Mat __get_new_frame() {
		// Get a new frame from camera.
		Mat frame;
		cap >> frame;

		return frame;
	};


	// todo: move to private if possible.
	/**
	 * Stop by releasing the VideoCapture.
	 */
	void __stop() {
		cap.release();
	};

private:
	cv::VideoCapture cap;
	std::deque<Keyframe> frame_buffer;
	eos::fitting::ReconstructionData reconstruction_data;

	using BinContent = std::vector<Keyframe>;
	std::vector<BinContent> bins;

	// latest pca_shape_coefficients
	std::vector<float> pca_shape_coefficients;

	std::size_t num_yaw_bins = 9;
	bool frame_buffer_changed = false;
	unsigned int frames_per_bin;

	// TODO: make set-able
	// total frames in processed, not persee in buffer (skipped frames)
	int total_frames;

	// total frames in buffer
	int n_frames;

	// number of frames to skip at before starting
	int skip_frames = 0;

	// minimum amount of frames to keep in buffer.
	int min_frames = 5;

	// Note: these settings are for future use
	int drop_frames = 0;

	unsigned int num_shape_coefficients_to_fit = 0;
};

}
}

#endif
