/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/video/Keyframe.hpp
 *
 * Copyright 2016, 2017 Patrik Huber
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

#ifndef KEYFRAME_HPP_
#define KEYFRAME_HPP_

#include "eos/fitting/FittingResult.hpp"
#include "eos/fitting/RenderingParameters.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/render/texture_extraction.hpp"

#include <boost/property_tree/ptree.hpp>

#include "opencv2/core/core.hpp"

#include <deque>

namespace eos {
namespace video {

/**
 * @brief A keyframe selected by the fitting algorithm.
 *
 * Contains the original frame, all necessary fitting parameters, and a score.
 */
class Keyframe {
public:
	Keyframe() {}
	/**
	 * Only used when only a frame is available.
	 *
	 * @param frame
	 */
	Keyframe(cv::Mat frame) {
		this->frame = frame;
	}

	/**
	 * Only used when score and frame is available
	 *
	 * @param frame
	 * @param score
	 */
	Keyframe(float score, cv::Mat frame, int frame_number) {
		this->score = score;
		this->frame = frame;
		this->frame_number = frame_number;
	}

	/**
	 *
	 * @param score
	 * @param frame
	 * @param fitting_result
	 */
	Keyframe(float score, cv::Mat frame, fitting::FittingResult fitting_result) {
		this->frame = frame;
		this->score = score;
		this->fitting_result = fitting_result;
	}

	cv::Mat frame;
	int frame_number;
	float score = 0.0f;

	fitting::FittingResult fitting_result;
};

/**
 * @brief A keyframe selection that selects keyframes according to yaw pose and score.
 *
 * Separates the +-90� yaw pose range into 20� intervals (i.e. 90 to 70, ..., -10 to 10, ...), and puts frames
 * into each bin, until full. Replaces keyframes with better frames if the score is higher than that of
 * current keyframes.
 *
 * The yaw pose bins are currently hard-coded (9 bins, 20� intervals).
 */
struct PoseBinningKeyframeSelector
{
public:
    PoseBinningKeyframeSelector(int frames_per_bin = 2) : frames_per_bin(frames_per_bin)
    {
        bins.resize(num_yaw_bins);
    };

    bool try_add(float frame_score, cv::Mat image, const fitting::FittingResult& fitting_result)
    {
        // Determine whether to add or not:
        auto yaw_angle = glm::degrees(glm::yaw(fitting_result.rendering_parameters.get_rotation()));
        auto idx = angle_to_index(yaw_angle);
        bool add_frame = false;
        if (bins[idx].size() < frames_per_bin) // always add when we don't have enough frames
            add_frame =
                true; // definitely adding - we wouldn't have to go through the for-loop on the next line.
        for (auto&& f : bins[idx])
        {
            if (frame_score > f.score)
                add_frame = true;
        }
        if (!add_frame)
        {
            return false;
        }
        // Add the keyframe:
        bins[idx].push_back(video::Keyframe{frame_score, image, fitting_result});
        if (bins[idx].size() > frames_per_bin)
        {
            // need to remove the lowest one:
            std::sort(std::begin(bins[idx]), std::end(bins[idx]),
                      [](const auto& lhs, const auto& rhs) { return lhs.score > rhs.score; });
            bins[idx].resize(frames_per_bin);
        }
        return true;
    };

    // Returns the keyframes as a vector.
    std::vector<Keyframe> get_keyframes() const
    {
        std::vector<Keyframe> keyframes;
        for (auto&& b : bins)
        {
            for (auto&& f : b)
            {
                keyframes.push_back(f);
            }
        }
        return keyframes;
    };

private:
    using BinContent = std::vector<Keyframe>;
    std::vector<BinContent> bins;
    const int num_yaw_bins = 9;
    int frames_per_bin;

    // Converts a given yaw angle to an index in the internal bins vector.
    // Assumes 9 bins and 20� intervals.
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
};

/**
 * @brief Extracts texture from each keyframe and merges them using a weighted mean.
 *
 * Uses the view angle as weighting.
 *
 * Note 1: Would be nice to eventually return a 4-channel texture map, with a sensible weight in the 4th
 * channel (i.e. the max of all weights for a given pixel).
 *
 * Note 2: On each call to this, it generates all isomaps. This is quite time-consuming (and we could compute
 * the weighted mean incrementally). But caching them is not trivial (maybe with a hashing or comparing the
 * cv::Mat frame data* member?).
 * On the other hand, for the more complex merging techniques (super-res, involving ceres, or a median
 * cost-func?), there might be no caching possible anyway and we will recompute the merged isomap from scratch
 * each time anyway, but not by first extracting all isomaps - instead we would just do a lookup of the
 * required pixel value(s) in the original image.
 *
 * // struct KeyframeMerger {};
 *
 * @param[in] keyframes The keyframes that will be merged.
 * @param[in] morphable_model The Morphable Model with which the keyframes have been fitted.
 * @param[in] blendshapes The blendshapes with which the keyframes have been fitted.
 * @return Merged texture map (isomap), 3-channel uchar.
 */
cv::Mat merge_weighted_mean(const std::vector<Keyframe>& keyframes,
                            const morphablemodel::MorphableModel& morphable_model,
                            const std::vector<morphablemodel::Blendshape>& blendshapes)
{
    assert(keyframes.size() >= 1);

    using cv::Mat;
    using std::vector;

	vector<Mat> isomaps;
	for (const auto& frame_data : keyframes) {
	//		VectorXf current_pca_shape = morphable_model.get_shape_model().draw_sample(frame_data.fitting_result.pca_shape_coefficients);
	//		const current_combined_shape = current_pca_shape + morphablemodel::to_matrix(blendshapes) * Eigen::Map<const Eigen::VectorXf>(frame_data.fitting_result.blendshape_coefficients);
		const Eigen::VectorXf shape =
			morphable_model.get_shape_model().draw_sample(frame_data.fitting_result.pca_shape_coefficients) +
			morphablemodel::to_matrix(blendshapes) * morphablemodel::to_vector(frame_data.fitting_result.blendshape_coefficients);
		const auto mesh =
			morphablemodel::sample_to_mesh(
					shape, {}, morphable_model.get_shape_model().get_triangle_list(), {}, morphable_model.get_texture_coordinates());
		const Mat affine_camera_matrix = fitting::get_3x4_affine_camera_matrix(
			frame_data.fitting_result.rendering_parameters, frame_data.frame.cols, frame_data.frame.rows
		);
		const Mat isomap = render::extract_texture(
			mesh, affine_camera_matrix, frame_data.frame, true, render::TextureInterpolation::NearestNeighbour, 1024
		);
		isomaps.push_back(isomap);
	}

    // Now do the actual merging:
    Mat r = Mat::zeros(isomaps[0].rows, isomaps[0].cols, CV_32FC1);
    Mat g = Mat::zeros(isomaps[0].rows, isomaps[0].cols, CV_32FC1);
    Mat b = Mat::zeros(isomaps[0].rows, isomaps[0].cols, CV_32FC1);
    Mat accumulated_weight = Mat::zeros(isomaps[0].rows, isomaps[0].cols, CV_32FC1);
    // Currently, this just uses the weights in the alpha channel for weighting - they contain only the
    // view-angle. We should use the keyframe's score as well. Plus the area of the source triangle.
    for (auto&& isomap : isomaps)
    {
        vector<Mat> channels;
        cv::split(isomap, channels);
        // channels[0].convertTo(channels[0], CV_32FC1);
        // We could avoid this explicit temporary, but then we'd have to convert both matrices
        // to CV_32FC1 first - and manually multiply with 1/255. Not sure which one is faster.
        // If we do it like this, the add just becomes '+=' - so I think it's fine like this.
        // The final formula is:
        // b += chan_0 * alpha * 1/255; (and the same for g and r respectively)
        Mat weighted_b, weighted_g, weighted_r;
        // // we scale the weights from [0, 255] to [0, 1]:
        cv::multiply(channels[0], channels[3], weighted_b, 1 / 255.0, CV_32FC1);
        cv::multiply(channels[1], channels[3], weighted_g, 1 / 255.0, CV_32FC1);
        cv::multiply(channels[2], channels[3], weighted_r, 1 / 255.0, CV_32FC1);
        b += weighted_b;
        g += weighted_g;
        r += weighted_r;
        channels[3].convertTo(channels[3], CV_32FC1); // needed for the '/ 255.0f' below to work
        cv::add(accumulated_weight, channels[3] / 255.0f, accumulated_weight, cv::noArray(), CV_32FC1);
    }
    b = b.mul(1.0 / (accumulated_weight)); // divide by number of frames used too?
    g = g.mul(1.0 / (accumulated_weight));
    r = r.mul(1.0 / (accumulated_weight));

    // Let's return accumulated_weight too: Normalise by num_isomaps * 255 (=maximum weight)
    // This sets the returned weight to the average from all the isomaps. Maybe the maximum would make more
    // sense? => Not returning anything for now.
    // accumulated_weight = (accumulated_weight / isomaps.size()) * 255;

    Mat merged_isomap;
    cv::merge({b, g, r}, merged_isomap);
    merged_isomap.convertTo(merged_isomap, CV_8UC3);
    return merged_isomap;
};

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
 * BufferedVideo Iterator will keep a buffer of the last seen n_frames. By calling .next() it will load a new
 * frame from the given video and you will get a pointer to the front of the buffer (which has n_frames).
 *
 * Just imagine a sliding window accross the video, this is what we aim to implement here.
 *
 * Example:
 *    vid_iterator = bufferedvideoiterator<cv::mat>(videofile.string(), landmark_annotation_list);
 *
 *    std::deque<cv::mat> frames = vid_iterator.next();
 *    while(!(frames.empty())) {
 *        for (std::deque<cv::mat>::iterator it = frames.begin(); it!=frames.end(); ++it) {
 *            std::cout << ' ' << *it;
 *        }
 *
 *         frames = vid_iterator.next();
 *    }
 *
 * @tparam T
 */
// Note for this template: later we can use other templates for easy testing (not using cv:Mat but <int> for example).
class BufferedVideoIterator {
public:
	int width;
	int height;

	BufferedVideoIterator() {};

	// TODO: build support for setting the amount of max_frames in the buffer.
	BufferedVideoIterator(std::string videoFilePath, boost::property_tree::ptree settings) {
		std::ifstream file(videoFilePath);
		std::cout << "video file path: " << videoFilePath << std::endl;
		if (!file.is_open()) {
			throw std::runtime_error("Error opening given file: " + videoFilePath);
		}

		cv::VideoCapture tmp_cap(videoFilePath); // open video file

		if (!tmp_cap.isOpened()) { // check if we succeeded
			throw std::runtime_error("Could not play video");
		}

		this->cap = tmp_cap;
		this->max_frames = settings.get<int>("video.max_frames", 5);
		this->min_frames = settings.get<int>("video.min_frames", 5);
		this->drop_frames = settings.get<int>("video.drop_frames", 0);
		this->laplacian_threshold = settings.get<int>("video.blur_threshold", 1000);

		// TODO: Implement this.
		this->skip_frames = settings.get<int>("video.skip_frames", 0);
	}

	/**
	 * Set next frame and return frame_buffer.
	 *
	 * @return dequeue<Mat> frame buffer.
	 *
	 * TODO: build support for returning landmarks AND frames.
	 */
	std::deque <Keyframe> next() {
		long frame_buffer_length = frame_buffer.size();

		// Get a new frame from camera.
		cv::Mat frame;
		cap >> frame;

		// Pop if we exceeded max_frames.
		if (n_frames > max_frames) {
			frame_buffer.pop_front();
			n_frames--;
		}

		float frame_laplacian_score = static_cast<float>(variance_of_laplacian(frame));

		if (frame_laplacian_score < laplacian_threshold && frame_laplacian_score > 0) {
			frame_buffer.push_back(Keyframe(frame_laplacian_score, frame, total_frames_processed));
			n_frames++;
			std::cout << total_frames_processed << ": laplacian score " << frame_laplacian_score << std::endl;
		} else {
			std::cout << total_frames_processed << ": skipping frame(";
			if (frame_laplacian_score == 0) {
				std::cout << "total black): " << frame_laplacian_score << std::endl;
			} else {
				std::cout << "too blurry): " << frame_laplacian_score << std::endl;
			}

		}

		total_frames_processed++;

		// fill up the buffer until we hit the minimum frames we want in the buffer.
		if(n_frames < min_frames) {
			frame_buffer = next();
		}

		return frame_buffer;
	}

	std::deque <Keyframe> get_frame_buffer() {
		return frame_buffer;
	}

private:
	cv::VideoCapture cap;
	std::deque<Keyframe> frame_buffer;

	// TODO: make set-able
	// total frames in processed, not persee in buffer (skipped frames)
	int total_frames_processed = 0;

	// total frames in buffer
	int n_frames = 0;

	// number of frames to skip at before starting
	int skip_frames = 0;

	// minimum amount of frames to keep in buffer.
	int min_frames = 5;

	// keep max_frames into the buffer.
	int max_frames = 5;

	// Note: these settings are for future use
	int drop_frames = 0;

	// laplacian threshold
	double laplacian_threshold = 10000000;
};

/**
* @brief Merges isomaps from a live video with a weighted averaging, based
* on the view angle of each vertex to the camera.
*
* An optional merge_threshold can be specified upon construction. Pixels with
* a view-angle above that threshold will be completely discarded. All pixels
* below the threshold are merged with a weighting based on its vertex view-angle.
* Assumes the isomaps to be 512x512.
*/
class WeightedIsomapAveraging
{
public:
	/**
	 * @brief Constructs a new object that will hold the current averaged isomap and
	 * be able to add frames from a live video and merge them on-the-fly.
	 *
	 * The threshold means: Each triangle with a view angle smaller than the given angle will be used to merge.
	 * The default threshold (90°) means all triangles, as long as they're a little bit visible, are merged.
	 *
	 * @param[in] merge_threshold View-angle merge threshold, in degrees, from 0 to 90.
	 */
	WeightedIsomapAveraging(float merge_threshold = 90.0f)
	{
		assert(merge_threshold >= 0.f && merge_threshold <= 90.f);

		visibility_counter = cv::Mat::zeros(512, 512, CV_32SC1);
		merged_isomap = cv::Mat::zeros(512, 512, CV_32FC4);

		// map 0° to 255, 90° to 0:
		float alpha_thresh = (-255.f / 90.f) * merge_threshold + 255.f;
		if (alpha_thresh < 0.f) // could maybe happen due to float inaccuracies / rounding?
			alpha_thresh = 0.0f;
		threshold = static_cast<unsigned char>(alpha_thresh);
	};

	/**
	 * @brief Merges the given new isomap with all previously processed isomaps.
	 *
	 * @param[in] isomap The new isomap to add.
	 * @return The merged isomap of all images processed so far, as 8UC4.
	 */
	cv::Mat add_and_merge(const cv::Mat& isomap)
	{
		// Merge isomaps, add the current to the already merged, pixel by pixel:
		for (int r = 0; r < isomap.rows; ++r)
		{
			for (int c = 0; c < isomap.cols; ++c)
			{
				if (isomap.at<cv::Vec4b>(r, c)[3] <= threshold)
				{
					continue; // ignore this pixel, not visible in the extracted isomap of this current frame
				}
				// we're sure to have a visible pixel, merge it:
				// merged_pixel = (old_average * visible_count + new_pixel) / (visible_count + 1)
				merged_isomap.at<cv::Vec4f>(r, c)[0] = (merged_isomap.at<cv::Vec4f>(r, c)[0] * visibility_counter.at<int>(r, c) + isomap.at<cv::Vec4b>(r, c)[0]) / (visibility_counter.at<int>(r, c) + 1);
				merged_isomap.at<cv::Vec4f>(r, c)[1] = (merged_isomap.at<cv::Vec4f>(r, c)[1] * visibility_counter.at<int>(r, c) + isomap.at<cv::Vec4b>(r, c)[1]) / (visibility_counter.at<int>(r, c) + 1);
				merged_isomap.at<cv::Vec4f>(r, c)[2] = (merged_isomap.at<cv::Vec4f>(r, c)[2] * visibility_counter.at<int>(r, c) + isomap.at<cv::Vec4b>(r, c)[2]) / (visibility_counter.at<int>(r, c) + 1);
				merged_isomap.at<cv::Vec4f>(r, c)[3] = 255; // as soon as we've seen the pixel visible once, we set it to visible.
				++visibility_counter.at<int>(r, c);
			}
		}
		cv::Mat merged_isomap_uchar;
		merged_isomap.convertTo(merged_isomap_uchar, CV_8UC4);
		return merged_isomap_uchar;
	};

private:
	cv::Mat visibility_counter;
	cv::Mat merged_isomap;
	unsigned char threshold;
};

/**
* @brief Merges PCA coefficients from a live video with a simple averaging.
*/
class PcaCoefficientMerging
{
public:
/**
 * @brief Merges the given new PCA coefficients with all previously processed coefficients.
 *
 * @param[in] coefficients The new coefficients to add.
 * @return The merged coefficients of all images processed so far.
 */
std::vector<float> add_and_merge(const std::vector<float>& coefficients)
{
	if (merged_shape_coefficients.empty()) {
		merged_shape_coefficients = cv::Mat::zeros(coefficients.size(), 1, CV_32FC1);
	}

	assert(coefficients.size() == merged_shape_coefficients.rows);

	cv::Mat test(coefficients);
	merged_shape_coefficients = (merged_shape_coefficients * num_processed_frames + test) / (num_processed_frames + 1.0f);
	++num_processed_frames;
	return std::vector<float>(merged_shape_coefficients.begin<float>(), merged_shape_coefficients.end<float>());
};

private:
	int num_processed_frames = 0;
	cv::Mat merged_shape_coefficients;
};


	} /* namespace video */
} /* namespace eos */

#endif /* KEYFRAME_HPP_ */
