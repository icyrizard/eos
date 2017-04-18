#ifndef EOS_RECONSTRUCTIONDATA_HPP_
#define EOS_RECONSTRUCTIONDATA_HPP_

#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/morphablemodel/EdgeTopology.hpp"
#include "eos/fitting/contour_correspondence.hpp"

namespace eos {
namespace fitting {

struct ReconstructionData {
	morphablemodel::MorphableModel morphable_model;
	std::vector <morphablemodel::Blendshape> blendshapes;
	eos::core::LandmarkMapper landmark_mapper;
	std::vector <core::LandmarkCollection<cv::Vec2f>> landmark_list;
	eos::fitting::ModelContour model_contour;
	eos::fitting::ContourLandmarks contour_landmarks;
	eos::morphablemodel::EdgeTopology edge_topology;
};

}
}


#endif //EOS_RECONSTRUCTIONDATA_HPP
