//
// Created by RA Torenvliet on 12/04/2017.
//

#ifndef EOS_RECONSTRUCTIONDATA_HPP
#define EOS_RECONSTRUCTIONDATA_HPP

namespace eos {
namespace fitting {

struct ReconstructionData {
  morphablemodel::MorphableModel morphable_model;
  std::vector <morphablemodel::Blendshape> blendshapes;
  std::vector <core::LandmarkCollection<cv::Vec2f>> landmarks;
  std::vector <cv::Mat> affine_camera_matrix;

};

}
}


#endif //EOS_RECONSTRUCTIONDATA_HPP
