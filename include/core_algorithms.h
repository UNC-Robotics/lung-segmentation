// BSD 3-Clause License

// Copyright (c) 2022, The University of North Carolina at Chapel Hill
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//! @author Mengyu Fu

#pragma once
#ifndef LUNGSEG_CORE_ALG_H
#define LUNGSEG_CORE_ALG_H

#include "image_io.h"
#include "seg_config.h"

namespace unc::robotics::lungseg {

using namespace image;

template<unsigned NumThreads, bool Printout=false>
void FullSegmentation(const Image3DPtr& input_img, const SegmentConfig& config);

std::vector<IndexType> GetTracheaSeeds(const Image3DPtr& input_img, const SizeType& img_size);

template<bool Printout=false>
Image3DPtr ComputeRegionMask(const Image3DPtr& img, const SegmentConfig& config, const std::vector<IndexType>& seeds);

template<bool SaveIntermediate=false, bool Printout=false>
Image3DPtr AirwayReconstruction(const Image3DPtr& img, const unsigned& min_scale, const std::array<long unsigned, 6>& roi,
	const unsigned& num_scales, const unsigned& scale_step, const SegmentConfig& config);

template<bool Printout=false>
Image2DPtr SliceIteration(const Image2DPtr& sliceJ, const Image2DPtr& sliceI, const unsigned& max_iter);

template<unsigned Mode, bool Printout=false>
std::pair<Image3DPtr, Image3DPtr> ComputeVesselness(const Image3DPtr& img, const Image3DPtr& mask, const Image3DPtr& dist_map);

template<bool Printout=false>
Image3DPtr VesselnessIowa(const Image3DPtr& img, const double& sigma, const bool& output_binary=false,
	const PixelType& upper_bound=300, const PixelType& clearance=100, const PixelType& threshold=0.07);

template<bool Printout=false>
Image3DPtr VesselnessFrangi(const Image3DPtr& img, const double& sigma, const bool& output_binary=false,
    const PixelType& alpha=0.5, const PixelType& beta=0.5, const PixelType& gamma=70.0, 
    const PixelType& lower_bound=-1200, const PixelType& threshold=0.17);

Image3DPtr UpdateScale(const Image3DPtr& diff_img, const Image3DPtr& old_scale_map, const PixelType& new_scale);

template<bool Printout=false>
Image3DPtr ComputeLargeBronchialTube(const Image3DPtr& img, const IndexType& seed, const SegmentConfig& config);

template<bool Printout=false>
Image3DPtr RefineAirway(const Image3DPtr& img, const Image3DPtr& region, const IndexType& seed,
    const unsigned& radius, const PixelType& min, const PixelType& max, const PixelType& replace_val);

template<bool Printout=false>
Image3DPtr WallReconstruction(const Image3DPtr& img, const Image3DPtr& mask, const Image3DPtr& airway,
    const SegmentConfig& config, const unsigned& max_iter);

template<bool Printout=false>
void SaveImage(const Image3DPtr& img, const std::string& img_name, const SegmentConfig& config);

} // unc::robotics::lungseg

#include "impl/core_algorithms.hpp"

#endif // LUNGSEG_CORE_ALG_H