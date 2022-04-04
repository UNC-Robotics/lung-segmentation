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
#ifndef LUNGSEG_IMAGE_UTILS_H
#define LUNGSEG_IMAGE_UTILS_H

#include <array>
#include <itkFlatStructuringElement.h>

#include "image_io.h"

namespace unc::robotics::lungseg::image {

using Flat3DStructuringElementType = itk::FlatStructuringElement<3>;

inline bool WithinImage(const IndexType& index, const SizeType& img_size) {
    for (unsigned i = 0; i < 3; ++i) {
        if (index[i] < 0 || index[i] >= img_size[i]) {
            return false;
        }
    }

    return true;
}

template<typename ImageType=Image3DType>
bool Empty(const typename ImageType::Pointer& img);

template<typename ImageType=Image3DType>
typename ImageType::Pointer NewImage(const typename ImageType::Pointer& img, const PixelType& init_val=0);

template<typename ImageType=Image3DType>
typename ImageType::Pointer Copy(const typename ImageType::Pointer& img);

template<typename ImageType=Image3DType>
typename ImageType::Pointer Smoothing(const typename ImageType::Pointer& img, const int& iter, const float& step);

template<typename ImageType=Image3DType, unsigned Dimension=3>
typename ImageType::Pointer MeanFiltering(const typename ImageType::Pointer& img, const unsigned radius);

template<typename ImageType=Image3DType>
typename ImageType::Pointer ConnectedRegionGrowing(const typename ImageType::Pointer& img, const typename ImageType::PixelType& min,
    const typename ImageType::PixelType& max, const typename ImageType::IndexType& seed,
    const bool& smoothing=false, const int& iter=10, const float& step=0.125);

template<typename ImageType=Image3DType>
typename ImageType::Pointer Maximum(const typename ImageType::Pointer& img0, const typename ImageType::Pointer& img1);

template<typename ImageType=Image3DType>
typename ImageType::Pointer Minimum(const typename ImageType::Pointer& img0, const typename ImageType::Pointer& img1);

template<typename ImageType=Image3DType>
typename ImageType::Pointer Subtraction(const typename ImageType::Pointer& img0, const typename ImageType::Pointer& img1);

template<typename ImageType>
typename ImageType::Pointer Addition(const typename ImageType::Pointer& img0, const typename ImageType::Pointer& img1);

template<typename ElementType, typename ImageType=Image3DType>
typename ImageType::Pointer Erosion(const typename ImageType::Pointer& img, const ElementType& se);

template<typename ImageType=Image3DType, unsigned Dimension=3>
typename ImageType::Pointer Erosion(const typename ImageType::Pointer& img, const unsigned& radius);

template<typename ElementType, typename ImageType=Image3DType>
typename ImageType::Pointer Dilation(const typename ImageType::Pointer& img, const ElementType& se);

template<typename ImageType=Image3DType, unsigned Dimension=3>
typename ImageType::Pointer Dilation(const typename ImageType::Pointer& img, const unsigned& radius);

template<typename ImageType=Image3DType, unsigned Dimension=3>
typename ImageType::Pointer MorphologicalOpening(const typename ImageType::Pointer& img, const unsigned& radius);

template<typename ImageType=Image3DType, unsigned Dimension=3>
typename ImageType::Pointer MorphologicalClosing(const typename ImageType::Pointer& img, const unsigned& radius);

template<typename ImageType=Image3DType, unsigned Dimension=3>
typename ImageType::Pointer GrayscaleClosing(const typename ImageType::Pointer& img, const unsigned& radius);

Flat3DStructuringElementType FlatStructure3D(const float& radius, const unsigned& type=0, 
    const bool& parametric_radius=false, const bool& anisotropy=false, const float& x=1, const float& y=1, const float& z=1);

template<bool Lower=false, typename ImageType=Image3DType>
typename ImageType::Pointer Thresholding(const typename ImageType::Pointer& img, const PixelType& threshold);

template<typename ImageType=Image3DType>
typename ImageType::Pointer Thresholding(const typename ImageType::Pointer& img,
    const std::optional<PixelType> min, const std::optional<PixelType> max);

template<bool MaskBackground=true, typename ImageType=Image3DType>
typename ImageType::Pointer Masking(const typename ImageType::Pointer& img, const typename ImageType::Pointer& mask, const PixelType& val=0);

template<typename ImageType=Image3DType>
typename ImageType::Pointer IslandRemoving(typename ImageType::Pointer& img, const unsigned& size);

template<typename ImageType=Image3DType>
typename ImageType::Pointer Invert(const typename ImageType::Pointer& img, const PixelType& base_val=255);

template<typename ImageType=Image3DType>
std::size_t RegionVolume(const typename ImageType::Pointer& img, const PixelType& val=255);

template<typename ImageType=Image3DType>
typename ImageType::Pointer Downsampling(const typename ImageType::Pointer& img, const unsigned& kernel_size);

template<typename ImageType=Image3DType>
typename ImageType::Pointer Saturation(const typename ImageType::Pointer& img, const PixelType& min, const PixelType& max);

template<typename ImageType=Image3DType>
std::pair<PixelType, PixelType> ComputeMinMaxIntensity(const typename ImageType::Pointer& img);

std::array<long unsigned, 6> GetRegionBoundary(const Image3DPtr& img, const PixelType& background, const SizeType& img_size);

Image2DPtr Extract(const Image3DPtr& img_3d, const unsigned& direction, const unsigned& index);

template<bool PreserveRange=false, typename ImageType=Image3DType>
typename ImageType::Pointer LinearRescale(const typename ImageType::Pointer& img, const PixelType& min, PixelType max);

Image3DPtr MaximumInsert(const Image2DPtr& img_2d, Image3DPtr img_3d, const unsigned& index);

template<typename ImageType=Image3DType>
typename ImageType::Pointer ScalarMultiplication(const typename ImageType::Pointer& img, const PixelType& scalar);

template<typename ImageType=Image3DType>
typename ImageType::Pointer MaurerDistanceTransform(const typename ImageType::Pointer& img);

template<typename ImageType=Image3DType>
PixelType AdaptiveRegionGrowing(const typename ImageType::Pointer& img, const typename ImageType::IndexType& seed,
    const PixelType& init_threshold, const std::size_t& max_volume, const float& max_percent);

template<typename ImageType=Image3DType>
typename ImageType::Pointer BinaryThinning(const typename ImageType::Pointer& img);

} // namespace unc::robotics::lungseg::image

#include "impl/image_utils.hpp"

#endif // LUNGSEG_IMAGE_UTILS_H