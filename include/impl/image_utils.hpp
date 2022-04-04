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

#include <itkImageBase.h>
#include <itkImageRegionIterator.h>
#include <itkCurvatureFlowImageFilter.h>
#include <itkConnectedThresholdImageFilter.h>
#include <itkMaximumImageFilter.h>
#include <itkMinimumImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkBinaryMorphologicalClosingImageFilter.h>
#include <itkBinaryMorphologicalOpeningImageFilter.h>
#include <itkGrayscaleErodeImageFilter.h>
#include <itkGrayscaleDilateImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkBinaryShapeOpeningImageFilter.h>
#include <itkImageDuplicator.h>
#include <itkThresholdImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkMeanImageFilter.h>
#include <itkGrayscaleMorphologicalOpeningImageFilter.h>
#include <itkGrayscaleMorphologicalClosingImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkSignedMaurerDistanceMapImageFilter.h>
#include <itkBinaryThinningImageFilter3D.h>

namespace unc::robotics::lungseg::image {

template<typename ImageType>
bool Empty(const typename ImageType::Pointer& img) {
    typename ImageType::RegionType region = img->GetLargestPossibleRegion();
    typename itk::ImageRegionIterator<ImageType> iter(img, region);
    iter.GoToBegin();

    while(!iter.IsAtEnd()) {
        if(iter.Get() != 0) {
            return false;
        }
        ++iter;
    }

    return true;
}

template<typename ImageType>
typename ImageType::Pointer NewImage(const typename ImageType::Pointer& img, const PixelType& init_val) {
    typename ImageType::Pointer new_img = ImageType::New();
    new_img->SetRegions(img->GetLargestPossibleRegion());
    new_img->SetSpacing(img->GetSpacing());
    new_img->SetOrigin(img->GetOrigin());
    new_img->SetDirection(img->GetDirection());

    new_img->Allocate();
    new_img->FillBuffer(init_val);

    return new_img;
}

template<typename ImageType>
typename ImageType::Pointer Copy(const typename ImageType::Pointer& img) {
    using DuplicatorType = itk::ImageDuplicator<ImageType>;

    typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
    duplicator->SetInputImage(img);

    try {
        duplicator->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while copying a image" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return duplicator->GetOutput();
}

template<typename ImageType>
typename ImageType::Pointer Smoothing(const typename ImageType::Pointer& img, const int& iter, const float& step) {
    using CurvatureFlowImageFilterType = itk::CurvatureFlowImageFilter<ImageType, ImageType>;
    
    typename CurvatureFlowImageFilterType::Pointer filter = CurvatureFlowImageFilterType::New();
    filter->SetInput(img);
    filter->SetNumberOfIterations(iter);
    filter->SetTimeStep(step);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while smoothing the image" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

template<typename ImageType, unsigned Dimension>
typename ImageType::Pointer MeanFiltering(const typename ImageType::Pointer& img, const unsigned radius) {
    using MeanFilterType = itk::MeanImageFilter<ImageType, ImageType>;

    typename MeanFilterType::Pointer filter = MeanFilterType::New();
    typename ImageType::SizeType index_rad;
    for (unsigned i = 0; i < Dimension; ++i) {
        index_rad[i] = radius;
    }
    filter->SetRadius(index_rad);
    filter->SetInput(img);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while mean filtering the image" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

template<typename ImageType>
typename ImageType::Pointer ConnectedRegionGrowing(const typename ImageType::Pointer& img, const typename ImageType::PixelType& min, 
    const typename ImageType::PixelType& max, const typename ImageType::IndexType& seed, const bool& smoothing, const int& iter, const float& step)
{
    using ConnectedFilterType = itk::ConnectedThresholdImageFilter<ImageType, ImageType>;

    typename ConnectedFilterType::Pointer filter = ConnectedFilterType::New();

    if (smoothing) {
        filter->SetInput(Smoothing<ImageType>(img, iter, step));
    }
    else {
        filter->SetInput(img);
    }

    filter->SetLower(min);
    filter->SetUpper(max);
    filter->SetReplaceValue(255);
    filter->SetSeed(seed);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while doing connected region growing" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

template<typename ImageType>
typename ImageType::Pointer Maximum(const typename ImageType::Pointer& img0, const typename ImageType::Pointer& img1) {
    if (img0 == nullptr) {
        return img1;
    }

    if (img1 == nullptr) {
        return img0;
    }

    using MaximumImageFilterType = itk::MaximumImageFilter<ImageType>;

    typename MaximumImageFilterType::Pointer filter = MaximumImageFilterType::New();
    filter->SetCoordinateTolerance(1e-4);
    filter->SetInput(0, img0);
    filter->SetInput(1, img1);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while maximum filtering" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

template<typename ImageType>
typename ImageType::Pointer Minimum(const typename ImageType::Pointer& img0, const typename ImageType::Pointer& img1) {
    if (img0 == nullptr) {
        return img1;
    }

    if (img1 == nullptr) {
        return img0;
    }

    using MinimumImageFilterType = itk::MinimumImageFilter<ImageType>;

    typename MinimumImageFilterType::Pointer filter = MinimumImageFilterType::New();
    filter->SetCoordinateTolerance(1e-4);
    filter->SetInput(0, img0);
    filter->SetInput(1, img1);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while minimum filtering" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

template<typename ImageType>
typename ImageType::Pointer Subtraction(const typename ImageType::Pointer& img0, const typename ImageType::Pointer& img1) {
    using SubtractType = itk::SubtractImageFilter<ImageType>;
    
    typename SubtractType::Pointer filter = SubtractType::New();
    filter->SetInput1(img0);
    filter->SetInput2(img1);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << "Exception thrown while substracting the image" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

template<typename ImageType>
typename ImageType::Pointer Addition(const typename ImageType::Pointer& img0, const typename ImageType::Pointer& img1) {
    using AddType = itk::AddImageFilter< ImageType, ImageType, ImageType>;

    typename AddType::Pointer filter = AddType::New();
    filter->SetInput1(img0);
    filter->SetInput2(img1);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << "Exception thrown while adding images" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

template<typename ElementType, typename ImageType>
typename ImageType::Pointer Erosion(const typename ImageType::Pointer& img, const ElementType& se) {
    using Erode3DImageFilterType = itk::GrayscaleErodeImageFilter<ImageType, ImageType, ElementType>;

    typename Erode3DImageFilterType::Pointer filter = Erode3DImageFilterType::New();
    filter->SetInput(img);
    filter->SetKernel(se);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while doing erosion" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

template<typename ImageType, unsigned Dimension>
typename ImageType::Pointer Erosion(const typename ImageType::Pointer& img, const unsigned& radius) {
    using StructuringElementType = itk::BinaryBallStructuringElement<PixelType, Dimension>;

    StructuringElementType kernel;
    kernel.SetRadius(radius);
    kernel.CreateStructuringElement();

    return Erosion<StructuringElementType, ImageType>(img, kernel);
}

template<typename ElementType, typename ImageType>
typename ImageType::Pointer Dilation(const typename ImageType::Pointer& img, const ElementType& se) {
    using Dilate3DImageFilterType = itk::GrayscaleDilateImageFilter<ImageType, ImageType, ElementType>;

    typename Dilate3DImageFilterType::Pointer filter = Dilate3DImageFilterType::New();
    filter->SetInput(img);
    filter->SetKernel(se);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while doing dilation" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

template<typename ImageType, unsigned Dimension>
typename ImageType::Pointer Dilation(const typename ImageType::Pointer& img, const unsigned& radius) {
    using StructuringElementType = itk::BinaryBallStructuringElement<PixelType, Dimension>;

    StructuringElementType kernel;
    kernel.SetRadius(radius);
    kernel.CreateStructuringElement();

    return Dilation<StructuringElementType, ImageType>(img, kernel);
}

template<typename ImageType, unsigned Dimension>
typename ImageType::Pointer MorphologicalOpening(const typename ImageType::Pointer& img, const unsigned& radius) {
    using StructuringElementType = itk::BinaryBallStructuringElement<PixelType, Dimension>;
    using OpeningFilterType = itk::BinaryMorphologicalOpeningImageFilter<ImageType, 
                              ImageType, StructuringElementType>;

    StructuringElementType ball_kernel;
    ball_kernel.SetRadius(radius);
    ball_kernel.CreateStructuringElement();

    typename OpeningFilterType::Pointer filter = OpeningFilterType::New();
    filter->SetInput(img);
    filter->SetKernel(ball_kernel);
    filter->SetForegroundValue(255);
    filter->SetBackgroundValue(0);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while doing morphological opening" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

template<typename ImageType, unsigned Dimension>
typename ImageType::Pointer MorphologicalClosing(const typename ImageType::Pointer& img, const unsigned& radius) {
    using StructuringElementType = itk::BinaryBallStructuringElement<PixelType, Dimension>;
    using ClosingFilterType = itk::BinaryMorphologicalClosingImageFilter<ImageType, 
                              ImageType, StructuringElementType>;

    StructuringElementType ball_kernel;
    ball_kernel.SetRadius(radius);
    ball_kernel.CreateStructuringElement();
    
    typename ClosingFilterType::Pointer filter = ClosingFilterType::New();
    filter->SetInput(img);
    filter->SetKernel(ball_kernel);
    filter->SetForegroundValue(255);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while doing morphological closing" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

template<typename ImageType, unsigned Dimension>
typename ImageType::Pointer GrayscaleClosing(const typename ImageType::Pointer& img, const unsigned& radius) {
    using StructuringElementType = itk::BinaryBallStructuringElement<PixelType, Dimension>;
    using ClosingFilterType = itk::GrayscaleMorphologicalClosingImageFilter<ImageType,
                              ImageType, StructuringElementType>;

    StructuringElementType kernel;
    kernel.SetRadius(radius);
    kernel.CreateStructuringElement();

    typename ClosingFilterType::Pointer filter = ClosingFilterType::New();
    filter->SetInput(img);
    filter->SetKernel(kernel);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while doing grayscale closing" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

Flat3DStructuringElementType FlatStructure3D(const float& radius, const unsigned& type, 
    const bool& parametric_radius, const bool& anisotropy, const float& x, const float& y, const float& z)
{
    Flat3DStructuringElementType::RadiusType r;

    if (anisotropy) {
        r[0] = x;
        r[1] = y;
        r[2] = z;
    }
    else {
        r.Fill( radius );
    }

    if(type == 0) {
        return Flat3DStructuringElementType::Ball(r, parametric_radius);
    }
    else if(type == 1) {
        return Flat3DStructuringElementType::Box(r);
    }
    else if(type == 2) {
        return Flat3DStructuringElementType::Cross(r);
    }
    else {
        std::cerr << "Undefined structure type" << std::endl;
        exit(1);
    }
}

template<bool Lower, typename ImageType>
typename ImageType::Pointer Thresholding(const typename ImageType::Pointer& img, const PixelType& threshold) {
    if constexpr (Lower) {
        return Thresholding<ImageType>(img, {}, threshold);
    }
    else {
        return Thresholding<ImageType>(img, threshold, {});
    }
}

template<typename ImageType>
typename ImageType::Pointer Thresholding(const typename ImageType::Pointer& img,
    const std::optional<PixelType> min, const std::optional<PixelType> max)
{
    using BinaryThresholdImageFilterType = itk::BinaryThresholdImageFilter<ImageType, ImageType>;

    typename BinaryThresholdImageFilterType::Pointer filter = BinaryThresholdImageFilterType::New();

    filter->SetInput(img);

    std::pair<PixelType, PixelType> default_val;
    if (!min || !max) {
        default_val = ComputeMinMaxIntensity<ImageType>(img);
    }

    if (min) {
        filter->SetLowerThreshold(*min);
    }
    else {
        filter->SetLowerThreshold(default_val.first - 1);
    }

    if (max) {
        filter->SetUpperThreshold(*max);
    }
    else {
        filter->SetUpperThreshold(default_val.second + 1);
    }

    if (filter->GetUpperThreshold() < filter->GetLowerThreshold()) {
        std::cout << "Invalid threshold" << std::endl;
        std::cout << default_val.first << ", " << default_val.second << std::endl;
        std::cout << filter->GetLowerThreshold() << ", " << filter->GetUpperThreshold() << std::endl;
        return NewImage<ImageType>(img);
    }

    filter->SetOutsideValue(0);
    filter->SetInsideValue(255);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << "Exception thrown while thresholding image" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

template<bool MaskBackground, typename ImageType>
typename ImageType::Pointer Masking(const typename ImageType::Pointer& img, const typename ImageType::Pointer& mask, const PixelType& val) {
    typename ImageType::Pointer result = NewImage<ImageType>(img);
    typename ImageType::RegionType region = img->GetLargestPossibleRegion();
    typename itk::ImageRegionIterator<ImageType> org_iter(img, region);
    typename itk::ImageRegionIterator<ImageType> mask_iter(mask, region);
    typename itk::ImageRegionIterator<ImageType> res_iter(result, region);
    org_iter.GoToBegin();
    mask_iter.GoToBegin();
    res_iter.GoToBegin();

    while(!org_iter.IsAtEnd()) {
        if constexpr (MaskBackground) {
            if(mask_iter.Get() == 0) {
                res_iter.Set(val);
            }
            else {
                res_iter.Set(org_iter.Get());
            }
        }
        else {
            if(mask_iter.Get() > 0) {
                res_iter.Set(val);
            }
            else {
                res_iter.Set(org_iter.Get());
            }
        }

        ++org_iter;
        ++mask_iter;
        ++res_iter;
    }

    try {
        result->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << "Exception thrown while masking image" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return result;
}

template<typename ImageType>
typename ImageType::Pointer IslandRemoving(typename ImageType::Pointer& img, const unsigned& size) {
    using BinaryOpeningType = itk::BinaryShapeOpeningImageFilter<ImageType>;

    typename BinaryOpeningType::Pointer filter = BinaryOpeningType::New();
    filter->SetInput(img);
    filter->SetForegroundValue(255);
    filter->SetBackgroundValue(0);
    filter->SetLambda(size);
    filter->SetFullyConnected(true);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while removing islands" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

template<typename ImageType>
typename ImageType::Pointer Invert(const typename ImageType::Pointer& img, const PixelType& base_val) {
    typename ImageType::Pointer base_img = NewImage<ImageType>(img, base_val);
    return Subtraction<ImageType>(base_img, img);
}

template<typename ImageType>
std::size_t RegionVolume(const typename ImageType::Pointer& img, const PixelType& val) {
    std::size_t counter = 0;
    typename itk::ImageRegionIterator<ImageType> iter(img, img->GetLargestPossibleRegion());
    iter.GoToBegin();

    while(!iter.IsAtEnd()) {
        if(iter.Get() == val) {
            counter++;
        }

        ++iter;
    }

    return counter;
}

template<typename ImageType>
typename ImageType::Pointer Downsampling(const typename ImageType::Pointer& img, const unsigned& kernel_size) {
    typename ImageType::RegionType region = img->GetLargestPossibleRegion();
    typename ImageType::SizeType size = region.GetSize();
    typename ImageType::SpacingType spacing = img->GetSpacing();
    typename ImageType::PointType origin = img->GetOrigin();
    typename ImageType::DirectionType direction = img->GetDirection();

    typename ImageType::SizeType new_size;
    new_size[0] = size[0] / kernel_size;
    new_size[1] = size[1] / kernel_size;
    new_size[2] = size[2] / kernel_size;
    typename ImageType::RegionType new_region;
    new_region.SetSize(new_size);

    typename ImageType::SpacingType new_spacing;
    new_spacing[0] = (size[0] * spacing[0]) / new_size[0];
    new_spacing[1] = (size[1] * spacing[1]) / new_size[1];
    new_spacing[2] = (size[2] * spacing[2]) / new_size[2];

    typename ImageType::Pointer downsampled = ImageType::New();
    downsampled->SetRegions(new_region);
    downsampled->SetSpacing(new_spacing);
    downsampled->SetOrigin(origin);
    downsampled->SetDirection(direction);
    downsampled->Allocate();
    downsampled->FillBuffer(0.0);

    typename ImageType::IndexType org_idx;
    typename ImageType::IndexType new_idx;

    for(unsigned x = 0, i = 0; x < size[0]; x += kernel_size, ++i) {
        for(unsigned y = 0, j = 0; y < size[1]; y += kernel_size, ++j) {
            for(unsigned z = 0, k =0; z < size[2]; z += kernel_size, ++k) {
                org_idx[0] = x;
                org_idx[1] = y;
                org_idx[2] = z;

                new_idx[0] = i;
                new_idx[1] = j;
                new_idx[2] = k;
                downsampled->SetPixel(new_idx, img->GetPixel(org_idx));
            }
        }
    }

    try {
        downsampled->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while downsampling image" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return downsampled;
}

template<typename ImageType>
typename ImageType::Pointer Saturation(const typename ImageType::Pointer& img, const PixelType& min, const PixelType& max) {
    using ThresholdImageFilterType = itk::ThresholdImageFilter<ImageType>;

    typename ThresholdImageFilterType::Pointer filter = ThresholdImageFilterType::New();
    filter->SetInput(img);
    filter->ThresholdBelow(min);
    filter->SetOutsideValue(min);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while thresholding the image" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    typename ImageType::Pointer tmp = filter->GetOutput();

    filter = ThresholdImageFilterType::New();
    filter->SetInput(tmp);
    filter->ThresholdAbove(max);
    filter->SetOutsideValue(max);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while thresholding the image" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

template<typename ImageType>
std::pair<PixelType, PixelType> ComputeMinMaxIntensity(const typename ImageType::Pointer& img) {
    using ImageCalculatorFilterType = itk::MinimumMaximumImageCalculator<ImageType>;

    typename ImageCalculatorFilterType::Pointer filter = ImageCalculatorFilterType::New();
    filter->SetImage(img);

    try {
        filter->Compute();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while while computing min/max intensity" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return {filter->GetMinimum(), filter->GetMaximum()};
}

std::array<long unsigned, 6> GetRegionBoundary(const Image3DPtr& img, const PixelType& background, const SizeType& img_size) {
    IndexType index;
    std::array<long unsigned, 6> boundary;
    
    boundary[0] = img_size[0];
    boundary[1] = 0;
    boundary[2] = img_size[1];
    boundary[3] = 0;
    boundary[4] = img_size[2];
    boundary[5] = 0;

    for (int x = 0; x < img_size[0]; x++) {
        for (int y = 0; y < img_size[1]; y++) {
            for (int z = 0; z < img_size[2]; z++) {
                index[0] = x;
                index[1] = y;
                index[2] = z;

                if (img->GetPixel(index) > background) {
                    if (z < boundary[4]) {
                        boundary[4] = z;
                    }
                    break;
                }
            }

            for (int z = img_size[2] - 1; z > 0; z--) {
                index[0] = x;
                index[1] = y;
                index[2] = z;

                if(img->GetPixel(index) > background) {
                    if(z > boundary[5]) {
                        boundary[5] = z;
                    }
                    break;
                }
            }
        }
    }

    for (int y = 0; y < img_size[1]; y++) {
        for (int z = 0; z < img_size[2]; z++) {
            for (int x = 0; x < img_size[0]; x++) {
                index[0] = x;
                index[1] = y;
                index[2] = z;

                if(img->GetPixel(index) > background) {
                    if (x < boundary[0]) {
                        boundary[0] = x;
                    }
                    break;
                }
            }

            for (int x = img_size[0] - 1; x > 0; x--) {
                index[0] = x;
                index[1] = y;
                index[2] = z;

                if(img->GetPixel(index) > background) {
                    if(x > boundary[1]) {
                        boundary[1] = x;
                    }
                    break;
                }
            }
        }
    }

    for (int z = 0; z < img_size[2]; z++) {
        for (int x = 0; x < img_size[0]; x++) {
            for (int y = 0; y < img_size[1]; y++) {
                index[0] = x;
                index[1] = y;
                index[2] = z;

                if(img->GetPixel(index) > background) {
                    if(y < boundary[2]) {
                        boundary[2] = y;
                    }
                    break;
                }
            }

            for (int y = img_size[1] - 1; y > 0; y--) {
                index[0] = x;
                index[1] = y;
                index[2] = z;

                if(img->GetPixel( index ) > background) {
                    if(y > boundary[3]) {
                        boundary[3] = y;
                    }
                    break;
                }
            }
        }
    }

    return boundary;
}

Image2DPtr Extract(const Image3DPtr& img_3d, const unsigned& direction, const unsigned& index) {
    bool success = false;
    while (!success) {
        Image3DType::RegionType org_region = img_3d->GetLargestPossibleRegion();
        Image3DType::SizeType size = org_region.GetSize();

        if(index > size[direction]) {
            throw std::runtime_error("Index out of range when extracting 2D slice!");
        }

        size[direction] = 0;
        Image3DType::IndexType start = org_region.GetIndex();
        start[direction] = index;
        Image3DType::RegionType region(start, size);

        using SliceFilterType = itk::ExtractImageFilter<Image3DType, Image2DType>;

        typename SliceFilterType::Pointer filter = SliceFilterType::New();
        filter->SetDirectionCollapseToSubmatrix();
        filter->SetExtractionRegion(region);
        filter->SetInput(img_3d);

        try {
            filter->UpdateLargestPossibleRegion();
        }
        catch (itk::ExceptionObject& excp) {
            std::cerr << __func__ << std::endl;
            std::cerr << "Exception thrown while filting the image" << std::endl;
            std::cerr << excp << std::endl;
            std::cerr << "Rerunning this function! Do not terminate!" << std::endl;

            continue;
        }

        success = true;
        return filter->GetOutput();
    }

    return nullptr;
}

template<bool PreserveRange, typename ImageType>
typename ImageType::Pointer LinearRescale(const typename ImageType::Pointer& img,
    const PixelType& min, PixelType max)
{
    using RescaleFilterType = itk::RescaleIntensityImageFilter<ImageType, ImageType>;

    typename RescaleFilterType::Pointer filter = RescaleFilterType::New();
    filter->SetInput(img);

    if constexpr (PreserveRange) {
        PixelType range_max = filter->GetInputMaximum() - filter->GetInputMinimum() + min;

        if (range_max > max) {
            max = range_max;
        }
    }

    filter->SetOutputMinimum(min);
    filter->SetOutputMaximum(max);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << "Exception thrown while rescaling the image" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

Image3DPtr MaximumInsert(const Image2DPtr& img_2d, Image3DPtr img_3d, const unsigned& index) {
    Image2DType::IndexType idx_2d;
    Image3DType::IndexType idx_3d;
    Image2DType::RegionType region_2d = img_2d->GetLargestPossibleRegion();
    Image2DType::SizeType size_2d = region_2d.GetSize();

    idx_3d[2] = index;

    for(unsigned i = 0; i < size_2d[0]; i++) {
        for(unsigned j = 0; j < size_2d[1]; j++) {
            idx_2d[0] = i;
            idx_2d[1] = j;
            PixelType& intensity = img_2d->GetPixel(idx_2d);

            idx_3d[0] = i;
            idx_3d[1] = j;

            if(intensity > img_3d->GetPixel(idx_3d)) {
                img_3d->SetPixel(idx_3d, intensity);
            }
        }
    }

    try {
        img_3d->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << "Exception thrown while inserting a slice" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return img_3d;
}

template<typename ImageType>
typename ImageType::Pointer ScalarMultiplication(const typename ImageType::Pointer& img, const PixelType& scalar) {
    using MultiplyFilterType = itk::MultiplyImageFilter<ImageType, ImageType, Image3DType>;

    typename MultiplyFilterType::Pointer filter = MultiplyFilterType::New();
    filter->SetInput(img);
    filter->SetConstant(scalar);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while doing multiplication" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

template<typename ImageType>
typename ImageType::Pointer MaurerDistanceTransform(const typename ImageType::Pointer& img) {
    using TransformImageFilterType = itk::SignedMaurerDistanceMapImageFilter<ImageType, ImageType>;

    typename TransformImageFilterType::Pointer filter = TransformImageFilterType::New();
    filter->SetInput(img);
    filter->InsideIsPositiveOn();
    filter->SquaredDistanceOff();

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while doing morphological distance transform" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return filter->GetOutput();
}

template<typename ImageType>
PixelType AdaptiveRegionGrowing(const typename ImageType::Pointer& img, const typename ImageType::IndexType& seed,
    const PixelType& init_threshold, const std::size_t& max_volume, const float& max_percent)
{
    PixelType seed_intensity = img->GetPixel(seed);
    auto [img_min, img_max] = ComputeMinMaxIntensity<ImageType>(img);
    PixelType T = init_threshold;
    typename ImageType::Pointer region = nullptr;
    std::size_t region_volume0, region_volume;

    if (seed_intensity > init_threshold) {
        region = ConnectedRegionGrowing<ImageType>(img, init_threshold, img_max, seed);
        region_volume0 = RegionVolume<ImageType>(region, 255);
        region_volume = region_volume0;
        float inc_percent = 0.0;

        while (region_volume < max_volume && inc_percent < max_percent) {
            T -= 1;
            region = ConnectedRegionGrowing<ImageType>(img, T, img_max, seed);
            region_volume = RegionVolume<ImageType>(region, 255);
            if (region_volume0 > 0.3 * max_volume) {
                inc_percent = (region_volume - region_volume0)/region_volume0;
            }
            region_volume0 = region_volume;
        }

        T += 1;
    }
    else {
        region = ConnectedRegionGrowing<ImageType>(img, img_min, init_threshold, seed);
        region_volume0 = RegionVolume<ImageType>(region, 255);
        region_volume = region_volume0;
        float inc_percent = 0.0;

        while (region_volume < max_volume && inc_percent < max_percent) {
            T += 1;
            region = ConnectedRegionGrowing<ImageType>(img, img_min, T, seed);
            region_volume = RegionVolume<ImageType>(region, 255);
            if (region_volume0 > 0.3 * max_volume) {
                inc_percent = (region_volume - region_volume0)/region_volume0;
            }
            region_volume0 = region_volume;
        }

        T -= 1;
    }

    return T;
}

template<typename ImageType>
typename ImageType::Pointer BinaryThinning(const typename ImageType::Pointer& img) {
    using ThinningFilterType = itk::BinaryThinningImageFilter3D<ImageType, ImageType>;

    typename ThinningFilterType::Pointer filter = ThinningFilterType::New();
    filter->SetInput(LinearRescale<false, ImageType>(img, 0, 1));

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while doing binary thinning" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return LinearRescale<false, ImageType>(filter->GetOutput(), 0, 255);
}

} // namespace unc::robotics::lungseg::image