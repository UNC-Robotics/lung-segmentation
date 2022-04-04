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

#include <cmath>
#include <vector>
#include <iostream>
#include <mutex>
#include <chrono>

#include <omp.h>
#include <itkMultiThreaderBase.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkHessianRecursiveGaussianImageFilter.h>
#include <itkSymmetricSecondRankTensor.h>
#include <itkSymmetricEigenAnalysis.h>

#include "../image_utils.h"
#include "../image_io.h"
#include "../utils.h"

namespace unc::robotics::lungseg {

using namespace image;

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

template<unsigned NumThreads, bool Printout>
void FullSegmentation(const Image3DPtr& input_img, const SegmentConfig& config) {
    const TimePoint start_time = Clock::now();

    itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(NumThreads);
    if constexpr (Printout) {
        std::cout << "Set global maximum number of threads to " << NumThreads << std::endl;
    }

    bool valid_task = false;
    for (auto const& task : config.procedure_control) {
        if (task) {
            valid_task = true;
            break;
        }
    }

    if (!valid_task) {
        return;
    }

    SizeType img_size = input_img->GetLargestPossibleRegion().GetSize();
    SpacingType img_spacing = input_img->GetSpacing();

    std::vector<IndexType> seeds;
    if (config.ComputeRegionMask() || config.ComputeBronchialTree()) {
        seeds = GetTracheaSeeds(input_img, img_size);
    }

    /********************************
    Compute region mask
    *********************************/

    Image3DPtr region_mask = nullptr;

    if (config.ComputeRegionMask()) {
        if constexpr (Printout) {
            std::cout << "Computing region mask..." << std::endl;
        }
        region_mask = ComputeRegionMask<Printout>(input_img, config, seeds);
    }
    else {
        if constexpr (Printout) {
            std::cout << "Loading existing region mask from " << config.intermediate_dir << "..." << std::endl;
        }
        
        region_mask = io::ReadNII<Printout>(config.intermediate_dir, "RegionMask");
    }

    if (Empty(region_mask)) {
        throw std::runtime_error("[FullSegmentation] Region mask is empty!");
    }

    /********************************
    Downsampling if needed
    *********************************/

    Image3DPtr img = Copy(input_img);

    if (config.downsampling && img_spacing[0] < 0.5 && img_spacing[2] < 1.0) {
        if constexpr (Printout) {
            std::cout << "Doing downsampling for high res image..." << std::endl;
        }
        
        img = Downsampling(img, 2);
        region_mask = Downsampling(region_mask, 2);
        img_size = img->GetLargestPossibleRegion().GetSize();
        img_spacing = img->GetSpacing();

        for (IndexType& seed : seeds) {
            seed[0] /=2;
            seed[1] /=2;
            seed[2] /=2;
        }

        SaveImage<Printout>(img, "Downsampled", config);
    }

    if (config.ComputeRegionMask()) {
        SaveImage<Printout>(region_mask, "RegionMask", config);
    }

    if (!config.ComputeAirwayMap() && !config.ComputeBronchialTree() && !config.ComputeMajorVessels()
        && !config.ComputeVesselnessMap() && !config.ComputeAllObstacles() && !config.ComputeAirwaySkeleton())
    {
        return;
    }

    Image3DPtr org_img = Copy(img);
    img = Masking(img, region_mask, -1200);
    SaveImage<Printout>(img, "Masked", config);

    PixelType airway_threshold = config.airway_threshold;
    if (img_spacing[0] < 0.5 && airway_threshold < 220) {
        airway_threshold = 220;
    }

    std::array<long unsigned, 6> roi_boundary = {0, img_size[0], 0, img_size[1], 0, img_size[2]};
    roi_boundary = GetRegionBoundary(region_mask, 0, img_size);

    if constexpr (Printout) {
        std::cout << "Region of interest boundary: ["
              << roi_boundary[0] << ", " << roi_boundary[1] << ", "
              << roi_boundary[2] << ", " << roi_boundary[3] << ", "
              << roi_boundary[4] << ", " << roi_boundary[5] << "]. "
              << std::endl;
    }

    /********************************
    Get high intensity regions
    *********************************/

    Image3DPtr high_intensity = Thresholding(img, config.high_intensity_threshold);
    high_intensity = Masking<false>(high_intensity, Thresholding(img, 500));
    Image3DPtr assist_high_intensity = Thresholding(img, -600);

    /********************************
    Compute airway map
    *********************************/

    itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(1);

    Image3DPtr combined = nullptr;

    if (config.ComputeAirwayMap()) {
        if constexpr (Printout) {
            std::cout << "Trachea seed point set to " << seeds[0] << std::endl;
        }

        unsigned num_scales = fmin(std::round(15.0/img_spacing[0]), config.scales[1]);
        unsigned scale_step = (unsigned)std::round(1.0/img_spacing[0]);

        if constexpr (Printout) {
            std::cout << "Spacing: " << img_spacing
                << ", scale num: " << num_scales
                << ", scale step: " << scale_step
                << std::endl;
        }

        Image3DPtr windowed_img = Saturation(img, -1200, 200);
        combined = AirwayReconstruction<false, Printout>(windowed_img, config.scales[0], roi_boundary, num_scales, scale_step, config);
        combined = Masking<false>(combined, assist_high_intensity);
        SaveImage<Printout>(combined, "ScalesCombined", config);
    }
    else if (config.ComputeBronchialTree()) {
        combined = io::ReadNII<Printout>(config.intermediate_dir, "ScalesCombined");
    }

    /********************************
    Compute the rest in parallel
    *********************************/

    itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(NumThreads);

    if constexpr (Printout) {
        std::cout << "Computing distance transformation..." << std::endl;
    }

    Image3DPtr dist_map = MaurerDistanceTransform(region_mask);
    dist_map = Saturation(dist_map, 0, 1000);

    Image3DPtr vessel_iowa = nullptr;
    Image3DPtr vessel_iowa_scales = nullptr;
    Image3DPtr vessel_frangi = nullptr;
    Image3DPtr vessel_frangi_scales = nullptr;
    Image3DPtr major_vessels = nullptr;
    Image3DPtr airway_lumen = nullptr;
    Image3DPtr airway_refined = nullptr;
    Image3DPtr bronchial_tree = nullptr;
    Image3DPtr large_bronchial_tubes = nullptr;

    std::mutex io_mutex;

    constexpr unsigned num_tasks = 3;
    #pragma omp parallel for num_threads(num_tasks)
    for (unsigned task_id = 0; task_id < num_tasks; ++task_id) {
        if (task_id == 0 && config.ComputeMajorVessels()) {
            if constexpr (Printout) {
                std::cout << "Computing major blood vessels...\n" << std::flush;
            }

            std::tie(vessel_iowa, vessel_iowa_scales) = ComputeVesselness<0, false>(img, region_mask, dist_map);
        }
        else if (task_id == 1 && config.ComputeVesselnessMap()) {
            if constexpr (Printout) {
                std::cout << "Computing vesselness map...\n" << std::flush;
            }

            std::tie(vessel_frangi, vessel_frangi_scales) = ComputeVesselness<1, false>(img, region_mask, dist_map);
            Image3DPtr vesselness_map = Saturation(vessel_frangi, 0, config.major_vessel_threshold);
            vesselness_map = LinearRescale(vesselness_map, 0, 1);
            {
                std::lock_guard<std::mutex> lock(io_mutex);
                SaveImage<Printout>(vesselness_map, "VesselnessMap", config);

                if (config.save_as_text) {
                    io::WriteCostToFile(vessel_frangi, config.final_dir + "costs.txt");
                }
            }
        }
        else if (task_id == 2 && config.ComputeBronchialTree()) {
            if constexpr (Printout) {
                std::cout << "Computing bronchial tree from airway...\n" << std::flush;
            }

            for (auto const& seed : seeds) {
                large_bronchial_tubes = Maximum(large_bronchial_tubes, ComputeLargeBronchialTube<Printout>(org_img, seed, config));
            }

            {
                std::lock_guard<std::mutex> lock(io_mutex);
                SaveImage<Printout>(large_bronchial_tubes, "LargeBronchialTubes", config);
            }

            large_bronchial_tubes = Masking<false>(large_bronchial_tubes, assist_high_intensity);
            combined = Masking<false>(combined, large_bronchial_tubes, 999);

            for (auto const& seed : seeds) {
                Image3DPtr tmp = ConnectedRegionGrowing(combined, config.airway_threshold, 1000, seed);
                airway_lumen = Maximum(airway_lumen, tmp);
                
                tmp = RefineAirway<Printout>(img, tmp, seed, 1, 300, 2000, 0);
                airway_refined = Maximum(airway_refined, tmp);
            }
            
            airway_refined = Masking(airway_refined, region_mask);
            airway_refined = Masking<false>(airway_refined, assist_high_intensity);

            {
                std::lock_guard<std::mutex> lock(io_mutex);
                SaveImage<Printout>(airway_lumen, "AirwayLumen", config);
                SaveImage<Printout>(airway_refined, "AirwayLumenRefined", config);
            }

            if (config.ComputeAirwaySkeleton()) {
                Image3DPtr airway_skeleton = BinaryThinning(airway_refined);
                SaveImage<Printout>(airway_skeleton, "Skeleton", config);
            }

            if (config.in_vivo) {
                bronchial_tree = WallReconstruction<Printout>(img, region_mask, large_bronchial_tubes, config, 2);
            }
            else {
                bronchial_tree = WallReconstruction<Printout>(img, region_mask, airway_refined, config, 15);
            }

            bronchial_tree = Maximum(bronchial_tree, airway_refined);

            if constexpr (Printout) {
                std::cout << "Waiting for vessel segmentation to complete...\n" << std::flush;
            }
        }
        else if (task_id == 2 && !config.ComputeBronchialTree()) {
            if (config.ComputeAirwaySkeleton()) {
                airway_refined = io::ReadNII<Printout>(config.intermediate_dir, "AirwayLumenRefined");
                Image3DPtr airway_skeleton = BinaryThinning(airway_refined);
                SaveImage<Printout>(airway_skeleton, "Skeleton", config);
            }

            bronchial_tree = io::ReadNII<Printout>(config.intermediate_dir, "BronchialTree");
        }
    }

    if constexpr (Printout) {
        std::cout << "Finishing up some final adjustment...\n" << std::flush;
    }

    if (config.ComputeMajorVessels()) {
        Image3DPtr vessel_mask = Thresholding(vessel_iowa_scales, config.major_vessel_radius);
        major_vessels = Masking(Thresholding(vessel_iowa, config.major_vessel_threshold), vessel_mask);

        if (vessel_frangi) {
            vessel_mask = Thresholding(vessel_frangi_scales, config.major_vessel_radius);
            major_vessels = Maximum(major_vessels, Masking(Thresholding(vessel_frangi, config.major_vessel_threshold), vessel_mask));
        }

        major_vessels = Masking<false>(major_vessels, Thresholding(img, 500));
        major_vessels = Masking<false>(major_vessels, bronchial_tree);
        major_vessels = IslandRemoving(major_vessels, 1000);

        SaveImage<Printout>(major_vessels, "MajorVessels", config);
    }
    else if (config.ComputeAllObstacles()) {
        major_vessels = io::ReadNII<Printout>(config.intermediate_dir, "MajorVessels");
    }

    if (config.ComputeBronchialTree()) {
        Image3DPtr bronchial_mask = Dilation<Image3DType, 3>(large_bronchial_tubes, 3);
        Image3DPtr nearby_high_intensity = Masking(high_intensity, bronchial_mask);
        nearby_high_intensity = Masking<false>(nearby_high_intensity, major_vessels);

        bronchial_tree = Maximum(bronchial_tree, nearby_high_intensity);

        SaveImage<Printout>(bronchial_tree, "BronchialTree", config);
    }

    if (config.ComputeAllObstacles()) {
        Image3DPtr obstacles = Maximum(major_vessels, bronchial_tree);

        Image3DPtr bronchial_tree_dilated = Dilation(bronchial_tree, FlatStructure3D(3, 0));
        high_intensity = Masking<false>(high_intensity, bronchial_tree_dilated);
        SaveImage<Printout>(high_intensity, "HighIntensityRegion", config);

        if (config.enable_high_intensities) {
            obstacles = Maximum(obstacles, high_intensity);
        }

        obstacles = Masking(obstacles, region_mask, 255);

        if (config.save_as_text) {
            Image3DPtr mask = Dilation(region_mask, FlatStructure3D(2, 0));
            obstacles = Masking(obstacles, mask);

            io::WriteBinaryToFile(obstacles, config.final_dir + "obstacles.txt");
        }

        SaveImage<Printout>(obstacles, "AllObstacles", config);
    }

    if (config.provide_empty_manual_seg) {
        SaveImage<Printout>(NewImage(img), "ManualSegmentation", config);
    }

    if constexpr (Printout) {
        std::cout << "Segmentation finished after "
                  << std::chrono::duration_cast<std::chrono::duration<float>>(Clock::now() - start_time).count()
                  << " seconds in total!\n"
                  << std::flush;
    }
}

std::vector<IndexType> GetTracheaSeeds(const Image3DPtr& input_img, const SizeType& img_size) {
    bool input_new_seed = true;
    std::string answer;
    std::vector<IndexType> seeds;

    while (input_new_seed) {
        IndexType seed;
        SetGreenPrint();
        std::cout << "Input trachea seed point (RAS image coordinates, use space as delimiter, intensity of the seed point need to be < -980 [HU]): "
                  << std::endl;
        ResetPrint();
        std::cin >> seed[0] >> seed[1] >> seed[2];

        if (!WithinImage(seed, img_size)) {
            SetRedPrint();
            std::cout << "Invalid seed point! Please check!" << std::endl;
            std::cout << "Valid range: [0, " << img_size[0] - 1
                      << "], [0, " << img_size[1] - 1
                      << "], [0, " << img_size[2] - 1
                      << "]" << std::endl;
            ResetPrint();
        }
        else if (input_img->GetPixel(seed) > -980){
            SetRedPrint();
            std::cout << "Invalid seed point! "
                      << "Please make sure intensity of the seed point < -980 [HU]!"
                      << std::endl;
            ResetPrint();
        }
        else {
            seeds.push_back(seed);
            SetGreenPrint();
            std::cout << "Additional trachea seed point? [Y/n] ";
            ResetPrint();
            std::cin >> answer;

            if (answer != "Y") {
                input_new_seed = false;
            }
        }
    }

    return seeds;
}

template<bool Printout>
Image3DPtr ComputeRegionMask(const Image3DPtr& img, const SegmentConfig& config, const std::vector<IndexType>& seeds) {
    Image3DPtr mask = Image3DType::New();

    int closing_radius = config.in_vivo? 10 : 7;
    SizeType img_size = img->GetLargestPossibleRegion().GetSize();

    IndexType foreground_seed = seeds[0];
    IndexType background_seed;
    background_seed[0] = img_size[0] - 1;
    background_seed[1] = img_size[1] - 1;
    background_seed[2] = img_size[2] - 1;

    if (config.in_vivo) {
        if constexpr (Printout) {
            std::cout << "This is for in vivo!" << std::endl;
        }
        
        PixelType min = -1200;
        PixelType max = config.region_mask_threshold;

        Image3DPtr tmp0 = ConnectedRegionGrowing(img, min, max, foreground_seed, true, 20, 0.125);
        Image3DPtr tmp1 = ConnectedRegionGrowing(img, min, max, foreground_seed);
        mask = Maximum(tmp0, tmp1);
        mask = MorphologicalOpening(mask, 1);

        while (mask->GetPixel(background_seed) > 0) {
            SaveImage<Printout>(mask, "PrecomputedRegionMask", config);
            SetRedPrint();
            std::cout << "Background seed point is inside the precomputed mask, please update!\n"
                      << "New background seed point: " << std::endl;
            ResetPrint();
            std::cin >> background_seed[0] >> background_seed[1] >> background_seed[2];
        }

        mask = ConnectedRegionGrowing(mask, -1, 1, background_seed);
        mask = ConnectedRegionGrowing(mask, -1, 1, foreground_seed);
        mask = MorphologicalClosing(mask, closing_radius);

        Image3DPtr mask_boundary = Erosion(mask, FlatStructure3D(5, 0));
        mask_boundary = Subtraction(mask, mask_boundary);
        mask_boundary = Masking(Thresholding(img, config.high_intensity_threshold), mask_boundary);

        mask = Masking<false>(mask, mask_boundary);
        mask = MorphologicalClosing(mask, 2);
    }
    else {
        if constexpr (Printout) {
            std::cout << "This is for ex vivo!" << std::endl;
        }

        mask = Thresholding(img, -950);
        mask = IslandRemoving(mask, 1000);
        mask = MorphologicalClosing(mask, 1);
        Image3DPtr tmp = ConnectedRegionGrowing(mask, -1, 1, background_seed);

        unsigned iter = 0;
        while (RegionVolume(tmp) < 5000) {
            auto seed = background_seed;
            iter++;
            switch (iter) {
                case 1:
                    seed[0] = 0;
                    break;
                case 2:
                    seed[1] = 0;
                    break;
                case 3:
                    seed[2] = 0;
                    break;
                case 4:
                    seed[0] = 0;
                    seed[1] = 0;
                    break;
                case 5:
                    seed[0] = 0;
                    seed[2] = 0;
                    break;
                case 6:
                    seed[1] = 0;
                    seed[2] = 0;
                    break;
                case 7:
                    seed[0] = 0;
                    seed[1] = 0;
                    seed[2] = 0;
                    break;
                default:
                   SetRedPrint();
                   std::cout << "Failed to detect a valid background seed automatically, please input seed manually:" << std::endl;
                   ResetPrint();
                   std::cin >> seed[0] >> seed[1] >> seed[2];
                   break;
            }

            if constexpr (Printout) {
                std::cout << "Switching background seed point to ["
                          << seed[0] << ", "
                          << seed[1] << ", "
                          << seed[2] << "]."
                          << std::endl;
            }
            
            tmp = ConnectedRegionGrowing(mask, -1, 1, seed);
        }

        mask = MorphologicalClosing(Invert(tmp), closing_radius);
    }

    if constexpr (Printout) {
        std::cout << "Region mask generated." << std::endl;
    }

    return mask;
}

template<bool SaveIntermediate, bool Printout>
Image3DPtr AirwayReconstruction(const Image3DPtr& img, const unsigned& min_scale, const std::array<long unsigned, 6>& roi,
    const unsigned& num_scales, const unsigned& scale_step, const SegmentConfig& config)
{
    const TimePoint start_time = Clock::now();
    Image3DPtr result = nullptr;

    const int max_num_threads = omp_get_max_threads();
    const std::vector<int> num_threads_assigned = {1, max_num_threads};
    const unsigned start_slice = roi[4];
    const unsigned end_slice = roi[5];
    const unsigned num_slices = end_slice - start_slice + 1;

    Image3DPtr scale_img = nullptr;
    std::vector<Image3DPtr> combined_imgs(num_threads_assigned[0], nullptr);
    std::vector<Image2DPtr> sliceIs(num_slices, nullptr);
    std::vector<Image2DPtr> sliceJs(num_slices, nullptr);

    std::mutex mutex;

    #pragma omp parallel for num_threads(num_threads_assigned[0])
    for (unsigned i = min_scale; i < min_scale + num_scales; i += scale_step) {
        if constexpr (Printout) {
            std::cout << "Scale: " << i << std::endl;
        }

        #pragma omp parallel for num_threads(num_threads_assigned[1])
        for (unsigned j = start_slice; j <= end_slice; ++j) {
            unsigned index = j - start_slice;
            auto& sliceI = sliceIs[index];
            auto& sliceJ = sliceJs[index];
            {
                std::lock_guard<std::mutex> lock(mutex);
                sliceI = Extract(img, 2, j);
            }
            sliceI = MeanFiltering<Image2DType, 2>(sliceI, 1);
            sliceJ = GrayscaleClosing<Image2DType, 2>(sliceI, i);
            sliceJ = SliceIteration<Printout>(sliceJ, sliceI, 10000);
            sliceJ = Subtraction<Image2DType>(sliceJ, sliceI);
        }

        scale_img = NewImage(img);

        for (unsigned j = start_slice; j <= end_slice; ++j) {
            scale_img = MaximumInsert(sliceJs[j - start_slice], scale_img, j);
        }

        auto [min, max] = ComputeMinMaxIntensity(scale_img);

        if constexpr (Printout) {
            std::cout << "min: " << min << ", max: " << max << std::endl;
        }

        if constexpr (SaveIntermediate) {
            SaveImage<Printout>(scale_img, "Scale_" + std::to_string(i), config);
        }

        scale_img = LinearRescale<true>(scale_img, 0, 1000);
        auto& combined_img = combined_imgs[omp_get_thread_num()];
        combined_img = Maximum(combined_img, scale_img);
    }

    for (const auto& combined_img : combined_imgs) {
        result = Maximum(result, combined_img);
    }

    result = LinearRescale<true>(result, 0, 1000);

    if constexpr (Printout) {
        std::cout << "Total time is " << std::chrono::duration_cast<std::chrono::duration<float>>(Clock::now() - start_time).count()
                  << " seconds for " << num_scales << " scales." << std::endl;
    }

    return result;
}

template<bool Printout>
Image2DPtr SliceIteration(const Image2DPtr& sliceJ, const Image2DPtr& sliceI, const unsigned& max_iter) {
    Image2DPtr sliceJ0 = sliceJ;
    Image2DPtr sliceJ1 = NewImage<Image2DType>(sliceJ);
    Image2DPtr diff_slice = NewImage<Image2DType>(sliceJ);

    for (unsigned i = 0; i < max_iter; i++) {
        sliceJ1 = Maximum<Image2DType>(Erosion<Image2DType, 2>(sliceJ0, 1), sliceI);
        diff_slice = Subtraction<Image2DType>(sliceJ0, sliceJ1);

        if (Empty<Image2DType>(diff_slice)) {
            break;
        }

        sliceJ0 = sliceJ1;

        if constexpr (Printout) {
            if (i == max_iter - 1) {
                std::cout << "Fail to converge in " << max_iter << " iterations!" << std::endl;
            }
        }
    }

    return sliceJ1;
}

template<unsigned Mode, bool Printout>
std::pair<Image3DPtr, Image3DPtr> ComputeVesselness(const Image3DPtr& img, const Image3DPtr& mask, const Image3DPtr& dist_map) {
    Image3DPtr result = Copy(dist_map);

    double min_sigma = 1.0;
    double max_sigma = 4.0;
    unsigned sigma_step = 5;

    std::vector<PixelType> scales(sigma_step);
    double s = std::pow(max_sigma/min_sigma, 1.0/(sigma_step - 1));

    for (unsigned i = 0; i < sigma_step; i++) {
        scales[i] = min_sigma * std::pow(s, i);
    }

    std::vector<Image3DPtr> tmp_imgs(sigma_step, nullptr);
    unsigned num_threads = (sigma_step < omp_get_max_threads() ? sigma_step : omp_get_max_threads());
    std::vector<Image3DPtr> imgs(sigma_step - 1, nullptr);
    std::vector<Image3DPtr> scale_imgs(sigma_step - 1, nullptr);
    Image3DPtr helper_img = nullptr;

    if constexpr (Mode == 0) {
        if constexpr (Printout) {
            std::cout << "Constructing vessel map using Iowa's measure with "
                      << num_threads << " threads...\n"
                      << std::flush;
        }
        
        #pragma omp parallel for num_threads(num_threads)
        for (unsigned i = 0; i < sigma_step; ++i) {
            tmp_imgs[i] = VesselnessIowa<Printout>(img, scales[i]);
        }
    }
    else if constexpr (Mode == 1) {
        if constexpr (Printout) {
            std::cout << "Constructing vessel map using Frangi's measure with "
                      << num_threads << " threads...\n"
                      << std::flush;
        }

        #pragma omp parallel for num_threads(num_threads)        
        for (int i = 0; i < sigma_step; ++i) {
            tmp_imgs[i] = VesselnessFrangi<Printout>(img, scales[i]);
        }
    }
    else {
        throw std::runtime_error("[ComputeVesselness] Undefined vesselness mode!");
    }

    unsigned level = 0;
    scale_imgs[0] = NewImage(img, scales[level]);
    imgs[0] = tmp_imgs[0];

    level = 1;
    helper_img = tmp_imgs[1];
    scale_imgs[1] = UpdateScale(Subtraction(helper_img, imgs[0]), scale_imgs[0], scales[level]);
    imgs[1] = Maximum(imgs[0], helper_img);

    level = 2;
    helper_img = tmp_imgs[2];
    scale_imgs[1] = UpdateScale(Subtraction(helper_img, imgs[1]), scale_imgs[1], scales[level]);
    imgs[1] = Maximum(imgs[1], helper_img);

    level = 3;
    helper_img = tmp_imgs[3];
    scale_imgs[2] = UpdateScale(Subtraction(helper_img, imgs[1]), scale_imgs[1], scales[level]);
    imgs[2] = Maximum(imgs[1], helper_img);

    level = 4;
    helper_img = tmp_imgs[4];
    scale_imgs[3] = UpdateScale(Subtraction(helper_img, imgs[2]), scale_imgs[2], scales[level]);
    imgs[3] = Maximum(imgs[2], helper_img);

    using IteratorType = itk::ImageRegionIterator<Image3DType>;

    RegionType region = result->GetLargestPossibleRegion();
    IteratorType res_iter(result, region);
    res_iter.GoToBegin();
    std::vector<IteratorType> img_iters;
    std::vector<IteratorType> scale_iters;
    for (unsigned i = 0; i < sigma_step - 1; ++i) {
        img_iters.emplace_back(imgs[i], region);
        img_iters.back().GoToBegin();
        scale_iters.emplace_back(scale_imgs[i], region);
        scale_iters.back().GoToBegin();
    }

    while (!res_iter.IsAtEnd()) {
        PixelType r = res_iter.Get();
        PixelType intensty, scale;

        if (r <= 1.0) {
            intensty = img_iters[0].Get();
            scale = scale_iters[0].Get();
        }
        else if (r <= 2.0) {
            intensty = img_iters[1].Get();
            scale = scale_iters[1].Get();
        }
        else if (r <= sqrt(8)) {
            intensty = img_iters[2].Get();
            scale = scale_iters[2].Get();
        }
        else {
            intensty = img_iters[3].Get();
            scale = scale_iters[3].Get();
        }

        res_iter.Set(intensty);
        scale_iters[0].Set(scale);

        ++res_iter;
        for (unsigned i = 0; i < sigma_step - 1; ++i) {
            ++img_iters[i];
            ++scale_iters[i];
        }
    }

    try {
        result->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << "Exception thrown while updating the result image" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    try {
        scale_imgs[0]->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << "Exception thrown while updating the scale image" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    if constexpr (Printout) {
        std::cout << "Done computing vesselness." << std::endl;
    }

    return {Masking(result, mask), scale_imgs[0]};
}

template<bool Printout>
Image3DPtr VesselnessIowa(const Image3DPtr& img, const double& sigma, const bool& output_binary,
    const PixelType& upper_bound, const PixelType& clearance, const PixelType& threshold)
{
    if constexpr (Printout) {
        std::cout << "Computing Iowa vesselness with "
                  << "sigma " << sigma << "; "
                  << "upper bound " << upper_bound << "; "
                  << "clearance " << clearance << "; ";

        if (output_binary) {
            std::cout << "binary threshold: " << threshold << ";";
        }

        std::cout << std::endl;
    }

    Image3DPtr result = NewImage(img);

    using HessianFilterType = itk::HessianRecursiveGaussianImageFilter<Image3DType>;

    auto filter = HessianFilterType::New();
    filter->SetInput(img);
    filter->SetSigma(sigma);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while computing hessian of the image" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    using HessianPixelType = itk::SymmetricSecondRankTensor<double, 3>;
    using HessianImageType = itk::Image<HessianPixelType, 3>;

    HessianImageType::Pointer hessian_img = filter->GetOutput();
    HessianImageType::RegionType region = hessian_img->GetLargestPossibleRegion();

    using HessianIteratorType = itk::ImageRegionIteratorWithIndex<HessianImageType>;
    using IteratorType = itk::ImageRegionIterator<Image3DType>;

    HessianIteratorType hessian_iter(hessian_img, region);
    IteratorType obj_iter(img, region);
    IteratorType res_iter(result, region);
    hessian_iter.GoToBegin();
    obj_iter.GoToBegin();
    res_iter.GoToBegin();
    HessianPixelType tensor;

    using VectorType = itk::Vector<PixelType, 3>;
    using MatrixType = itk::Matrix<PixelType, 3, 3>;

    MatrixType eigen_mat, matrix;
    VectorType eigen_val;

    using EigenVectorAnalysisType = itk::SymmetricEigenAnalysis<MatrixType, VectorType, MatrixType>;

    EigenVectorAnalysisType eigen_analysis;
    eigen_analysis.SetDimension(3);

    while (!hessian_iter.IsAtEnd()) {
        eigen_analysis.SetOrderEigenValues(true);
        tensor = hessian_iter.Get();
        matrix[0][0] = tensor[0];
        matrix[0][1] = tensor[1];
        matrix[0][2] = tensor[2];
        matrix[1][0] = tensor[1];
        matrix[1][1] = tensor[3];
        matrix[1][2] = tensor[4];
        matrix[2][0] = tensor[2];
        matrix[2][1] = tensor[4];
        matrix[2][2] = tensor[5];
        eigen_analysis.ComputeEigenValuesAndVectors(matrix, eigen_val, eigen_mat);
        
        PixelType intensity = obj_iter.Get();

        if (eigen_val[1] < 0 && intensity <= upper_bound - clearance) {
            intensity = std::fabs(eigen_val[1])/(upper_bound - intensity);
        }
        else if (eigen_val[1] < 0) {
            intensity = std::fabs(eigen_val[1])/clearance;
        }
        else {
            intensity = 0;
        }

        intensity = std::pow(sigma, 2) * intensity;
        res_iter.Set(intensity);

        ++hessian_iter;
        ++obj_iter;
        ++res_iter;
    }

    try {
        result->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << "Exception thrown while updating the result image" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    if (output_binary) {
        result = Thresholding(result, threshold);
    }
    else {
        result = ScalarMultiplication(result, 1000.0);
    }

    return result;
}

template<bool Printout>
Image3DPtr VesselnessFrangi(const Image3DPtr& img, const double& sigma, const bool& output_binary,
    const PixelType& alpha, const PixelType& beta, const PixelType& gamma, 
    const PixelType& lower_bound, const PixelType& threshold)
{
    if constexpr (Printout) {
        std::cout << "Computing Frangi vesselness with "
                  << "sigma " << sigma << "; "
                  << "alpha " << alpha << "; "
                  << "beta " << beta << "; "
                  << "gamma " << gamma << "; "
                  << "lower bound " << lower_bound << "; ";

        if (output_binary) {
            std::cout << "binary threshold: " << threshold << ";";
        }

        std::cout << std::endl;
    }

    Image3DPtr result = NewImage(img);

    using HessianFilterType = itk::HessianRecursiveGaussianImageFilter<Image3DType>;

    auto filter = HessianFilterType::New();
    filter->SetInput(img);
    filter->SetSigma(sigma);

    try {
        filter->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << __func__ << std::endl;
        std::cerr << "Exception thrown while computing hessian of the image" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    using HessianPixelType = itk::SymmetricSecondRankTensor<double, 3>;
    using HessianImageType = itk::Image<HessianPixelType, 3>;

    HessianImageType::Pointer hessian_img = filter->GetOutput();
    HessianImageType::RegionType region = hessian_img->GetLargestPossibleRegion();

    using HessianIteratorType = itk::ImageRegionIteratorWithIndex<HessianImageType>;
    using IteratorType = itk::ImageRegionIterator<Image3DType>;

    HessianIteratorType hessian_iter(hessian_img, region);
    IteratorType obj_iter(img, region);
    IteratorType res_iter(result, region);
    hessian_iter.GoToBegin();
    obj_iter.GoToBegin();
    res_iter.GoToBegin();
    HessianPixelType tensor;

    using VectorType = itk::Vector<PixelType, 3>;
    using MatrixType = itk::Matrix<PixelType, 3, 3>;

    MatrixType eigen_mat, matrix;
    VectorType eigen_val;
    PixelType response, ra, rb, s;

    using EigenVectorAnalysisType = itk::SymmetricEigenAnalysis<MatrixType, VectorType, MatrixType>;

    EigenVectorAnalysisType eigen_analysis;
    eigen_analysis.SetDimension(3);

    while (!hessian_iter.IsAtEnd()) {
        eigen_analysis.SetOrderEigenMagnitudes(true);
        tensor = hessian_iter.Get();
        matrix[0][0] = tensor[0];
        matrix[0][1] = tensor[1];
        matrix[0][2] = tensor[2];
        matrix[1][0] = tensor[1];
        matrix[1][1] = tensor[3];
        matrix[1][2] = tensor[4];
        matrix[2][0] = tensor[2];
        matrix[2][1] = tensor[4];
        matrix[2][2] = tensor[5];
        eigen_analysis.ComputeEigenValuesAndVectors(matrix, eigen_val, eigen_mat);

        if (obj_iter.Get() < lower_bound || eigen_val[1] >= 0 || eigen_val[2] >= 0) {
            response = 0;
        }
        else {
            rb = std::fabs(eigen_val[0])/std::sqrt(std::fabs(eigen_val[1] * eigen_val[2]));
            ra = std::fabs(eigen_val[1])/std::fabs(eigen_val[2]);
            s = std::sqrt(std::pow(eigen_val[1], 2) + std::pow(eigen_val[2], 2));
            response = (1-std::exp(-std::pow(ra,2)/2/std::pow(alpha,2))) *
                       std::exp(-std::pow(rb,2)/2/std::pow(beta,2)) *
                       (1-std::exp(-std::pow(s,2)/2/std::pow(gamma,2)));
        }

        PixelType intensity = std::pow(sigma, 2) * response;

        res_iter.Set(intensity);

        ++hessian_iter;
        ++obj_iter;
        ++res_iter;
    }

    try {
        result->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << "Exception thrown while updating result image" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    if (output_binary) {
        result = Thresholding(result, threshold);
    }
    else {
        result = ScalarMultiplication(result, 1000.0);
    }

    return result;
}

Image3DPtr UpdateScale(const Image3DPtr& diff_img, const Image3DPtr& old_scale_map, const PixelType& new_scale) {
    Image3DPtr new_scale_map = NewImage(old_scale_map);
    RegionType region = old_scale_map->GetLargestPossibleRegion();

    using IteratorType = itk::ImageRegionIterator<Image3DType>;

    IteratorType res_iter(new_scale_map, region);
    IteratorType diff_iter(diff_img, region);
    IteratorType old_map_iter(old_scale_map, region);
    res_iter.GoToBegin();
    diff_iter.GoToBegin();
    old_map_iter.GoToBegin();

    while (!res_iter.IsAtEnd()) {
        if (diff_iter.Get() > 0) {
            res_iter.Set(new_scale);
        }
        else {
            res_iter.Set(old_map_iter.Get());
        }

        ++res_iter;
        ++diff_iter;
        ++old_map_iter;
    }

    try {
        new_scale_map->Update();
    }
    catch (itk::ExceptionObject& excp) {
        std::cerr << "Exception thrown while updating scale image" << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return new_scale_map;
}

template<bool Printout>
Image3DPtr ComputeLargeBronchialTube(const Image3DPtr& img, const IndexType& seed, const SegmentConfig& config) {
    Image3DPtr region = nullptr;
    PixelType optimal_threshold;

    if (config.bronch_threshold < -1000) {
        optimal_threshold = AdaptiveRegionGrowing(img, seed, -980, 100000, 0.5);

        if (config.check_bronch_threshold) {
            SetGreenPrint();
            std::cout << "Is threshold " << optimal_threshold
                      << " for seed [" << seed[0] << "," << seed[1] << "," << seed[2]
                      << "] OK? [Y/n]" << std::endl;
            ResetPrint();

            std::string answer;
            std::cin >> answer;

            if (answer != "Y") {
                SetGreenPrint();
                std::cout << "Please suggest threshold manually: " << std::endl;
                ResetPrint();
                std::cin >> optimal_threshold;

                while (optimal_threshold > -600) {
                    SetRedPrint();
                    std::cout << "Threshold " << optimal_threshold << " is too high (> -600 [HU])! Input new value: " << std::endl;
                    ResetPrint();
                    std::cin >> optimal_threshold;
                } 
            }
        }
        else {
            if constexpr (Printout) {
                std::cout << "Adaptive region growing result threshold: " << optimal_threshold << std::endl;
            }
        }
    }
    else {
        optimal_threshold = config.bronch_threshold;
    }

    if constexpr (Printout) {
        std::cout << "Using threshold " << optimal_threshold
                  << " for large bronchial tube segmentation." << std::endl;
    }
    
    region = ConnectedRegionGrowing(img, -1200, optimal_threshold, seed);

    return MorphologicalClosing(region, 8);
}

template<bool Printout>
Image3DPtr RefineAirway(const Image3DPtr& img, const Image3DPtr& region, const IndexType& seed,
    const unsigned& radius, const PixelType& min, const PixelType& max, const PixelType& replace_val)
{
    if constexpr (Printout) {
        std::cout << "Refining airway segmentation..." << std::endl;
    }
    
    Image3DPtr tmp = Masking<false>(img, region, replace_val);
    tmp = GrayscaleClosing(tmp, radius);
    tmp = Subtraction(tmp, img);

    tmp = ConnectedRegionGrowing(tmp, min, max, seed);
    tmp = Maximum(region, tmp);

    return tmp;
}

template<bool Printout>
Image3DPtr WallReconstruction(const Image3DPtr& img, const Image3DPtr& mask, const Image3DPtr& airway,
    const SegmentConfig& config, const unsigned& max_iter)
{
    if constexpr (Printout) {
        std::cout << "Reconstructing bronchial tube walls..." << std::endl;
    }

    PixelType absolute_threshold = -800;
    PixelType diff_threshold = 0;
    Image3DPtr eroded_mask = Erosion(mask, FlatStructure3D(3, 0));

    Flat3DStructuringElementType se = FlatStructure3D(1, 2);

    Image3DPtr bronchial_tree = Dilation(airway, se);
    Image3DPtr wall = Subtraction(bronchial_tree, airway);
    Image3DPtr absolute_val = Masking(img, wall, -1200);

    Image3DPtr airway1 = Masking(img, airway, -1200);
    Image3DPtr bronchial_tree1 = Dilation(airway1, se);
    Image3DPtr diff_val = Subtraction(img, bronchial_tree1);
    diff_val = Masking(diff_val, wall, -1200);

    absolute_val = Thresholding(absolute_val, absolute_threshold);
    diff_val = Thresholding(diff_val, diff_threshold);
    Image3DPtr sys_img = Minimum(absolute_val, diff_val);
    Image3DPtr result = Maximum(sys_img, airway);
    result = GrayscaleClosing(result, 1);

    for (unsigned i = 0; i < max_iter; ++i) {
        absolute_threshold = config.in_vivo ? -200 : -600;
        diff_threshold = 10;

        bronchial_tree = Dilation(result, se);
        wall = Subtraction(bronchial_tree, result);
        absolute_val = Masking(img, wall, -1200);

        airway1 = Masking(img, result, -1200);
        bronchial_tree1 = Dilation(airway1, se);
        diff_val = Subtraction(img, bronchial_tree1);
        diff_val = Masking(diff_val, wall, -1200);

        absolute_val = Thresholding(absolute_val, absolute_threshold);
        diff_val = Thresholding(diff_val, 0 - diff_threshold);
        sys_img = Minimum(absolute_val, diff_val);
        result = Maximum(sys_img, result);
        result = GrayscaleClosing(result, 1);

        if (Empty(sys_img)) {
            break;
        }
    }

    return result;
}

template<bool Printout>
void SaveImage(const Image3DPtr& img, const std::string& img_name, const SegmentConfig& config) {
    if (config.save_intermediate_results) {
        io::WriteNII<Printout>(img, config.intermediate_dir + img_name);
    }

    if (config.final_results.find(img_name) != config.final_results.end()) {
        io::WriteNII<Printout>(img, config.final_dir + img_name);
    }
}

} // namespace unc::robotics::lungseg
