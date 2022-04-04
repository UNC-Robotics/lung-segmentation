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
#ifndef LUNGSEG_CONFIG_H
#define LUNGSEG_CONFIG_H

#include <array>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_set>

#include "utils.h"
#include "file_arrange.h"

namespace unc::robotics::lungseg {

class SegmentConfig {
public:
    SegmentConfig(const std::string& file_name) {
        std::ifstream file;
        file.open(file_name);

        std::string line;
        std::istringstream stream;

        for (unsigned i = 0; i < procedure_control.size(); ++i) {
            std::getline(file, line);
            stream = std::istringstream(line);
            stream >> procedure_control[i];
        }

        std::getline(file, line);
        stream = std::istringstream(line);
        stream >> provide_empty_manual_seg;

        std::getline(file, line);
        stream = std::istringstream(line);
        stream >> in_vivo;

        std::getline(file, line);
        stream = std::istringstream(line);
        stream >> downsampling;

        std::getline(file, line);
        stream = std::istringstream(line);
        stream >> enable_high_intensities;

        std::getline(file, line);
        stream = std::istringstream(line);
        stream >> check_bronch_threshold;

        std::getline(file, line);
        stream = std::istringstream(line);
        stream >> save_as_text;

        std::getline(file, line);
        stream = std::istringstream(line);
        stream >> save_intermediate_results;


        std::getline(file, line);
        stream = std::istringstream(line);
        stream >> region_mask_threshold;

        std::getline(file, line);
        stream = std::istringstream(line);
        stream >> scales[0] >> scales[1];

        std::getline(file, line);
        stream = std::istringstream(line);
        stream >> airway_threshold;

        std::getline(file, line);
        stream = std::istringstream(line);
        stream >> major_vessel_threshold;

        std::getline(file, line);
        stream = std::istringstream(line);
        stream >> high_intensity_threshold;

        std::getline(file, line);
        stream = std::istringstream(line);
        stream >> bronch_threshold;
    }

    void AddFinalResultName(const std::string& img_name) {
        final_results.insert(img_name);
    }

    void SetOutputDir(const std::string& intermediate, const std::string& final) {
        intermediate_dir = file::UnifiedDir(intermediate);
        final_dir = file::UnifiedDir(final);
    }

    void SetOutputDir(const std::string& final) {
        final_dir = file::UnifiedDir(final);
    }

    void Show(const bool choice, const std::string& name) const {
        if (choice) {
            SetGreenPrint();
            std::cout << name << ": Yes" << std::endl;
        }
        else {
            SetRedPrint();
            std::cout << name << ": No" << std::endl;
        }
        ResetPrint();
    }

    void Show(const unsigned& number, const std::string& name) const {
        SetYellowPrint();
        std::cout << name << ": " << number << std::endl;
        ResetPrint();
    }

    void Show(const float& number, const std::string& name) const {
        SetYellowPrint();
        std::cout << name << ": " << number << std::endl;
        ResetPrint();
    }

    void ShowAll() const {
        std::cout << "\033[1;37m" << "Segmentation config:" << "\033[0m" << std::endl;

        Show(procedure_control[0], "Compute region mask");
        Show(procedure_control[1], "Compute airway map");
        Show(procedure_control[2], "Compute bronchial tree");
        Show(procedure_control[3], "Compute major vessels");
        Show(procedure_control[4], "Compute vesselness map");
        Show(procedure_control[5], "Compute combined obstacles");
        Show(procedure_control[6], "Compute airway skeleton");

        Show(provide_empty_manual_seg, "Provide empty manual segmentation file");
        Show(in_vivo, "If in vivo");
        Show(downsampling, "If enable downsampling");
        Show(enable_high_intensities, "If enable high-intensity regions as obstacles");
        Show(check_bronch_threshold, "If enable manual check for bronchial tube threshold");
        Show(save_as_text, "If save final obstacles and costs as text files");
        Show(save_intermediate_results, "If save intermediate result images");

        Show(region_mask_threshold, "Threshold for computing region mask");
        Show(scales[0], "Airway minimum scale");
        Show(scales[1], "Airway maximum scale");
        Show(airway_threshold, "Threshold for reconstructing airway from airway map");
        Show(major_vessel_threshold, "Threshold for reconstructing major vessels");
        Show(high_intensity_threshold, "Threshold for determining high-intensity regions");
        Show(bronch_threshold, "Threshold for reconstructing large bronchial tubes");

        SetCyanPrint();
        std::cout << "Final result images: " << std::endl;
        for (auto const& name : final_results) {
            std::cout << "\t" << name << "\n";
        }
        ResetPrint();

        std::cout << std::endl;
    }

    bool ComputeRegionMask() const {
        return procedure_control[0];
    }

    bool ComputeAirwayMap() const {
        return procedure_control[1];
    }

    bool ComputeBronchialTree() const {
        return procedure_control[2];
    }

    bool ComputeMajorVessels() const {
        return procedure_control[3];
    }

    bool ComputeVesselnessMap() const {
        return procedure_control[4];
    }

    bool ComputeAllObstacles() const {
        return procedure_control[5];
    }

    bool ComputeAirwaySkeleton() const {
        return procedure_control[6];
    }

    std::array<bool, 7> procedure_control;
    bool provide_empty_manual_seg;
    bool in_vivo;
    bool downsampling;
    bool enable_high_intensities;
    bool check_bronch_threshold;
    bool save_as_text;
    bool save_intermediate_results;

    float region_mask_threshold;
    std::array<unsigned, 2> scales;
    float airway_threshold;
    float major_vessel_threshold;
    float major_vessel_radius{1.1};
    float high_intensity_threshold;
    float bronch_threshold;

    std::string intermediate_dir{""};
    std::string final_dir{""};
    std::unordered_set<std::string> final_results;
};

} // unc::robotics::lungseg

#endif // LUNGSEG_CONFIG_H