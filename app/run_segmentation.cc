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

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include "file_arrange.h"
#include "image_io.h"
#include "seg_config.h"
#include "core_algorithms.h"

using namespace unc::robotics::lungseg;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " input_image output_image_dir [config_file]"
                  << std::endl;
        exit(0);
    }

    constexpr bool show_printout = true;

    using Image3DPtr = image::Image3DType::Pointer;
    using namespace image::io;

    std::string input_dir = argv[1];
    if (!file::Exists(input_dir)) {
        throw std::runtime_error("Invalid input directory!");
    }

    Image3DPtr input_img = nullptr;
    if (!file::IsFile(input_dir)) {
        input_dir = file::UnifiedDir(input_dir);
        input_img = ReadDICOM<show_printout>(input_dir);
    }
    else {
        input_img = ReadNII<show_printout>(input_dir);
    }

    std::cout << "Input image properties:" << std::endl;
    PrintImageInfo(input_img);

    std::string output_dir = file::UnifiedDir(argv[2]);
    file::CreateDir<show_printout>(output_dir);

    std::string config_file = "../segment_config.txt";
    if (argc > 3) {
        config_file = argv[3];
        if (!file::Exists(config_file)) {
            throw std::runtime_error("Invalid segment config file!");
        }
    }

    SegmentConfig config(config_file);

    if (config.save_intermediate_results) {
        std::string intermediate_dir = file::CreateFolder<show_printout>(output_dir, "intermediate");
        std::string final_dir = file::CreateFolder<show_printout>(output_dir, "final");
        config.SetOutputDir(intermediate_dir, final_dir);
    }
    else {
        config.SetOutputDir(output_dir);
    }

    config.AddFinalResultName("RegionMask");
    config.AddFinalResultName("BronchialTree");
    config.AddFinalResultName("MajorVessels");
    config.AddFinalResultName("AllObstacles");
    config.AddFinalResultName("VesselnessMap");
    config.ShowAll();

    FullSegmentation<64, show_printout>(input_img, config);

    return 0;
}