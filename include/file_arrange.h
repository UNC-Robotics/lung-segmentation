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
#ifndef LUNGSEG_FILE_H
#define LUNGSEG_FILE_H

#include <string>
#include <boost/filesystem.hpp>

namespace unc::robotics::lungseg::file {

inline bool Exists(const std::string& dir) {
    return boost::filesystem::exists(dir);
}

inline bool IsFile(const std::string& dir) {
    return boost::filesystem::is_regular_file(dir);
}

inline std::string UnifiedDir(const std::string& dir) {
    if (dir.back() == '/') {
        return dir;
    }

    return dir + "/";
}

std::string FileName(const std::string& dir, const std::string& file) {
    return UnifiedDir(dir) + file;
}

template<bool Printout=false>
std::string CreateDir(const std::string& dir) {
    if (Exists(dir)) {
        if constexpr (Printout) {
            std::cout << dir << " already exists" << std::endl;
        }

        return UnifiedDir(dir);
    }

    boost::filesystem::create_directory(dir);
    if constexpr (Printout) {
        std::cout << dir << " created" << std::endl;
    }

    return UnifiedDir(dir);
}

template<bool Printout=false>
std::string CreateFolder(const std::string& dir, const std::string& folder_name){
    if (!Exists(dir)) {
        throw std::runtime_error("[CreateFolder] Invalid directory provided!");
    }

    return CreateDir<Printout>(UnifiedDir(dir) + folder_name + "/");
}

} // namespace unc::robotics::lungseg::file

#endif // LUNGSEG_FILE_H