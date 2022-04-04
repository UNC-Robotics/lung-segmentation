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

#include "../file_arrange.h"

namespace unc::robotics::lungseg::image::io {

void PrintImageInfo(const Image3DType::Pointer& image) {
    std::cout << "Spacing: "<< image->GetSpacing() << std::endl;
    std::cout << "Origin: " << image->GetOrigin() << std::endl;
    std::cout << "Image size: " << image->GetLargestPossibleRegion().GetSize() << std::endl;
    std::cout << "Direction: " << image->GetDirection() << std::endl;
}

Eigen::Matrix4d GetIJKtoRASMatrix(const Image3DType::Pointer& image) {
    Image3DType::SpacingType spacing = image->GetSpacing();
    Image3DType::PointType origin = image->GetOrigin();
    Image3DType::DirectionType direction = image->GetDirection();

    Eigen::Matrix3d dir;
    dir << direction[0][0], direction[0][1], direction[0][2],
           direction[1][0], direction[1][1], direction[1][2],
           direction[2][0], direction[2][1], direction[2][2];

    Eigen::Matrix3d kLPStoRAS;
    kLPStoRAS << -1,0,0,0,-1,0,0,0,1;

    dir = kLPStoRAS * dir;

    origin[0] *= -1;
    origin[1] *= -1;

    Eigen::Matrix4d mat;
    mat.setIdentity();
    int row, col;
    for (row = 0; row < 3; row++) {
        for (col = 0; col < 3; col++) {
            mat(row, col) = spacing[col] * dir(row, col);
        }
        mat(row, 3) = origin[row];
    }

    return mat;
}

template<bool Printout>
Image3DType::Pointer ReadNII(const std::string& filename) {
    auto const& s = filename.size();
    if (filename.compare(s - 4, 4, ".nii") != 0) {
        throw std::runtime_error("[ReadNII] Cannot read a non-nii file!");
    }

    FileReaderType::Pointer reader = FileReaderType::New();
    reader->SetFileName(filename);

    if constexpr (Printout) {
        std::cout << "Reading " << filename << std::endl;
    }

    try{
        reader->Update();
    }
    catch(itk::ExceptionObject &excp) {
        std::cerr << "Exception thrown while reading " << filename << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return reader->GetOutput();
}

template<bool Printout>
Image3DType::Pointer ReadNII(const std::string& dir, const std::string& filename) {
    std::string full_dir = dir + filename + ".nii";
    return ReadNII(full_dir);
}

template<bool Printout>
void WriteNII(const Image3DType::Pointer& image, std::string filename) {
    FileWriterType::Pointer writer = FileWriterType::New();
    auto const& s = filename.size();
    if (filename.compare(s - 4, 4, ".nii") != 0) {
        filename += ".nii";
    }
    writer->SetFileName(filename);
    writer->SetInput(image);

    if constexpr (Printout) {
        std::cout << "Writing " << filename << std::endl;
    }

    try{
        writer->Update();
    }
    catch(itk::ExceptionObject &excp) {
        std::cerr << "Exception thrown while writing " << filename << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }
}

template<bool Printout>
Image3DType::Pointer ReadDICOM(const std::string& dir) {
    auto namesGenerator = NamesGeneratorType::New();
    namesGenerator->SetInputDirectory(dir);
    const SeriesReaderType::FileNamesContainer& filenames = namesGenerator->GetInputFileNames();
    std::size_t numberOfFileNames = filenames.size();
    auto dicomIO = ImageIOType::New();

    SeriesReaderType::Pointer reader = SeriesReaderType::New();
    reader->SetFileNames(filenames);
    reader->SetImageIO(dicomIO);

    if constexpr (Printout) {
        std::cout << "Reading " << numberOfFileNames << " DICOM files from " << dir << std::endl;
    }

    try {
        reader->Update();
    }
    catch(itk::ExceptionObject& excp) {
        std::cerr << "Exception thrown while reading " << dir << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return reader->GetOutput();
}

template<bool Printout>
void WriteDICOM(const Image3DType::Pointer& image, const std::string& dir) {
    itksys::SystemTools::MakeDirectory(dir);
    auto dicomIO = ImageIOType::New();
    
    OutputNamesGeneratorType::Pointer outputNames = OutputNamesGeneratorType::New();
    std::string seriesFormat = dir + "/image-%05d.dcm";
    outputNames->SetSeriesFormat(seriesFormat.c_str());
    outputNames->SetStartIndex(1);
    outputNames->SetEndIndex(image->GetLargestPossibleRegion().GetSize()[2]);

    SeriesWriterType::Pointer writer = SeriesWriterType::New();
    writer->SetInput(image);
    writer->SetImageIO(dicomIO);
    writer->SetFileNames(outputNames->GetFileNames());

    if constexpr (Printout) {
        std::cout << "Writing " << dir << " (DICOM files)..." << std::endl;
    }

    try {
        writer->Update();
    }
    catch(itk::ExceptionObject& excp) {
        std::cerr << "Exception thrown while writing series " << dir << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }
}

template<bool Printout>
Image3DType::Pointer ReadMHD(const std::string& dir) {
    FileReaderType::Pointer reader = FileReaderType::New();
    reader->SetFileName(dir);

    if constexpr (Printout) {
        std::cout << "Reading " << dir << " (.mhd/.raw file)..." << std::endl;
    }

    try {
        reader->Update();
    }
    catch(itk::ExceptionObject& excp) {
        std::cerr << "Exception thrown while reading" << dir << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }

    return reader->GetOutput();
}

template<bool Printout>
void WriteMHD(const Image3DType::Pointer& image, const std::string& dir, const std::string& image_name) {
    std::string filename = file::FileName(dir, image_name);
    itksys::SystemTools::MakeDirectory(filename);

    FileWriterType::Pointer writer = FileWriterType::New();
    std::string mhd_name = filename + "/" + image_name + ".mhd";
    writer->SetFileName(mhd_name);
    writer->SetInput(image);

    if constexpr (Printout) {
        std::cout << "Writing " << filename << " (.mhd/.raw file)..." << std::endl;
    }

    try {
        writer->Update();
    }
    catch(itk::ExceptionObject& excp) {
        std::cerr << "Exception thrown while writing " << filename << std::endl;
        std::cerr << excp << std::endl;
        exit(1);
    }
}

template<bool Printout>
void WriteBinaryToFile(const Image3DType::Pointer& image, const std::string& filename, const bool in_ras) {
    std::ofstream fout;
    fout.open(filename);

    std::string delimiter = " ";
    auto const& s = filename.size();
    if (filename.compare(s - 4, 4, ".csv") == 0) {
        delimiter = ",";
    }

    if (!fout.is_open()) {
        std::cerr << "Cannot open " << filename << std::endl;
        exit(1);
    }

    auto const affine = GetIJKtoRASMatrix(image);

    if (!in_ras) {
        for (auto i = 0; i < 4; ++i) {
            fout << affine(i, 0) << delimiter
                 << affine(i, 1) << delimiter
                 << affine(i, 2) << delimiter
                 << affine(i, 3) 
                 << "\n";
        }
    }
    else {
        if constexpr (Printout) {
            std::cout << "IJK to RAS affine: \n" << affine << std::endl;
        }
    }

    Image3DType::IndexType idx;
    Eigen::Vector4d p;
    auto const size = image->GetLargestPossibleRegion().GetSize();
    if (!in_ras) {
        fout << size[0] << delimiter << size[1] << delimiter << size[2] << "\n";
    }

    for (idx[0] = 0; idx[0] < size[0]; ++idx[0]) {
        for (idx[1] = 0; idx[1] < size[1]; ++idx[1]) {
            for (idx[2] = 0; idx[2] < size[2]; ++idx[2]) {
                if (image->GetPixel(idx) > 0) {
                    if (!in_ras) {
                        fout << idx[0] << delimiter << idx[1] << delimiter << idx[2] << "\n";
                    }
                    else {
                        p << idx[0], idx[1], idx[2], 1;
                        p = affine*p;
                        fout << p[0] << delimiter << p[1] << delimiter << p[2] << "\n";
                    }
                }
            }
        }
    }

    fout.close();

    if constexpr (Printout) {
        std::cout << "Binary image written to " << filename << std::endl;
    }
}

template<bool Printout>
void WriteCostToFile(const Image3DType::Pointer& image, const std::string& filename) {
    std::ofstream fout;
    fout.open(filename);

    std::string delimiter = " ";
    auto const& s = filename.size();
    if (filename.compare(s - 4, 4, ".csv") == 0) {
        delimiter = ",";
    }

    if (!fout.is_open()) {
        std::cerr << "Cannot open " << filename << std::endl;
        exit(1);
    }

    Image3DType::IndexType idx;
    const auto size = image->GetLargestPossibleRegion().GetSize();

    for (idx[0] = 0; idx[0] < size[0]; ++idx[0]) {
        for (idx[1] = 0; idx[1] < size[1]; ++idx[1]) {
            for (idx[2] = 0; idx[2] < size[2]; ++idx[2]) {
                if (image->GetPixel(idx) > 0.01) {
                    fout << idx[0] << delimiter
                         << idx[1] << delimiter
                         << idx[2] << delimiter
                         << image->GetPixel(idx) << delimiter
                         << "\n";
                }
            }
        }
    }

    fout.close();

    if constexpr (Printout) {
        std::cout << "Cost image written to " << filename << std::endl;
    }
}

} // namespace unc::robotics::lungseg::image::io