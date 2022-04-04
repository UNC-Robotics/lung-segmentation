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
#ifndef LUNGSEG_IMAGE_IO_H
#define LUNGSEG_IMAGE_IO_H

#include <iostream>
#include <string>
#include <Eigen/Dense>

#include <itkImage.h>
#include <itkImageRegion.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMacro.h>

#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkImageSeriesReader.h>
#include <itkImageSeriesWriter.h>
#include <itkNumericSeriesFileNames.h>
#include <itksys/SystemTools.hxx>

namespace unc::robotics::lungseg::image {
	using PixelType = float;
	using Image2DType = itk::Image<PixelType, 2>;
	using Image2DPtr = Image2DType::Pointer;
	using Image3DType = itk::Image<PixelType, 3>;
	using Image3DPtr = Image3DType::Pointer;
	using FileReaderType = itk::ImageFileReader<Image3DType>;
	using FileWriterType = itk::ImageFileWriter<Image3DType>;
	using SeriesReaderType = itk::ImageSeriesReader<Image3DType>;
	using SeriesWriterType = itk::ImageSeriesWriter<Image3DType, Image2DType>;
	using ImageIOType = itk::GDCMImageIO;
	using NamesGeneratorType = itk::GDCMSeriesFileNames;
	using OutputNamesGeneratorType = itk::NumericSeriesFileNames;

	using IndexType = Image3DType::IndexType;
	using RegionType = Image3DType::RegionType;
	using SizeType = Image3DType::SizeType;
	using SpacingType = Image3DType::SpacingType;

	namespace io {
		void PrintImageInfo(const Image3DType::Pointer& image);
		Eigen::Matrix4d GetIJKtoRASMatrix(const Image3DType::Pointer& image);

		template<bool Printout=false>
		Image3DType::Pointer ReadNII(const std::string& filename);
		template<bool Printout=false>
		Image3DType::Pointer ReadNII(const std::string& dir, const std::string& filename);
		template<bool Printout=false>
		void WriteNII(const Image3DType::Pointer& image, std::string filename);

		template<bool Printout=false>
		Image3DType::Pointer ReadDICOM(const std::string& dir);
		template<bool Printout=false>
		void WriteDICOM(const Image3DType::Pointer& image, const std::string& dir);

		template<bool Printout=false>
		Image3DType::Pointer ReadMHD(const std::string& dir);
		template<bool Printout=false>
		void WriteMHD(const Image3DType::Pointer& image, const std::string& dir, const std::string& image_name);

		template<bool Printout=false>
		void WriteBinaryToFile(const Image3DType::Pointer& image, const std::string& filename, const bool in_ras=false);
		template<bool Printout=false>
		void WriteCostToFile(const Image3DType::Pointer& image, const std::string& filename);
	} // namespace io

} // namespace unc::robotics::lungseg::image

#include "impl/image_io.hpp"

#endif // LUNGSEG_IMAGE_IO_H