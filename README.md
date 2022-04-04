# lung-segmentation

Code for the segmentation part of IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2018 paper *Safe Motion Planning for Steerable Needles using Cost Maps Automatically Extracted from Pulmonary Images*. [[Paper](https://ieeexplore.ieee.org/document/8593407)]

## Lung Segmentation for Steerable Needle Motion planning

Steerable needles are highly flexible medical devices able to follow 3D curvilinear trajectories inside the human body, reaching clinically significant targets while safely avoiding critical anatomical structures. Compared with traditional rigid-medical instruments, steerable needles can reduce a patient’s trauma, increase safety, and provide minimally invasive access to targets that were previously inaccessible. Steerable needles have been considered in a wide range of diagnostic and treatment procedures including biopsy, and radioactive seed implantation for cancer treatment.

For lung nodule biopsy using steerable needles, reconstructing the anatomical environment of lungs is important. Code in this reposotory automatically segments anatomical structures in a CT scan of a lung including the bronchial tree, major blood vessels, and the pulmonary pleura. Meanwhile, a cost map encoding the information of small blood vessels is also constructed as a cost metric, enabling the motion planner to minimize puncturing small blood vessels.

## Requirements

* C++17 compatible compiler (GCC 7+ for Linux, Clang 5+ for maxOS)

&nbsp;&nbsp;&nbsp;&nbsp;***For maxOS:*** *Xcode clang does not support OpenMP, system clang is required.* Use homebrew to install clang with
```
brew install llvm
```

* [CMake](https://cmake.org/) 3.8+
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) 3.3+
* [Boost](https://www.boost.org/) 1.68+
* [ITK](https://itk.org/) 5.0.1+ (when installing ITK, enable `Module_Thickness3D` which is OFF by default)

&nbsp;&nbsp;&nbsp;&nbsp;*Building ITK from source is recommended.*

&nbsp;&nbsp;&nbsp;&nbsp;***For maxOS:*** *Use the system clang to build ITK (keep things consistent to avoid mismatch).* More specifically, Run `cmake` for ITK with
```
CC=/usr/local/opt/llvm/bin/clang CXX=/usr/local/opt/llvm/bin/clang++ cmake ..
```

## Usage

### Download

```
git clone git@github.com:UNC-Robotics/lung-segmentation.git [{YOUR_LOCAL_REPO}]
cd {YOUR_LOCAL_REPO}
```

### Build

```
mkdir -p {YOUR_LOCAL_REPO}/build
cd {YOUR_LOCAL_REPO}/build
cmake ..
make
```

&nbsp;&nbsp;&nbsp;&nbsp;***For maxOS:*** *Xcode clang does not support OpenMP, system clang is required.* Run `cmake` with
```
CC=/usr/local/opt/llvm/bin/clang CXX=/usr/local/opt/llvm/bin/clang++ cmake ..
```


### Run


An example usage is `app/run_segmentation.cc`. It takes as input a chest CT image (DICOM and NII files are supported), an output directory, and a configuration file ([`segment_config.txt`](https://github.com/UNC-Robotics/lung-segmentation/blob/main/segment_config.txt)); and it outputs multiple images representing the segmentation results.
Run the example with:
```
cd {YOUR_LOCAL_REPO}/build
./app/segment input_image output_directory [segmentation_config_file]
```

*When the input image is in DICOM format, provide a directory. When the input image is in NII format, provide a file.*

*Please note, the default configuration file is for ex-vivo procine lungs. For in-vivo procine lungs or human chest CT, please adjust the parameters accordingly!*

*During the segmentation, if you are asked to provide seed points, please provide image coordinates in RAS frame!*

You will be asked to:

1. Provide seed point(s) inside the airway. At least one seed point is required. Intensity of the seed point(s) is required to be lower than -980 [HU].
2. If you choose to manually validate the threshold for large bronchial tube segmentation (adjust this in [`segment_config.txt`](https://github.com/UNC-Robotics/lung-segmentation/blob/main/segment_config.txt)), a threshold value computed with adaptive region growing will be displayed and you can either choose to accept the value or enter a different value instead.


The final results will be saved under `output_directory/final` (or `output_directory` if saving intermediate results is not enabled) and by default, the following images are saved in NII format. An output image is either a *binary image* (an image with either 0 or 255 intensities with 255 representing the segmented objects) or a image with float-number intensities.

* RegionMask: a binary image marking the lung region
* BronchialTree: a binary image marking the bronchial tree
* MajorVessels: a binary image marking the major blood vessels (that are considered as obstacles)
* AllObstacles: a binary image makring all obstacles to avoid
* VesselnessMap: a image representing the vesselness of each voxel, the higher the intensity is the more likely there exists a blood vessel

### Generating compatible results for planning

If `saving final obstacles and costs as text files` is enabled in the configuration file ([`segment_config.txt`](https://github.com/UNC-Robotics/lung-segmentation/blob/main/segment_config.txt)), two files will be generated along with the above images.
These two files can be used as input for [steerable-needle-planner](https://github.com/UNC-Robotics/steerable-needle-planner). To use the segmetation results, simply place the output text files under `steerable-needle-planner/data/input`.


## Citation

If you use this source code, please cite the following papers accordingly:
```
@inproceedings{Fu2018_IROS,
  author={Fu, Mengyu and Kuntz, Alan and Webster, Robert J. and Alterovitz, Ron},
  booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Safe Motion Planning for Steerable Needles Using Cost Maps Automatically Extracted from Pulmonary Images}, 
  year={2018},
  volume={},
  number={},
  pages={4942-4949},
  doi={10.1109/IROS.2018.8593407}
}
```
