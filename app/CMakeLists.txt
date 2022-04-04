add_executable(segment run_segmentation.cc)
target_link_libraries(segment ${ITK_LIBRARIES} ${Boost_LIBRARIES})
target_compile_features(segment PRIVATE cxx_std_17)
