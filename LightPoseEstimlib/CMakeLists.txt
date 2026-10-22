cmake_minimum_required(VERSION 3.18)
project(LightPoseEstimLib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LibTorch must be the PyTorch version the export was built with (see pose_estimator.json);
# point CMAKE_PREFIX_PATH at the venv's torch/share/cmake to use the same one.
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Torch REQUIRED)
find_package(OpenCV 4.5.3 REQUIRED COMPONENTS core imgproc imgcodecs calib3d)  # 4.5.3: SQPnP
find_package(nlohmann_json 3 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(LightPoseEstimLib
        pose/Position.cpp
        pose/Quaternion.cpp
        pose/Pose.cpp
        ExportConfig.cpp
        PoseEstimator.cpp
)
target_include_directories(LightPoseEstimLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/pose)
target_link_libraries(LightPoseEstimLib
        PUBLIC Eigen3::Eigen ${TORCH_LIBRARIES} ${OpenCV_LIBS}
        PRIVATE nlohmann_json::nlohmann_json)

# Runs an export on images and prints one JSON line per image.
add_executable(main main.cpp)
target_link_libraries(main PRIVATE LightPoseEstimLib nlohmann_json::nlohmann_json)
