cmake_minimum_required(VERSION 3.8)
project(rail_detector)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(ONNXRUNTIME_ROOT "$ENV{HOME}/onnxruntime")
include_directories(${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session)
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/build/Linux/Release)

add_executable(rail_detector_node src/rail_detector_node.cpp)

target_include_directories(rail_detector_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_ROOT}/include
)

add_library(row_counter_node SHARED
  src/row_counter_node.cpp
)

target_link_libraries(rail_detector_node
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_ROOT}/build/Linux/Release/libonnxruntime.so
  ${ONNXRUNTIME_ROOT}/build/Linux/Release/libonnxruntime_providers_cuda.so
)

target_include_directories(row_counter_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
ament_target_dependencies(row_counter_node
  rclcpp rclcpp_components sensor_msgs nav_msgs std_msgs
  cv_bridge OpenCV)
rclcpp_components_register_nodes(row_counter_node "RowCounterNode")

add_executable(row_counter_node_exec
  src/row_counter_node.cpp)      
target_compile_definitions(row_counter_node_exec PRIVATE
  ROW_COUNTER_BUILD_EXECUTABLE)  
target_include_directories(row_counter_node_exec PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(row_counter_node_exec row_counter_node)

install(TARGETS row_counter_node row_counter_node_exec
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION lib/${PROJECT_NAME})

ament_target_dependencies(rail_detector_node
  rclcpp
  sensor_msgs
  image_transport
  cv_bridge
  ament_index_cpp
)

install(TARGETS rail_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

