cmake_minimum_required(VERSION 3.8)
project(scale_adjuster)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_compile_options(-O3)


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
ament_auto_add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME}
  tbb
)

target_include_directories(${PROJECT_NAME} PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PRIVATE ${PCL_DEFINITIONS})

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_auto_package()