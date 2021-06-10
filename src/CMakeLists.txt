cmake_minimum_required(VERSION 3.1)
project(ICP_CW1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use IMGUI"          ON)

find_package(LIBIGL REQUIRED QUIET)

#file(GLOB SRC_FILES *.cpp)

set(SRC_FILES ${PROJECT_SOURCE_DIR}/main.cpp
${PROJECT_SOURCE_DIR}/icp.cpp
${PROJECT_SOURCE_DIR}/scene.cpp
${PROJECT_SOURCE_DIR}/icp.h
${PROJECT_SOURCE_DIR}/scene.h
${PROJECT_SOURCE_DIR}/nanoflann.hpp
)
                          
add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} igl::core igl::opengl_glfw igl::opengl_glfw_imgui)
