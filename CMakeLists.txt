cmake_minimum_required(VERSION 3.0)
project(BE)

find_package(Torch REQUIRED)
add_executable(BE main.cpp Net.cpp Net.h Config.cpp Config.h)
target_link_libraries(BE "${TORCH_LIBRARIES}")
set_property(TARGET BE PROPERTY CXX_STANDARD 14)
