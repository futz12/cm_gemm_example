cmake_minimum_required(VERSION 3.15)
project(vulkan_cm_gemm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

add_executable(vulkan_cm_gemm
    main.cpp
    gemm_cm.comp
)

target_include_directories(vulkan_cm_gemm PRIVATE
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(vulkan_cm_gemm PRIVATE
    Vulkan::Vulkan
)

if(MSVC)
    target_compile_options(vulkan_cm_gemm PRIVATE /W4)
else()
    target_compile_options(vulkan_cm_gemm PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(vulkan_cm_gemm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
