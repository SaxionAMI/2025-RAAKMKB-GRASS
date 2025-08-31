cmake_minimum_required(VERSION 3.14)
project(audio_processing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release unless explicitly set (non-MSVC)
if(NOT CMAKE_BUILD_TYPE AND NOT MSVC)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(WIN32)
  # ---- Windows (MSVC + vcpkg) ----
  include_directories(
    "C:/Python313/include"
    "C:/Users/Gebruiker/Desktop/GRASS_Pipeline/venv/Lib/site-packages/pybind11/include"
    ${CMAKE_SOURCE_DIR}/../vcpkg/installed/x64-windows/include
  )
  link_directories(
    "C:/Python313/libs"
    ${CMAKE_SOURCE_DIR}/../vcpkg/installed/x64-windows/lib
  )

  set(PYTHON_LIB python313)
  set(PYTHON_MODULE_EXTENSION ".pyd")

  # Name FFTW lib explicitly on Windows
  set(FFTW_LIBRARIES fftw3f)

else()
  # ---- Linux / Docker ----
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  find_package(pybind11 REQUIRED)

  find_path(FFTW_INCLUDE_DIR fftw3.h)
  find_library(FFTW_LIBRARY fftw3f)

  include_directories(${Python3_INCLUDE_DIRS})
  include_directories(${FFTW_INCLUDE_DIR})
  link_directories(${Python3_LIBRARY_DIRS})

  set(PYTHON_LIB ${Python3_LIBRARIES})
  set(PYTHON_MODULE_EXTENSION ".so")
  set(FFTW_LIBRARIES ${FFTW_LIBRARY})
endif()

# Build the pybind11 module
add_library(audio_cpp MODULE
  bindings.cpp
  processing.cpp
)

# Correct file extension for importable Python module
set_target_properties(audio_cpp PROPERTIES
  PREFIX ""
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# MSVC optimizations (apply to this target only)
target_compile_options(audio_cpp PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/O2>
  $<$<CXX_COMPILER_ID:MSVC>:/fp:fast>
  $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
)

# Link required libraries
target_link_libraries(audio_cpp PRIVATE ${PYTHON_LIB} ${FFTW_LIBRARIES})
