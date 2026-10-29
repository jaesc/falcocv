cmake_minimum_required(VERSION 3.20)
project(falco_kernel LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

find_package(pybind11 QUIET CONFIG)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_DIR_RAW
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT PYBIND11_DIR_RAW)
        message(FATAL_ERROR "pybind11 not found. Install: ${Python_EXECUTABLE} -m pip install pybind11")
    endif()
    file(TO_CMAKE_PATH "${PYBIND11_DIR_RAW}" PYBIND11_DIR)
    find_package(pybind11 CONFIG REQUIRED HINTS "${PYBIND11_DIR}")
endif()

pybind11_add_module(_falco_kernel MODULE src/kernel.cpp)
set_target_properties(_falco_kernel PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../bin"
)
install(TARGETS _falco_kernel
    LIBRARY DESTINATION falcocv
    RUNTIME DESTINATION falcocv
    ARCHIVE DESTINATION falcocv
)

if(MSVC)
    # Use the VC++ OpenMP runtime (vcomp140.dll), which ships with the
    # Microsoft Visual C++ Redistributable. /openmp:llvm would depend on
    # libomp140.x86_64.dll, which is not redistributable with the wheel.
    target_compile_options(_falco_kernel PRIVATE /O2 /openmp)
else()
    find_package(OpenMP REQUIRED)
    target_compile_options(_falco_kernel PRIVATE -O3)
    target_link_libraries(_falco_kernel PRIVATE OpenMP::OpenMP_CXX)
endif()
