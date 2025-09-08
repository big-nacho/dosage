cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

file(READ VERSION.txt PROJECT_VERSION)
project(dosage VERSION ${PROJECT_VERSION} LANGUAGES C CXX)

set(
    LIB_SOURCES_LIST
    lib/src/dosage.c
)

find_package(
    Python
    COMPONENTS Interpreter Development.Module NumPy
    REQUIRED
)

add_custom_command(
    OUTPUT dosage_cython.c
    COMMENT
    "Making ${CMAKE_CURRENT_BINARY_DIR}/dosage_cython.c from ${CMAKE_CURRENT_SOURCE_DIR}/dosage.pyx"
    COMMAND Python::Interpreter -m cython
    "${CMAKE_CURRENT_SOURCE_DIR}/src/dosage/dosage.pyx" --output-file dosage_cython.c
    --module-name dosage
    DEPENDS src/dosage/dosage.pyx
    VERBATIM
)

python_add_library(
    dosage MODULE
    dosage_cython.c
    ${LIB_SOURCES_LIST}
    WITH_SOABI
)

if (Python_NumPy_FOUND)
    target_compile_definitions(dosage PRIVATE NPY_NO_DEPRECATED_API)
    target_include_directories(dosage PRIVATE "${Python_NumPy_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Could not find NumPy headers")
endif()

find_library(PROFILER_LIB profiler REQUIRED)
target_link_libraries(dosage PRIVATE ${PROFILER_LIB})
include_directories("/opt/homebrew/include")
target_compile_options(dosage PRIVATE -g)

target_compile_options(dosage PRIVATE -Wall)
target_include_directories(dosage PRIVATE lib/include)
install(TARGETS dosage DESTINATION dosage)
