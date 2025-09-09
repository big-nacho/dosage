cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

file(READ VERSION.txt PROJECT_VERSION)
project(pydosage VERSION ${PROJECT_VERSION} LANGUAGES C CXX)

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
    OUTPUT pydosage_cython.c
    COMMENT
    "Making ${CMAKE_CURRENT_BINARY_DIR}/pydosage_cython.c from ${CMAKE_CURRENT_SOURCE_DIR}/pydosage.pyx"
    COMMAND Python::Interpreter -m cython
    "${CMAKE_CURRENT_SOURCE_DIR}/src/pydosage/pydosage.pyx" --output-file pydosage_cython.c
    --module-name pydosage
    DEPENDS src/pydosage/pydosage.pyx
    VERBATIM
)

python_add_library(
    pydosage MODULE
    pydosage_cython.c
    ${LIB_SOURCES_LIST}
    WITH_SOABI
)

if (Python_NumPy_FOUND)
    target_compile_definitions(pydosage PRIVATE NPY_NO_DEPRECATED_API)
    target_include_directories(pydosage PRIVATE "${Python_NumPy_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Could not find NumPy headers")
endif()

target_compile_options(pydosage PRIVATE -Wall)
target_include_directories(pydosage PRIVATE lib/include)
install(TARGETS pydosage DESTINATION pydosage)
