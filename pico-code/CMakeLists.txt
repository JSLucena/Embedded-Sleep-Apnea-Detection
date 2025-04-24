cmake_minimum_required(VERSION 3.13)

# Include the Pico SDK
include ($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

# Project setup
project(tflite_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Initialize the Pico SDK
pico_sdk_init()

# Add path to TFLite Micro static library
set(TFLITE_MICRO_LIB_DIR "/home/jslucena/Work/tflite-micro/gen/cortex_m_generic_cortex-m0plus_default_gcc/lib")

set(TFLITE_MICRO_DIR "/home/jslucena/Work/tflite-micro")
set(TFLITE_MICRO_LIB "${TFLITE_MICRO_LIB_DIR}/libtensorflow-microlite.a")

message(STATUS "Using TensorFlow Lite Micro static library: ${TFLITE_MICRO_LIB}")
message(STATUS "TFLITE_MICRO_DIR is set to: ${TFLITE_MICRO_DIR}")
# Add your main application source file
add_executable(tflite_test
    main.cpp
)
# Include directories (minimal, only for your application)
target_include_directories(tflite_test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/.
    ${TFLITE_MICRO_DIR}
    ${TFLITE_MICRO_DIR}/tensorflow
    ${TFLITE_MICRO_DIR}/tensorflow/lite
    ${TFLITE_MICRO_DIR}/tensorflow/lite/micro
    ${TFLITE_MICRO_DIR}/tensorflow/lite/core/api
    ${TFLITE_MICRO_DIR}/tensorflow/lite/kernels
    ${TFLITE_MICRO_DIR}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
    ${TFLITE_MICRO_DIR}/tensorflow/lite/micro/tools/make/downloads/gemmlowp
)

# Compiler flags
set_target_properties(tflite_test PROPERTIES
    COMPILE_FLAGS "-fno-rtti -fno-exceptions -fno-threadsafe-statics"
)

# Link libraries
target_link_libraries(tflite_test
    pico_stdlib
    ${TFLITE_MICRO_LIB}  # Link against the prebuilt TFLite Micro library
)

# Enable USB output for debug messages
pico_enable_stdio_usb(tflite_test 1)
pico_enable_stdio_uart(tflite_test 0)

# Generate additional output files (e.g., .uf2)
pico_add_extra_outputs(tflite_test)