cmake_minimum_required(VERSION 3.13)

# Include the Pico SDK
# Set these variables BEFORE pico_sdk_init
set(PICO_PLATFORM "rp2350")  # Or appropriate platform for RP2350
set(PICO_BOARD "pico2_w")     # Replace with the correct board type for RP2350
set(PICO_TARGET_PROCESSOR "cortex-m33")
include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)
include($ENV{PICO_EXTRAS_PATH}/external/pico_extras_import.cmake)
# Project setup
project(tflite_test C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

# Set paths
set(PICO_SLEEP_DIR "$ENV{PICO_EXTRAS_PATH}/src/rp2_common/pico_sleep")
# Create pico_sleep library manually if needed
if(EXISTS ${PICO_SLEEP_DIR}/sleep.c)
    add_library(pico_sleep_manual STATIC
        ${PICO_SLEEP_DIR}/sleep.c
    )
    target_include_directories(pico_sleep_manual PUBLIC
        ${PICO_SLEEP_DIR}/include
    )
    target_link_libraries(pico_sleep_manual
        pico_stdlib
        pico_aon_timer
        hardware_clocks
        hardware_rosc
        hardware_xosc
        hardware_powman
        #hardware_rtc
    )
endif()



set(TFLITE_MICRO_DIR "/home/jslucena/Work/tflite-micro")
#set(TFLITE_MICRO_LIB_DIR "${TFLITE_MICRO_DIR}/gen/cortex_m_generic_cortex-m0plus_default_gcc/lib")
#set(TFLITE_MICRO_LIB_DIR "${TFLITE_MICRO_DIR}/gen/cortex_m_generic_cortex-m0plus_default_cmsis_nn_gcc/lib")

#set(TFLITE_MICRO_LIB_DIR "${TFLITE_MICRO_DIR}/gen/cortex_m_generic_cortex-m33_default_gcc/lib")
set(TFLITE_MICRO_LIB_DIR "${TFLITE_MICRO_DIR}/gen/cortex_m_generic_cortex-m33_default_cmsis_nn_gcc/lib")
set(TFLITE_MICRO_LIB "${TFLITE_MICRO_LIB_DIR}/libtensorflow-microlite.a")

message(STATUS "Using TensorFlow Lite Micro static library: ${TFLITE_MICRO_LIB}")

# Add executable
add_executable(tflite_test
    main.cpp
)

# Include directories
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
    ${PICO_SLEEP_DIR}/include
)

# Define macros consistently
target_compile_definitions(tflite_test PRIVATE
    TF_LITE_STATIC_MEMORY
    TF_LITE_MCU_DEBUG_LOG
)
target_compile_options(tflite_test PRIVATE
    -O3
    -fno-exceptions
)
target_link_options(tflite_test PRIVATE
    -Wl,--print-memory-usage
)
# Compiler flags (note: -static has no effect here, optional)
#set_target_properties(tflite_test PROPERTIES
#    COMPILE_FLAGS "-fno-exceptions"
#)

# Link libraries
target_link_libraries(tflite_test
    pico_stdlib
    hardware_clocks
    hardware_rosc     # This is building from pico-extras
    hardware_xosc
    #hardware_rtc
    hardware_powman
    pico_time
    hardware_gpio
    hardware_irq
    pico_sleep_manual
    ${TFLITE_MICRO_LIB}
)

# USB for printf/debug
pico_enable_stdio_usb(tflite_test 1)
pico_enable_stdio_uart(tflite_test 0)

# Generate UF2 and bin files
pico_add_extra_outputs(tflite_test)
