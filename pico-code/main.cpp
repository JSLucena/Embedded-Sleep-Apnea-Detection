#include <stdio.h>
#include "pico/stdlib.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// TensorFlow Lite globals
namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    
    // Tensor arena (adjust size based on model requirements)
    constexpr int kTensorArenaSize = 10 * 1024;  // 10KB (adjust as needed)
    alignas(16) uint8_t tensor_arena[kTensorArenaSize];
    }  // namespace

const int led_pin = 25;
int main() {
    // Initialize LED pin...
    gpio_init(led_pin);
    gpio_set_dir(led_pin, GPIO_OUT);
    gpio_put(led_pin, true);

    stdio_init_all();
    printf("Initializing TFLite Micro...\n");
    sleep_ms(20000); // Wait for USB to initialize

    printf("Testing pico-tflmicro...\n");

    // Initialize TFLite Micro
    tflite::InitializeTarget();
    printf("TFLite Micro initialized successfully!\n");

    while (true) {
        gpio_put(led_pin, true);
        sleep_ms(100);
        gpio_put(led_pin, false);
        sleep_ms(900);
        printf("System running...\n");
    }

    return 0;
}
