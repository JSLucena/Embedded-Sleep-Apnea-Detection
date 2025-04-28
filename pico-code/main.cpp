#include <stdio.h>
#include "pico/stdlib.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// Model headers generated for each quantization type
#include "models/baseline.h"
//#include "models/q-aware.h"
//#include "models/int8.h" 
//#include "models/float16.h"
//#include "models/w8_16.h"

typedef struct {
    const unsigned char* model_data;
    unsigned int model_len;
    const char* model_name;
    bool is_quantized;
  } ModelConfig;

const ModelConfig baseline_config = {
  .model_data = models_baseline_tflite,
  .model_len = models_baseline_tflite_len,
  .model_name = "Baseline (float32)",
  .is_quantized = false
};
  

// Buffer for incoming data
#define MAX_SEGMENTS 1600
#define SEGMENT_LENGTH 60
float received_data[SEGMENT_LENGTH];
double average_runtime = 0.0;
uint32_t data_length = 0;

bool receive_float_array() {
  printf("Waiting to receive %d floats (%d bytes)...\n", SEGMENT_LENGTH, SEGMENT_LENGTH * sizeof(float));

  uint8_t* float_bytes = (uint8_t*)received_data;
  uint32_t total_bytes = SEGMENT_LENGTH * sizeof(float);

  uint32_t bytes_read = 0;
  
  while (bytes_read < total_bytes) {
      int c = getchar_timeout_us(0);
      if (c != PICO_ERROR_TIMEOUT) { 
          float_bytes[bytes_read++] = (uint8_t)c;
      } else {
          sleep_ms(1);
      }
  }

  data_length = SEGMENT_LENGTH;
  printf("Received %d floats successfully\n", data_length);

  printf("First 5 values: ");
  for (uint32_t i = 0; i < (data_length < 5 ? data_length : 5); i++) {
      printf("%.6f ", received_data[i]);
  }
  printf("\n");

  return true;
}

void run_benchmark(const ModelConfig* config) {
    printf("\nBenchmarking %s model...\n", config->model_name);
    
    // Load model
    const tflite::Model* model = tflite::GetModel(config->model_data);
    
    // Initialize resolver with all operations needed for your model
    static tflite::MicroMutableOpResolver<10> resolver;  // Adjust size if needed

    // Add ops based on your model's requirements:
    resolver.AddFullyConnected();      // For FULLY_CONNECTED layers
    resolver.AddConv2D();              // For CONV_2D layers
    resolver.AddMaxPool2D();           // For MAX_POOL_2D layers
    resolver.AddLogistic();            // For LOGISTIC (sigmoid) activation
    resolver.AddReshape();             // For RESHAPE operations
    resolver.AddStridedSlice();        // For STRIDED_SLICE ops
    resolver.AddPack();                // For PACK ops (if used in model)
    resolver.AddShape();               // For SHAPE ops (if used in model)

    // Quantization-related ops (if your model is quantized)
    //resolver.AddQuantize();            // For QUANTIZE ops
    //resolver.AddDequantize();          // For DEQUANTIZE ops
  
    // Allocate arena (adjust size per model if needed)
    constexpr int kTensorArenaSize = 16 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];
  
    tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    
    if (interpreter.AllocateTensors() != kTfLiteOk) {
      printf("Allocation failed for %s!\n", config->model_name);
      return;
    }
  
    // Get input/output tensors
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
  
    // Benchmark loop
    uint32_t start_time = to_ms_since_boot(get_absolute_time());
    for (int i = 0; i < 100; i++) { // Run 100 inferences
      // Fill input with test data (adjust for your model)
      if (config->is_quantized) {
        // For quantized models

        //for (int j = 0; j < input->bytes; j++) {
        //  input->data.int8[j] = ...; // Your quantized input
        //}
      } else {
        // For float models
        for (int j = 0; j < input->bytes / sizeof(float); j++) {
          input->data.f[j] = received_data[j]; // Your float input
        }
      }
  
      if (interpreter.Invoke() != kTfLiteOk) {
        printf("Inference failed!\n");
        break;
      }
    }
    uint32_t duration = to_ms_since_boot(get_absolute_time()) - start_time;
    
    printf("Avg latency: %.2f ms\n", duration / 100.0f);
    printf("Model size: %d KB\n", config->model_len / 1024);
    printf("Arena used bytes: %d", interpreter.arena_used_bytes());
}

const int led_pin = 25;
int main() {
    // Initialize LED pin...
    gpio_init(led_pin);
    gpio_set_dir(led_pin, GPIO_OUT);
    gpio_put(led_pin, true);

    stdio_init_all();
    printf("Initializing TFLite Micro...\n");
    sleep_ms(5000); // Wait for USB to initialize

    printf("Testing pico-tflmicro...\n");

    // Initialize TFLite Micro
    tflite::InitializeTarget();
    printf("TFLite Micro initialized successfully!\n");


    //run_benchmark(baseline_config);
    gpio_put(led_pin, false);
    sleep_ms(500);
    gpio_put(led_pin, true);
    sleep_ms(500);
    gpio_put(led_pin, false);
    sleep_ms(500);
    gpio_put(led_pin, true);
    sleep_ms(500);
    printf("Pico ready to receive data...\n");
    while (true) {
        if (receive_float_array()) {
            printf("data recieved\n");
        }
    }

    return 0;
}
