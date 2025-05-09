#include <stdio.h>
#include "pico/stdlib.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/api/error_reporter.h" // Base class might still be needed by interpreter header
#include "tensorflow/lite/micro/micro_log.h" // Include for potential MicroPrintf

// Model headers generated for each quantization type
//#include "models/baseline.h"
//#include "models/dynamic.h"
#include "models/q-aware.h"
//#include "models/int8.h" 
//#include "models/float16.h"
//#include "models/w8_a16.h"

#define TF_LITE_STATIC_MEMORY
typedef struct {
    const unsigned char* model_data;
    unsigned int model_len;
    const char* model_name;
    bool is_quantized;
  } ModelConfig;

const ModelConfig baseline_config = {
  .model_data = models_q_aware_tflite,
  .model_len = models_q_aware_tflite_len,
  .model_name = "qat",
  .is_quantized = false
};
  

// Buffer for incoming data
#define MAX_SEGMENTS 1600
#define SEGMENT_LENGTH 60
float received_data[SEGMENT_LENGTH];
double average_runtime = 0.0;
uint32_t data_length = 0;


static tflite::MicroMutableOpResolver<10> resolver;  // Adjust size if needed

const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
constexpr int kTensorArenaSize = 16 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

//void set_clock_speed_khz(uint32_t khz) {
//  set_sys_clock_khz(khz, true);  // true = allow overclocking
//}


void normalize(float* X, int length) {
    const float min_val = 50.0f;
    const float max_val = 100.0f;
    const float epsilon = 1e-8f; // Small value to avoid division by zero
    
    for (int i = 0; i < length; i++) {
        // Clip the normalized value between 0 and 1
        float normalized = (X[i] - min_val) / (max_val - min_val + epsilon);
        
        // Apply clipping to ensure value is between 0 and 1
        if (normalized < 0.0f) {
            normalized = 0.0f;
        } else if (normalized > 1.0f) {
            normalized = 1.0f;
        }
        
        X[i] = normalized;
    }
}




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
  //printf("Received %d floats successfully\n", data_length);

  printf("First 5 values: ");
  for (uint32_t i = 0; i < (data_length < 5 ? data_length : 5); i++) {
      printf("%.6f ", received_data[i]);
  }
  printf("\n");

  return true;
}

void setup(const ModelConfig* config){
  // Initialize TFLite Micro
  tflite::InitializeTarget();
  printf("TFLite Micro initialized successfully!\n");
  // Load model
  model = tflite::GetModel(config->model_data);
  // --- Model Load Check ---
  if (model == nullptr || model->version() != TFLITE_SCHEMA_VERSION) {
    printf("ERROR: Failed to load model or version mismatch for %s!\n", config->model_name);
    MicroPrintf("ERROR: Failed to load model or version mismatch for %s!\n", config->model_name); // Try MicroPrintf too
    stdio_flush();
    return; // Exit early
  }
  printf("Model loaded successfully. Version: %d\n", model->version());
  stdio_flush();

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
  resolver.AddQuantize();            // For QUANTIZE ops
  resolver.AddDequantize();          // For DEQUANTIZE ops


  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  // --- Allocate Tensors ---
  printf("Allocating tensors...\n");
  stdio_flush();
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed for %s!\n", config->model_name);
    MicroPrintf("ERROR: AllocateTensors() failed for %s!\n", config->model_name);
    stdio_flush();
    return;
  }
  printf("Tensor allocation successful. Arena used: %d bytes\n", (int)interpreter->arena_used_bytes()); // Cast size_t to int for printf
  printf("Model inputs: %d, Model outputs: %d\n", (int)interpreter->inputs_size(), (int)interpreter->outputs_size()); // Cast size_t
  stdio_flush();

  
}
void run_benchmark(const ModelConfig* config) {
  // Get input/output tensors
    TfLiteTensor* input = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);
    printf("\nBenchmarking %s model...\n", config->model_name);
    uint32_t start_time = to_ms_since_boot(get_absolute_time());
    normalize(received_data, SEGMENT_LENGTH);
    printf("First 5 values: ");
    for (uint32_t i = 0; i < (data_length < 5 ? data_length : 5); i++) {
        printf("%.6f ", received_data[i]);
    }
    if (config->is_quantized) {
       // Quantize input: float → uint8
      float scale = input->params.scale;
      int zero_point = input->params.zero_point;
      uint8_t* quant_input = input->data.uint8;

      for (int i = 0; i < SEGMENT_LENGTH; i++) {
          int32_t quantized_val = (int32_t)(roundf(received_data[i] / scale) + zero_point);
          if (quantized_val > 255) quantized_val = 255;
          if (quantized_val < 0) quantized_val = 0;
          quant_input[i] = (uint8_t)quantized_val;
      }
    } else {
      memcpy(input->data.f, received_data, input->bytes);
    }
    if (interpreter->Invoke() != kTfLiteOk) {
      printf("Invoke Failed!\n");
    }
    printf("Output type: %d, bytes: %d\n", output->type, output->bytes);

    uint32_t duration = to_ms_since_boot(get_absolute_time()) - start_time;
    
    printf("Latency: %d\n", duration);
    //printf("Model size: %d KB\n", config->model_len / 1024);
    //printf("Arena used bytes: %d\n", interpreter.arena_used_bytes());
    printf("Result: ");
    if (config->is_quantized) {
        // Dequantize output: uint8 → float
        float scale = output->params.scale;
        int zero_point = output->params.zero_point;
        uint8_t* quant_output = output->data.uint8;

        for (int j = 0; j < output->bytes; j++) {
            float value = scale * ((int)quant_output[j] - zero_point);
            printf("%f ", value);
        }
    } else {
        for (int j = 0; j < output->bytes / sizeof(float); j++) {
            printf("%f ", output->data.f[j]);
        }
    }
        printf("\n");

    fflush(stdout);
    stdio_flush();
}

const int led_pin = 25;
int main() {
    // Initialize LED pin...
    gpio_init(led_pin);
    gpio_set_dir(led_pin, GPIO_OUT);
    

    stdio_init_all();
    printf("Initializing TFLite Micro...\n");
    sleep_ms(5000); // Wait for USB to initialize
    gpio_put(led_pin, true);
    printf("Testing pico-tflmicro...\n");
    sleep_ms(5000); // Wait for USB to initialize
    setup(&baseline_config);
    


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
            gpio_put(led_pin, false);
            run_benchmark(&baseline_config);
            printf("Benchmark Complete\n");
            gpio_put(led_pin, true);
        }
    }

    return 0;
}
