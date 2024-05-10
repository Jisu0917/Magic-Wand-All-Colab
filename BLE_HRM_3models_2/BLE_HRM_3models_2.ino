// magic_wand_heart
#include <TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "magic_wand_model_data.h"
#include "rasterize_stroke.h"
#include "imu_provider.h"

#define BLE_SENSE_UUID(val) ("4798e0f2-" val "-4d68-af64-8a8f5258404e")

namespace {

  const int VERSION = 0x00000000;

  // Constants for image rasterization
  constexpr int raster_width = 32;
  constexpr int raster_height = 32;
  constexpr int raster_channels = 3;
  constexpr int raster_byte_count = raster_height * raster_width * raster_channels;
  int8_t raster_buffer[raster_byte_count];

  // BLE settings
  BLEService        service                       (BLE_SENSE_UUID("0000"));
  BLECharacteristic strokeCharacteristic          (BLE_SENSE_UUID("300a"), BLERead, stroke_struct_byte_count);
  
  // String to calculate the local and device name
  String name;
  
  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 30 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
  
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  
  // -------------------------------------------------------------------------------- //
  // UPDATE THESE VARIABLES TO MATCH THE NUMBER AND LIST OF GESTURES IN YOUR DATASET  //
  // -------------------------------------------------------------------------------- //
  constexpr int label_count_digit = 15;
  const char* labels_digit[label_count_digit] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
  constexpr int label_count_figure = 3;
  const char* labels_figure[label_count_figure] = {"circle", "heart", "star"};
  constexpr int label_count_alphabet = 2;
  const char* labels_alphabet[label_count_alphabet] = {"a", "b"};
}  // namespace


// BLE_HRM
#include <ArduinoBLE.h>
#include <Wire.h>

#include "algorithm_by_RF.h"
#include "max30102.h"

const char serviceUuid[] = "8bff20de-32fb-4350-bddb-afe103ef9640";
const char characteristicUuid[] = "1c8dd778-e8c3-45b0-a9f3-48c33a400315";
const char accuracyCharacteristicUuid[] = "b8ae0c39-6204-407c-aa43-43087ec29a63";
const char flagCharacteristicUuid[] = "6f9a8b7c-5d2e-4a1f-91be-837c6d09f102";

BLEService hrmService(serviceUuid);
BLEIntCharacteristic heartRateReading(characteristicUuid, BLERead | BLENotify);
BLEUnsignedIntCharacteristic accuracyReading(accuracyCharacteristicUuid, BLERead | BLENotify);
BLEIntCharacteristic flagWriting(flagCharacteristicUuid, BLEWrite | BLENotify);

const byte oxiInt = 10; // pin connected to MAX30102 INT

uint8_t uch_dummy;

uint32_t aun_ir_buffer[BUFFER_SIZE]; //infrared LED sensor data
uint32_t aun_red_buffer[BUFFER_SIZE];  //red LED sensor data


void setup() {
  // Start serial
  Serial.begin(9600);
  Serial.println("Started");

  // Start IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialized IMU!");
    while (1);
  }
  SetupIMU();

  // BLE_HRM
  Wire.begin();

  maxim_max30102_reset(); //resets the MAX30102
  delay(1000);

  maxim_max30102_read_reg(REG_INTR_STATUS_1,&uch_dummy);  //Reads/clears the interrupt status register
  maxim_max30102_init();  //initialize the MAX30102

  if (!BLE.begin()) {
    Serial.println("starting BLE failed!");
    while (1);
  }
  String address = BLE.address();

  // Output BLE settings over Serial
  Serial.print("address = ");
  Serial.println(address);


  BLE.setLocalName("MyPet-BLE");
  BLE.setAdvertisedService(hrmService);
  hrmService.addCharacteristic(heartRateReading);
  hrmService.addCharacteristic(accuracyReading);
  hrmService.addCharacteristic(flagWriting);
  BLE.addService(hrmService);

  BLE.advertise();

  Serial.println("Starting!");

  // magic_wand_heart
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  static tflite::MicroErrorReporter micro_error_reporter;  // NOLINT
  error_reporter = &micro_error_reporter;

}

void loop() {
  BLEDevice central = BLE.central();

  // magic_wand_heart
  // if a central is connected to the peripheral:
  static bool was_connected_last = false;  
  if (central && !was_connected_last) {
    Serial.print("Connected to central: ");
    // print the central's BT address:
    Serial.println(central.address());
  }
  was_connected_last = central;

  while (central.connected()) {
    // make sure IMU data is available then read in data
    const bool data_available = IMU.accelerationAvailable() || IMU.gyroscopeAvailable();
    if (!data_available) {
      return;
    }
    int accelerometer_samples_read;
    int gyroscope_samples_read;
    ReadAccelerometerAndGyroscope(&accelerometer_samples_read, &gyroscope_samples_read);

    // Parse and process IMU data
    bool done_just_triggered = false;
    if (gyroscope_samples_read > 0) {
      EstimateGyroscopeDrift(current_gyroscope_drift);
      UpdateOrientation(gyroscope_samples_read, current_gravity, current_gyroscope_drift);
      UpdateStroke(gyroscope_samples_read, &done_just_triggered);
      if (central && central.connected()) {
        strokeCharacteristic.writeValue(stroke_struct_buffer, stroke_struct_byte_count);
      }
    }
    if (accelerometer_samples_read > 0) {
      EstimateGravityDirection(current_gravity);
      UpdateVelocity(accelerometer_samples_read, current_gravity);
    }

    // Wait for a gesture to be done
    if (done_just_triggered) {
      // Rasterize the gesture
      RasterizeStroke(stroke_points, *stroke_transmit_length, 0.6f, 0.6f, raster_width, raster_height, raster_buffer);
      for (int y = 0; y < raster_height; ++y) {
        char line[raster_width + 1];
        for (int x = 0; x < raster_width; ++x) {
          const int8_t* pixel = &raster_buffer[(y * raster_width * raster_channels) + (x * raster_channels)];
          const int8_t red = pixel[0];
          const int8_t green = pixel[1];
          const int8_t blue = pixel[2];
          char output;
          if ((red > -128) || (green > -128) || (blue > -128)) {
            output = '#';
          } else {
            output = '.';
          }
          line[x] = output;
        }
        line[raster_width] = 0;
        Serial.println(line);
      }

      int label_count = 0;
      const char** labels;

      // Check if there's data available to read from flagWriting characteristic
      if (flagWriting.written()) {
        // Read the value written to flagWriting characteristic
        int flag_value = flagWriting.value();
        Serial.print("Received flag value: ");
        Serial.println(flag_value);

        if (flag_value == 1) {
          // Map the model into a usable data structure. This doesn't involve any
          // copying or parsing, it's a very lightweight operation.
          model = tflite::GetModel(g_magic_wand_model_digit_data);
          if (model->version() != TFLITE_SCHEMA_VERSION) {
            TF_LITE_REPORT_ERROR(error_reporter,
                                "Model provided is schema version %d not equal "
                                "to supported version %d.",
                                model->version(), TFLITE_SCHEMA_VERSION);
            return;
          }

          // Pull in only the operation implementations we need.
          // This relies on a complete list of all the ops needed by this graph.
          // An easier approach is to just use the AllOpsResolver, but this will
          // incur some penalty in code space for op implementations that are not
          // needed by this graph.
          static tflite::MicroMutableOpResolver<4> micro_op_resolver;  // NOLINT
          micro_op_resolver.AddConv2D();
          micro_op_resolver.AddMean();
          micro_op_resolver.AddFullyConnected();
          micro_op_resolver.AddSoftmax();

          // Build an interpreter to run the model with.
          static tflite::MicroInterpreter static_interpreter(
              model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
          interpreter = &static_interpreter;

          // Allocate memory from the tensor_arena for the model's tensors.
          interpreter->AllocateTensors();

          // Set model input settings
          TfLiteTensor* model_input = interpreter->input(0);
          if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
              (model_input->dims->data[1] != raster_height) ||
              (model_input->dims->data[2] != raster_width) ||
              (model_input->dims->data[3] != raster_channels) ||
              (model_input->type != kTfLiteInt8)) {
            TF_LITE_REPORT_ERROR(error_reporter,
                                "Bad input tensor parameters in model");
            return;
          }

          label_count = label_count_digit;
          labels = labels_digit;

          // Set model output settings
          TfLiteTensor* model_output = interpreter->output(0);
          if ((model_output->dims->size != 2) || (model_output->dims->data[0] != 1) ||
              (model_output->dims->data[1] != label_count) ||
              (model_output->type != kTfLiteInt8)) {
            TF_LITE_REPORT_ERROR(error_reporter,
                                "Bad output tensor parameters in model");
            return;
          }



        }
        else if (flag_value == 2) {
          // Map the model into a usable data structure. This doesn't involve any
          // copying or parsing, it's a very lightweight operation.
          model = tflite::GetModel(g_magic_wand_model_figure_data);
          if (model->version() != TFLITE_SCHEMA_VERSION) {
            TF_LITE_REPORT_ERROR(error_reporter,
                                "Model provided is schema version %d not equal "
                                "to supported version %d.",
                                model->version(), TFLITE_SCHEMA_VERSION);
            return;
          }

          // Pull in only the operation implementations we need.
          // This relies on a complete list of all the ops needed by this graph.
          // An easier approach is to just use the AllOpsResolver, but this will
          // incur some penalty in code space for op implementations that are not
          // needed by this graph.
          static tflite::MicroMutableOpResolver<4> micro_op_resolver;  // NOLINT
          micro_op_resolver.AddConv2D();
          micro_op_resolver.AddMean();
          micro_op_resolver.AddFullyConnected();
          micro_op_resolver.AddSoftmax();

          // Build an interpreter to run the model with.
          static tflite::MicroInterpreter static_interpreter(
              model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
          interpreter = &static_interpreter;

          // Allocate memory from the tensor_arena for the model's tensors.
          interpreter->AllocateTensors();

          // Set model input settings
          TfLiteTensor* model_input = interpreter->input(0);
          if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
              (model_input->dims->data[1] != raster_height) ||
              (model_input->dims->data[2] != raster_width) ||
              (model_input->dims->data[3] != raster_channels) ||
              (model_input->type != kTfLiteInt8)) {
            TF_LITE_REPORT_ERROR(error_reporter,
                                "Bad input tensor parameters in model");
            return;
          }

          label_count = label_count_figure;
          labels = labels_figure;

          // Set model output settings
          TfLiteTensor* model_output = interpreter->output(0);
          if ((model_output->dims->size != 2) || (model_output->dims->data[0] != 1) ||
              (model_output->dims->data[1] != label_count) ||
              (model_output->type != kTfLiteInt8)) {
            TF_LITE_REPORT_ERROR(error_reporter,
                                "Bad output tensor parameters in model");
            return;
          }




        } else if (flag_value == 3) {
          // Map the model into a usable data structure. This doesn't involve any
          // copying or parsing, it's a very lightweight operation.
          model = tflite::GetModel(g_magic_wand_model_alphabet_data);
          if (model->version() != TFLITE_SCHEMA_VERSION) {
            TF_LITE_REPORT_ERROR(error_reporter,
                                "Model provided is schema version %d not equal "
                                "to supported version %d.",
                                model->version(), TFLITE_SCHEMA_VERSION);
            return;
          }

          // Pull in only the operation implementations we need.
          // This relies on a complete list of all the ops needed by this graph.
          // An easier approach is to just use the AllOpsResolver, but this will
          // incur some penalty in code space for op implementations that are not
          // needed by this graph.
          static tflite::MicroMutableOpResolver<4> micro_op_resolver;  // NOLINT
          micro_op_resolver.AddConv2D();
          micro_op_resolver.AddMean();
          micro_op_resolver.AddFullyConnected();
          micro_op_resolver.AddSoftmax();

          // Build an interpreter to run the model with.
          static tflite::MicroInterpreter static_interpreter(
              model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
          interpreter = &static_interpreter;

          // Allocate memory from the tensor_arena for the model's tensors.
          interpreter->AllocateTensors();

          // Set model input settings
          TfLiteTensor* model_input = interpreter->input(0);
          if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
              (model_input->dims->data[1] != raster_height) ||
              (model_input->dims->data[2] != raster_width) ||
              (model_input->dims->data[3] != raster_channels) ||
              (model_input->type != kTfLiteInt8)) {
            TF_LITE_REPORT_ERROR(error_reporter,
                                "Bad input tensor parameters in model");
            return;
          }

          label_count = label_count_alphabet;
          labels = labels_alphabet;

          // Set model output settings
          TfLiteTensor* model_output = interpreter->output(0);
          if ((model_output->dims->size != 2) || (model_output->dims->data[0] != 1) ||
              (model_output->dims->data[1] != label_count) ||
              (model_output->type != kTfLiteInt8)) {
            TF_LITE_REPORT_ERROR(error_reporter,
                                "Bad output tensor parameters in model");
            return;
          }
        } else {
          return;
        }

        // Pass to the model and run the interpreter
        TfLiteTensor* model_input = interpreter->input(0);
        for (int i = 0; i < raster_byte_count; ++i) {
          model_input->data.int8[i] = raster_buffer[i];
        }
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
          TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
          return;
        }
        TfLiteTensor* output = interpreter->output(0);

        // Parse the model output
        int8_t max_score;
        int max_index;
        for (int i = 0; i < label_count; ++i) {
          const int8_t score = output->data.int8[i];
          if ((i == 0) || (score > max_score)) {
            max_score = score;
            max_index = i;
          }
        }
        TF_LITE_REPORT_ERROR(error_reporter, "Found %s (%d)", labels[max_index], max_score);
        
        Serial.println(labels[max_index]);
        heartRateReading.writeValue(max_index);
        accuracyReading.writeValue(max_score);
      } 
    }
  }
}
