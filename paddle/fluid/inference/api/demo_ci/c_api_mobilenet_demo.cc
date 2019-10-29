// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * This file contains demo of mobilenet for tensorrt.
 */

#include "paddle/fluid/inference/capi/c_api.h"
#include <gflags/gflags.h>
#include <glog/logging.h>  // use glog instead of CHECK to avoid importing other paddle header files.
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"  // NOLINT

DECLARE_double(fraction_of_gpu_memory_to_use);
DEFINE_string(modeldir, "", "Directory of the inference model.");
DEFINE_string(refer, "", "path to reference result for comparison.");
DEFINE_string(
    data, "",
    "path of data; each line is a record, format is "
    "'<space splitted floats as data>\t<space splitted ints as shape'");

namespace paddle {
namespace demo {

/*
 * Use the c-api to inference the demo.
 */
void Main() {
  PD_AnalysisConfig* config = PD_NewAnalysisConfig();
  const char* prog_file = FLAGS_modeldir + "/__model__\0";
  const char* params_file = FLAGS_modeldir + "/__params__\0";
  PD_SetModel(config, prog_file, params_file);

  PD_Tensor* input = PD_NewPaddleTensor();
  PD_PaddleBuf* buf = PD_NewPaddleBuf();
  PD_PaddleBufReset(buf, record.data.data(),
                    record.data.size() * sizeof(float));

  VLOG(3) << "begin to process data";
  // Just a single batch of data.
  char* line;
  size_t len = 0;
  FILE* fp;
  fp = fopen(FLAGS_data, "r");
  getline(&line, &len, fp);
  fclose(fp);
  Record record = ProcessALine(line);

  PD_SetPaddleTensorDType(input, PD_FLOAT32);
  PD_SetPaddleTensorShape(input, record.shape, record.shape.size());
  PD_SetPaddleTensorData(input, buf);

  PD_Tensor* out_data = PD_NewPaddleTensor();
  int out_size;
  PD_PredictorRun(config, input, 1, &out_data, &out_size, 1);

  printf("out_data size is: %s\n", out_size);
  printf("out_data name is: %s\n", PD_GetPaddleTensorName(out_data));
  printf("out_data data type is: %d\n", PD_GetPaddleTensorDType(out_data));
  PD_PaddleBuf* b = PD_GetPaddleTensorData(out_data);
  int length = PD_PaddleBufLength(b) / sizeof(float);
  printf("out_data's data's length is: %d\n", length);
  float* result = (float*)(PD_PaddleBufData(b));  // NOLINT
  printf("out_data's data is: ");
  for (int i = 0; i < length; ++i) {
    printf("%f ", result[i]);
  }
  printf("\n");
  int* size;
  int* out_shape;
  out_shape = PD_GetPaddleTensorShape(out_data, &size);
  printf("out_data shape is: {");
  for (int i = 0; i < *size; ++i) {
    printf("%d ", out_shape[i]);
  }
  printf("}\n");
}

}  // namespace demo
}  // namespace paddle

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  paddle::demo::Main();
  return 0;
}
