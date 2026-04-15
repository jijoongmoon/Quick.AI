// SPDX-License-Identifier: Apache-2.0
/**
 * @file   llm-qnn-test.h
 * @brief  QNN model extension template
 * @note   This file demonstrates how to create a custom QNN Quick.AI model
 *         by extending the base CausalLM class from nntrainer.
 *
 */

#ifndef __LLM_QNN_TEST_H__
#define __LLM_QNN_TEST_H__

#include "quick_dot_ai_qnn.h"

namespace causallm {

/**
 * @brief LLM_QNN_TEST class
 * @note  This is the main class you register with the Factory.
 *
 */
class LLM_QNN_TEST : public Quick_Dot_AI_QNN {

public:
  static constexpr const char *architectures = "LLM_QNN_TEST";

  LLM_QNN_TEST(json &cfg, json &generation_cfg, json &nntr_cfg)
      : Quick_Dot_AI_QNN(cfg, generation_cfg, nntr_cfg) {}

  virtual ~LLM_QNN_TEST() = default;

  void initialize_input_outputs();

  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = true) override;

private:
  // Input/output tensors
  uint16_t *attention_mask;
  uint16_t *sliding_attention_mask;
  uint16_t *generation_attention_mask;
  uint16_t *generation_sliding_attention_mask;

  uint16_t *position_ids_cos;
  uint16_t *position_ids_sin;
  uint16_t *swa_position_ids_cos;
  uint16_t *swa_position_ids_sin;
  uint16_t *prefill_position_ids_cos;
  uint16_t *prefill_position_ids_sin;
  uint16_t *prefill_swa_position_ids_cos;
  uint16_t *prefill_swa_position_ids_sin;
  uint16_t *generation_position_ids_cos;
  uint16_t *generation_position_ids_sin;
  uint16_t *generation_swa_position_ids_cos;
  uint16_t *generation_swa_position_ids_sin;

  float *input_sample;
  float *generation_sample;

  std::vector<ml::train::TensorDim::IO_TensorType> prefill_inputs;
  std::vector<ml::train::TensorDim::IO_TensorType> generation_inputs;

  // KV cache variables
  std::vector<uint16_t *> prefill_kvs;
  std::vector<int> prefill_kv_sizes;
  std::vector<uint16_t *> prefill_fresh_kvs;
  
  std::vector<uint16_t *> kvs;
  std::vector<int> kv_sizes;
  std::vector<uint16_t *> fresh_kvs;
};

} // namespace causallm

#endif
