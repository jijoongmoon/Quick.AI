// SPDX-License-Identifier: Apache-2.0
/**
 * @file   context.cpp
 * @brief  Out-of-line key function for nntrainer::Context (and the
 *         trivial one for ContextData). Pins the typeinfo / vtable
 *         emission of both base classes to libnntrainer.so so that
 *         cross-DSO dynamic_cast<Context &> / dynamic_cast<ContextData &>
 *         from an Android-dlopen'd plugin DSO resolves against a
 *         single exported type_info object instead of a weak local
 *         copy. See layer_devel.cpp for the exact same rationale
 *         applied to Layer.
 */
#include <context.h>

namespace nntrainer {

Context::~Context() = default;

ContextData::~ContextData() = default;

} // namespace nntrainer
