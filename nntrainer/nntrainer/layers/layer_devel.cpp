// SPDX-License-Identifier: Apache-2.0
/**
 * @file   layer_devel.cpp
 * @brief  Out-of-line key function for nntrainer::Layer.
 *
 * Forces Layer's typeinfo and vtable to be emitted exactly once — in
 * the translation unit that ships with libnntrainer.so — instead of as
 * a weak COMDAT in every TU that includes layer_devel.h. Without this
 * the Android bionic dynamic linker keeps multiple private copies of
 * Layer's type_info across DSOs, which makes cross-DSO
 * dynamic_cast<Layer &> throw std::bad_cast when a plugin (such as
 * libqnn_context.so) is dlopen'd.
 *
 * Keep this file minimal: anything else added here would leak
 * nntrainer internals out of libnntrainer.so's well-defined ABI
 * surface and risk versioning pain.
 */
#include <layer_devel.h>

namespace nntrainer {

Layer::~Layer() = default;

} // namespace nntrainer
