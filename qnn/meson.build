# qnn/meson.build — builds libqnn_context.so (android + enable-qnn only)

if not enable_qnn
  subdir_done()
endif

# ── QNN SDK wrapper sources ─────────────────────────────────────────────
qnn_sdk_sources = files(
  'jni/qnn/Log/Logger.cpp',
  'jni/qnn/Log/LogUtils.cpp',
  'jni/qnn/PAL/src/common/GetOpt.cpp',
  'jni/qnn/PAL/src/common/StringOp.cpp',
  'jni/qnn/PAL/src/linux/Directory.cpp',
  'jni/qnn/PAL/src/linux/DynamicLoading.cpp',
  'jni/qnn/PAL/src/linux/FileOp.cpp',
  'jni/qnn/PAL/src/linux/Path.cpp',
  'jni/qnn/Utils/DataUtil.cpp',
  'jni/qnn/Utils/DynamicLoadUtil.cpp',
  'jni/qnn/Utils/IOTensor.cpp',
  'jni/qnn/Utils/QnnSampleAppUtils.cpp',
  'jni/qnn/WrapperUtils/QnnWrapperUtils.cpp',
  'jni/qnn/op/QNNGraph.cpp',
  'jni/qnn/op/QNNLinear.cpp',
  'jni/qnn/qnn_properties.cpp',
  'jni/qnn-api/BackendExtensions.cpp',
  'jni/qnn-api/dlwrap.cpp',
)

# ── QNN context sources ─────────────────────────────────────────────────
qnn_context_sources = files(
  'qnn_context.cpp',
  'jni/qnn_rpc_manager.cpp',
)

# ── Include directories ──────────────────────────────────────────────────
qnn_inc = include_directories(
  '.',
  'jni',
  'jni/qnn',
  'jni/qnn/QNN',
  'jni/qnn/Log',
  'jni/qnn/PAL/include',
  'jni/qnn/Utils',
  'jni/qnn/WrapperUtils',
  'jni/qnn/op',
  'jni/qnn-api',
)

# ── Build libqnn_context.so ──────────────────────────────────────────────
#
# -fvisibility=default: vague-linkage typeinfo symbols (for
# nntrainer::Context et al.) need to share one copy between
# libnntrainer.so and libqnn_context.so, otherwise libc++'s
# dynamic_cast<T&> throws std::bad_cast at dlopen time during the
# plugin's static initializers.
#
# -fno-rtti: disables RTTI emission for this library and makes every
# dynamic_cast / typeid inside it a compile error. The sources we
# actually compile here (Qualcomm's netrun / sample_app / PAL helpers
# plus our own QNNContext / QNNGraph / QNNLinear wrappers) do not use
# dynamic_cast on anything polymorphic, so -fno-rtti succeeds — and
# once it succeeds no std::bad_cast can come out of this .so's static
# initializers. Pair this with -fvisibility=default so the plugin
# still satisfies the weak vtable / typeinfo references that are
# required by virtual function dispatch *against* libnntrainer.so's
# classes.
qnn_context_lib = shared_library('qnn_context',
  qnn_sdk_sources + qnn_context_sources,
  include_directories: [qnn_inc],
  dependencies: [nntrainer_dep, log_dep],
  cpp_args: ['-DENABLE_QNN=1', '-DPLUGGABLE=1',
             '-fvisibility=default',
             '-fno-rtti'],
  install: true,
)
