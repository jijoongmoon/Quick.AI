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
# -fvisibility=default: vague-linkage typeinfo and vtable symbols have
# to be a single exported copy between libnntrainer.so and
# libqnn_context.so, otherwise libc++ ends up comparing two hidden
# type_info objects across the DSO boundary and throws std::bad_cast
# out of dlopen while the plugin's static initialisers run.
#
# (We tried -fno-rtti earlier but several nntrainer headers pulled in
# through common_properties.h / node_exporter.h use `typeid(*t).name()`
# in inline template bodies, so disabling RTTI breaks compilation.)
qnn_context_lib = shared_library('qnn_context',
  qnn_sdk_sources + qnn_context_sources,
  include_directories: [qnn_inc],
  dependencies: [nntrainer_dep, log_dep],
  cpp_args: ['-DENABLE_QNN=1', '-DPLUGGABLE=1',
             '-fvisibility=default'],
  install: true,
)
