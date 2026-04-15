# api/meson.build — builds libquick_dot_ai_api.so

if not get_option('enable-api')
  subdir_done()
endif

api_inc = include_directories('.')

# Extension model headers

quick_dot_ai_inc_args = [
  '-I' + meson.current_source_dir() / '..' / 'src' / 'models' / 'llm-qnn-test',
]

# Link the extension model implementations (quick_dot_ai_qnn.cpp et al.)
# STATICALLY into libquick_dot_ai_api.so via quick_dot_ai_static instead of
# going through the quick_dot_ai_dep shared-library dependency. On Android
# the plugin flavour (libquick_dot_ai.so) is not packaged into the AAR's
# jniLibs, so a NEEDED dependency on it would leave the JNI consumer
# unable to resolve symbols like Quick_Dot_AI_QNN::initialize at runtime
# — the consumer ends up calling into a stale copy baked into older
# binaries in prebuilt_libs/ because the plugin .so is never updated from
# the AAR copy step. Static linking bakes the current sources into
# libquick_dot_ai_api.so itself, which IS copied to prebuilt_libs/.
quick_dot_ai_api_lib = shared_library('quick_dot_ai_api',
    'quick_dot_ai_api.cpp',
    'model_config.cpp',
    'streamer.cpp',
    'callback_streamer.cpp',
    include_directories: [api_inc, ext_inc, quick_dot_ai_inc],
    link_whole: [quick_dot_ai_static],
    dependencies: [nntrainer_dep, causallm_dep, openmp_dep, thread_dep, log_dep],
    cpp_args: quick_dot_ai_inc_args,
    install: true,
)

quick_dot_ai_api_dep = declare_dependency(
    link_with: quick_dot_ai_api_lib,
    include_directories: api_inc,
)
