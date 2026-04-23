project('nntrainer-causallm', 'c', 'cpp',
  version: '0.1.0',
  license: ['apache-2.0'],
  meson_version: '>=0.55.0',
  default_options: [
    'warning_level=1',
    'c_std=gnu89',
    'cpp_std=c++17',
    'buildtype=release',
  ],
)

# Pull in nntrainer via the bundled submodule as a meson subproject.
# We disable Applications so that nntrainer's own (now legacy) in-tree
# Applications/CausalLM is not built alongside this standalone one, and
# we skip tests to keep the dependency build lean.
nntrainer_proj = subproject('nntrainer',
  default_options: [
    'enable-app=false',
    'enable-test=false',
    'enable-tflite-backbone=false',
    'enable-tflite-interpreter=false',
    'werror=false',
  ],
)

nntrainer_dep       = nntrainer_proj.get_variable('nntrainer_dep')
nntrainer_ccapi_dep = nntrainer_proj.get_variable('nntrainer_ccapi_dep')
application_install_dir = get_option('prefix') / get_option('libdir') / 'nntrainer-causallm'

openmp_dep = dependency('openmp')
powershell_prog = find_program('powershell', required: (build_machine.system() == 'windows'))

causallm_src = [
    meson.current_source_dir() / 'huggingface_tokenizer.cpp',
    meson.current_source_dir() / 'llm_util.cpp',
    meson.current_source_dir() / 'api' / 'causal_lm_api.cpp',
    meson.current_source_dir() / 'api' / 'model_config.cpp',
]

executable_src = [
    meson.current_source_dir() / 'main.cpp',
]

causallm_inc_abs = [meson.current_source_dir()]
causallm_inc = [include_directories('.')]

# Build layer dependency
causallm_layer_dependencies = []
subdir('layers')

# Add common layers to dependencies
causallm_layer_dependencies += [
    causallm_rms_norm_dep,
    causallm_tie_word_embedding_dep,
    causallm_lm_head_dep,
    causallm_swiglu_dep,
    causallm_mha_core_dep,
    causallm_embedding_layer_dep,
    causallm_lm_head_dep,
    causallm_reshaped_rms_norm_dep,
    causallm_qkv_layer_dep,
    causallm_embedding_pooling_layer_dep,
    causallm_embedding_normalize_layer_dep,
]

subdir('models')

if (get_option('platform') == 'windows') and (build_machine.system() == 'windows')
    run_command(powershell_prog, '-ExecutionPolicy', 'Bypass', '-File', join_paths(meson.current_source_dir(), 'jni', 'prepare_encoder.ps1'), meson.build_root(), '0.2', check: true)
elif get_option('platform') != 'tizen'
    run_command([meson.current_source_dir() / 'jni' / 'prepare_encoder.sh', meson.build_root(), '0.2'], check: true)
endif

tokenizer_dependencies = []
tokenizer_path = meson.current_source_dir() / 'lib' / 'libtokenizers_c.a'

cpp_args = []

if get_option('default_library') == 'shared'
    cpp_args += [ '-DPLUGGABLE' ]
endif

causallm = shared_library('causallm',
    causallm_src,
    dependencies: [
        nntrainer_dep,
        nntrainer_ccapi_dep,
        causallm_layer_dependencies,
        tokenizer_dependencies,
        openmp_dep
    ],
    include_directories: causallm_inc,
    install: true,
    install_dir: application_install_dir,
    link_args: [tokenizer_path],
    cpp_args: cpp_args
)

causallm_dep = declare_dependency(
    link_with: [causallm],
    include_directories: causallm_inc
)

e = executable('nntr_causallm',
    executable_src,
    include_directories: causallm_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, causallm_layer_dependencies, causallm_dep],
)


quantize_src = [
    meson.current_source_dir() / 'quantize.cpp',
]

e_quantize = executable('nntr_quantize',
    quantize_src,
    include_directories: causallm_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, causallm_layer_dependencies, causallm_dep],
)


test_api_src = [
    meson.current_source_dir() / 'api' / 'test_api.cpp',
]

test_api = executable('test_api',
    test_api_src,
    include_directories: causallm_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, causallm_layer_dependencies, causallm_dep],
    build_by_default: true,
    install: false,
)
