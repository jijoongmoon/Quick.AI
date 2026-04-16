# QuickDotAI — QNN (NPU) 사용 가이드

QuickDotAI 의 `NativeQuickDotAI` 구현을 통해 QNN (Qualcomm Neural Networks) 기반 on-device LLM 을 Android 앱에서 실행하는 방법을 정리합니다.

## 지원 범위

- **CPU / 디바이스**: Qualcomm Snapdragon (HTP/DSP 가 있는 SoC — 8 Gen 1 ~ 8 Gen 3 급)
- **ABI**: `arm64-v8a` 전용
- **모델**: `ModelId.LLM_QNN_TEST` (QNN 백엔드로 사전 컴파일된 serialized context binary)
- **지원 안 되는 것**:
  - `chatRunStreaming` 등 structured-chat API (native 백엔드는 아직 단일 prompt 만 지원. multi-turn 은 `LiteRTLm` 을 써야 함)
  - `runMultimodal*` (이미지 입력)
  - armv7 / x86

## 사전 준비

### 1) Qualcomm QNN SDK

Qualcomm 웹사이트 로그인 후 QAIRT (QNN SDK) 다운로드. 이하 설치 경로를 `$QNN_SDK_ROOT` 로 표기:

```
$QNN_SDK_ROOT/
└── lib/aarch64-android/
    ├── libQnnHtp.so
    ├── libQnnSystem.so
    ├── libQnnHtpPrepare.so
    ├── libQnnHtpNetRunExtensions.so
    ├── libQnnHtpV73Skel.so       # 8 Gen 1 ~ 2
    ├── libQnnHtpV75Skel.so       # 8 Gen 3 (SM-S948U 등)
    └── ...
```

### 2) Android NDK + CMake

- NDK: r26 이상 (r28b 검증됨)
- CMake: 3.22.1 이상
- Android SDK `platform-tools`, `build-tools;36.0.0`, `platforms;android-36`

```bash
export ANDROID_HOME=$HOME/Android/Sdk
export ANDROID_NDK=$ANDROID_HOME/ndk/<버전>
export PATH=$ANDROID_HOME/platform-tools:$PATH
```

### 3) QNN 모델 파일

- 사전 준비한 serialized QNN context binary (예: `e4h4w4_..._qnn.serialized.bin`)
- 모델에 맞는 `config.json`, `generation_config.json`, `nntr_config.json`, `tokenizer.json`, `tokenizer_config.json`

## 빌드

### Native 라이브러리 (libnntrainer.so / libqnn_context.so / libquick_dot_ai_api.so)

Quick.AI 루트에서:

```bash
cd <Quick.AI repo root>
./build.sh --platform=android --enable-qnn --clean
```

산출물:
```
nntrainer/builddir/android_build_result/lib/arm64-v8a/
  libnntrainer.so
  libccapi-nntrainer.so

builddir_android/
  api/libquick_dot_ai_api.so
  qnn/libqnn_context.so
```

### AAR 용 `prebuilt_libs/` 채우기

AAR 빌드 시 `QuickDotAI/prebuilt_libs/` 안의 `.so` 들이 `jniLibs/arm64-v8a/` 로 복사됩니다. 다음 목록이 모두 있어야 합니다:

```bash
PB=nntrainer/Applications/QuickAI/QuickDotAI/prebuilt_libs

# nntrainer
cp -v nntrainer/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so        $PB/
cp -v nntrainer/builddir/android_build_result/lib/arm64-v8a/libccapi-nntrainer.so  $PB/

# QuickDotAI API + QNN plugin
cp -v builddir_android/api/libquick_dot_ai_api.so  $PB/
cp -v builddir_android/qnn/libqnn_context.so       $PB/

# Qualcomm QNN 런타임
SRC=$QNN_SDK_ROOT/lib/aarch64-android
cp -v $SRC/libQnnHtp.so                         $PB/
cp -v $SRC/libQnnSystem.so                      $PB/
cp -v $SRC/libQnnHtpPrepare.so                  $PB/
cp -v $SRC/libQnnHtpNetRunExtensions.so         $PB/
cp -v $SRC/libQnnHtpV75Skel.so                  $PB/   # 디바이스 DSP 버전에 맞춰
```

최종 `prebuilt_libs/` 기대 구성:
```
libc++_shared.so
libnntrainer.so
libccapi-nntrainer.so
libquick_dot_ai_api.so
libqnn_context.so
libQnnHtp.so
libQnnSystem.so
libQnnHtpPrepare.so
libQnnHtpNetRunExtensions.so
libQnnHtpV75Skel.so
```

### AAR 빌드

```bash
cd nntrainer/Applications/QuickAI
./gradlew :QuickDotAI:assembleRelease
# ⇒ QuickDotAI/build/outputs/aar/QuickDotAI-release.aar
```

## 디바이스에 HTP 설정 파일 배치

QNN HTP backend extension 은 앱의 **현재 작업 디렉터리** 에서 설정 JSON 을 찾습니다. `NativeQuickDotAI.load()` 가 호출될 때 작업 디렉터리는:

```
/sdcard/Android/data/<package>/files
```

로 `chdir` 됩니다. 그 경로에 다음 파일을 넣어주세요:

```bash
BASE=/sdcard/Android/data/com.example.yourapp/files

adb push htp_backend_ext_config.json $BASE/
adb push htp_config.json             $BASE/
```

### `htp_backend_ext_config.json` 예시

```json
{
  "backend_extensions": {
    "shared_library_path": "libQnnHtpNetRunExtensions.so",
    "config_file_path": "htp_config.json"
  },
  "context_configs": {
    "weight_sharing_enabled": false
  },
  "graph_configs": [
    {
      "graph_names": [""],
      "vtcm_mb": 8,
      "O": 3,
      "fp16_relaxed_precision": 1,
      "dlbc": 1
    }
  ]
}
```

### `htp_config.json` 예시

```json
{
  "devices": [
    { "dsp_arch": "v75", "soc_id": 57 }
  ]
}
```

`dsp_arch` 와 `soc_id` 는 타겟 디바이스에 맞춰 조정. QNN SDK 의 `examples/QNN/NetRun/config/` 에 전체 템플릿이 있습니다.

## 모델 파일 배치

```
/sdcard/Android/data/<package>/files/models/llm_qnn_test/
  ├── config.json
  ├── generation_config.json
  ├── nntr_config.json
  ├── tokenizer.json
  ├── tokenizer_config.json
  └── <weight>.serialized.bin
```

`config.json` 의 `architectures` 필드는 `["LLM_QNN_TEST"]` 여야 native Factory 가 해당 모델을 생성합니다. `nntr_config.json` 의 `model_file_name` 은 weight 파일의 절대경로 또는 `./models/llm_qnn_test/<weight>.serialized.bin` 같은 상대경로.

## Kotlin 사용 예시

```kotlin
import com.example.quickdotai.*

class MyActivity : AppCompatActivity() {

    private lateinit var engine: QuickDotAI

    private fun loadModel() {
        engine = NativeQuickDotAI()

        val req = LoadModelRequest(
            backend      = BackendType.NPU,
            model        = ModelId.LLM_QNN_TEST,
            quantization = QuantizationType.W4A32,
            nativeLibDir = applicationContext.applicationInfo.nativeLibraryDir,
            // modelPath 는 생략 시 기본값 "./models/llm_qnn_test" 사용.
            // 명시할 경우 절대경로를 주고, 그 grandparent 가 cwd 로 chdir 됨.
        )

        when (val r = engine.load(req)) {
            is BackendResult.Ok  -> Log.i(TAG, "Model loaded")
            is BackendResult.Err -> Log.e(TAG, "Load failed: ${r.error} ${r.message}")
        }
    }

    private fun runPrompt(prompt: String) {
        engine.runStreaming(prompt, object : StreamSink {
            override fun onDelta(text: String) {
                // UI 업데이트는 main thread 로
                runOnUiThread { textView.append(text) }
            }
            override fun onDone() = runOnUiThread { /* 완료 처리 */ }
            override fun onError(err: QuickAiError, msg: String?) {
                Log.e(TAG, "[$err] $msg")
            }
        })
    }

    override fun onDestroy() {
        super.onDestroy()
        engine.close()
    }
}
```

### Thread 모델

- `NativeQuickDotAI` 는 thread-safe 하지 않습니다. 단일 worker thread 에서만 호출하세요.
- `StreamSink.onDelta` 는 worker thread 에서 동기 호출됩니다. UI 업데이트는 반드시 `runOnUiThread` / `Handler` 로.

## 내부 구조 (문제 발생 시 참고)

```
Kotlin (UI / applicationContext)
    │
    └── NativeQuickDotAI.load() ────► NativeCausalLm.loadModelHandleNative
                                          │
                                          ▼    (JNI — libquickai_jni.so)
                                      Java_..._loadModelHandleNative
                                          │
                                          ▼    (C API — libquick_dot_ai_api.so)
                                      loadModelHandle
                                          │
                                          ▼
                                      Factory::create("LLM_QNN_TEST")
                                          │
                                          ▼
                                      Quick_Dot_AI_QNN::initialize(nativeLibDir)
                                          │
                                          ▼    (libnntrainer.so)
                                      Engine::registerContext("libqnn_context.so",
                                                              nativeLibDir)
                                          │
                                          ▼    (dlopen)
                                      libqnn_context.so static init
                                          │
                                          ▼    (QNN SDK — libQnnHtp.so 등)
                                      QNNContext::init()
                                          │
                                          ▼
                                      BackendExtensions(htp_backend_ext_config.json)
                                          │
                                          ▼
                                      registerFactory(qnn_linear, weight, tensor, qnn_graph)
                                          │
                                          ▼
                                      createLayer("qnn_graph")
                                          │
                                          ▼
                                      Graph retrieve / Model load
```

## 트러블슈팅

### 1) `UnsatisfiedLinkError: dlopen failed: "libqnn_context.so" not found`

→ `prebuilt_libs/` 에 `libqnn_context.so` 가 없거나 AAR 에 번들되지 않았음. 위 "prebuilt_libs 채우기" 단계 확인.

### 2) `loadModelHandle failed (errorCode=1) INVALID_PARAMETER`

→ `ModelId` enum 과 C 쪽 `ModelType` enum 의 ordinal 이 맞지 않음. 현재 정확한 값:

| Kotlin `ModelId` | 순서 | C 쪽 `ModelType` |
|---|---|---|
| `QWEN3_0_6B` | 0 | `CAUSAL_LM_MODEL_QWEN3_0_6B = 0` |
| `GEMMA4` | — | (Kotlin-only; LiteRTLm 로 routing) |
| `LLM_QNN_TEST` | **1** | `CAUSAL_LM_MODEL_LLM_QNN_TEST = 1` |

`NativeQuickDotAI.mapModelId()` 를 확인.

### 3) `loadModelHandle failed (errorCode=2) MODEL_LOAD_FAILED` — std::bad_cast

→ Android classloader linker namespace 이슈. `libqnn_context.so` 의 DSO-local 심볼이 `libnntrainer.so` 의 export 와 collapse 되지 않음. 다음이 반영됐는지 확인:

- `NativeCausalLm.ensureLoaded()` 가 `System.loadLibrary("qnn_context")` 를 `System.loadLibrary("quickai_jni")` 보다 먼저 호출
- `Engine::Global()` 이 header-only template 이 아니라 `libnntrainer.so` 에 strong symbol 로 out-of-line 정의됨

### 4) `registering qnn layers failed!!, reason: Unable to load backend extensions config`

→ `htp_backend_ext_config.json` 이 `/sdcard/Android/data/<package>/files/` 에 없음. 위 "디바이스에 HTP 설정 파일 배치" 참고.

리팩토링 이후엔 이 에러가 나도 layer 등록은 성공하고 `createLayer("qnn_graph")` 경로까지 진행되지만, 실제 graph forwarding 은 여전히 실패합니다.

### 5) `NativeChatSession... UNSUPPORTED`

→ native 백엔드는 structured chat 을 지원하지 않음. `engine.run(prompt)` 또는 `engine.runStreaming(prompt, sink)` 로 단일 prompt 만 사용. 대화 형식이 필요하면 Gemma 모델 + `LiteRTLm` 백엔드로 바꾸거나, caller 측에서 `messages` 를 직접 single prompt 문자열로 직렬화.

### 6) 로그 필터

진단 시 유용한 tag 조합:

```bash
adb logcat -c
adb logcat \
  'NativeQuickDotAI:V' 'NativeCausalLm:V' 'QuickAI:V' \
  'quickai_jni:V' 'Quick_Dot_AI_QNN:V' 'QNNContext:V' \
  'nntrainer:V' 'nntrainer_engine:V' \
  'AndroidRuntime:E' '*:S'
```

native 크래시 (SIGSEGV / SIGABRT) 도 보고 싶으면 `'DEBUG:V' 'libc:V' 'libc++abi:V'` 를 추가.

## 라이선스

Apache-2.0. QNN SDK 런타임 (`libQnn*.so`) 은 Qualcomm 라이선스 하에 배포됨 — 앱 배포 시 Qualcomm 과의 라이선스 계약 별도 확인 필요.
