cmake_minimum_required(VERSION 3.22)

# Must be set before project(): project() creates this cache entry itself.
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")

project(VAESynth VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(VAE_WITH_SURGE "Build the Surge XT engine backend (makes the binary GPL-3.0-or-later)" OFF)
option(VAE_BUILD_TESTS "Build unit tests" ON)
set(VAE_PLUGIN_FORMATS "VST3;Standalone" CACHE STRING "JUCE plug-in formats (AU is added on macOS)")

# --- dependencies ------------------------------------------------------------
include(VaeJuce)
include(VaeOnnxRuntime)
include(VaeSchema)
if(VAE_WITH_SURGE)
    include(VaeSurge)
endif()

# --- schema ------------------------------------------------------------------
set(VAE_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
vae_generate_schema("${CMAKE_CURRENT_SOURCE_DIR}/schema/controls.json"
                    "${VAE_GENERATED_DIR}/schema/ControlSchema.generated.h")

# Project code (except the vae_params static library) is compiled into each
# final target (plug-in, tests) through INTERFACE sources, the same way JUCE
# modules are, so JUCE code is compiled exactly once per binary.
add_library(vae_schema INTERFACE)
target_include_directories(vae_schema INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/source"
    "${VAE_GENERATED_DIR}/schema")

# --- engines -----------------------------------------------------------------
add_library(vae_engine INTERFACE)
target_sources(vae_engine INTERFACE
    source/engine/EngineFactory.cpp
    source/engine/lite/LiteEngine.cpp
    source/engine/lite/LitePatch.cpp
    source/engine/lite/dsp/Wavetables.cpp)
target_link_libraries(vae_engine INTERFACE vae_schema juce::juce_audio_basics juce::juce_dsp)
if(VAE_WITH_SURGE)
    target_sources(vae_engine INTERFACE source/engine/surge/SurgeEngine.cpp)
    target_link_libraries(vae_engine INTERFACE vae::surge)
endif()

# --- ML (ONNX Runtime, loaded at runtime) -------------------------------------
add_library(vae_ml INTERFACE)
target_sources(vae_ml INTERFACE
    source/ml/OrtRuntime.cpp
    source/ml/OnnxModel.cpp
    source/ml/PresetGenerator.cpp)
target_link_libraries(vae_ml INTERFACE vae_schema vae::onnxruntime_headers juce::juce_events)

include(cmake/NativeTokenizer.cmake)
add_library(vae_retrieval INTERFACE)
target_sources(vae_retrieval INTERFACE source/ml/SameSearch.cpp source/ml/StableAudioGenerator.cpp)
target_link_libraries(vae_retrieval INTERFACE vae_tokenizer vae_ml juce::juce_audio_formats juce::juce_cryptography)

# --- isolated joint VAE experiment ---------------------------------------------
# The Lab's look: the design study's reusable controls and orbit view (no study data or model code).
add_subdirectory(design-study/juce/Source/ui pitcher-ui)
add_library(vae_joint_experiment INTERFACE)
target_sources(vae_joint_experiment INTERFACE
    source/experiments/joint_vae/JointVae.cpp
    source/experiments/joint_vae/JointVaePanel.cpp
    source/experiments/joint_vae/LabPad.cpp
    source/experiments/joint_vae/LabOrbit.cpp
    source/experiments/joint_vae/LabAxisList.cpp
    source/experiments/joint_vae/StudentCodec.cpp
    source/experiments/joint_vae/LiteTranslation.cpp
    source/experiments/joint_vae/XyPad.cpp
    source/experiments/joint_vae/LatentOrbit.cpp
    source/experiments/joint_vae/PromptWords.cpp
    source/experiments/joint_vae/PromptModel.cpp)
target_link_libraries(vae_joint_experiment INTERFACE vae_ml vae_tokenizer juce::juce_gui_basics pitcher_study_ui)
if(VAE_WITH_SURGE)
    target_sources(vae_joint_experiment INTERFACE source/experiments/joint_vae/SurgePatchBuilder.cpp)
    target_link_libraries(vae_joint_experiment INTERFACE vae::surge)
endif()

# --- parameter store (clean-room, docs/specs/parameter-manager.md) ------------
add_subdirectory(source/params)

# --- plug-in -------------------------------------------------------------------
set(VAE_FORMATS ${VAE_PLUGIN_FORMATS})
if(APPLE)
    list(APPEND VAE_FORMATS AU)
endif()

juce_add_plugin(VAESynth
    COMPANY_NAME "VAEProject"
    BUNDLE_ID "com.vaeproject.vaesynth"
    PLUGIN_MANUFACTURER_CODE Vaep
    PLUGIN_CODE Vaes
    FORMATS ${VAE_FORMATS}
    PRODUCT_NAME "VAE Synth"
    IS_SYNTH TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE
    COPY_PLUGIN_AFTER_BUILD FALSE
    VST3_CATEGORIES Instrument Synth
    AU_MAIN_TYPE kAudioUnitType_MusicDevice)

juce_add_binary_data(VAESynthAssets
    NAMESPACE BinaryData
    SOURCES
        assets/fonts/SpaceMono-Regular.ttf
        assets/fonts/SpaceMono-Bold.ttf
        assets/fonts/Inter-Regular.ttf
        assets/fonts/Inter-Bold.ttf
        models/stub_decoder.onnx)

target_sources(VAESynth PRIVATE
    source/plugin/PluginProcessor.cpp
    source/plugin/PluginEditor.cpp
    source/plugin/ui/VaeLookAndFeel.cpp)

target_compile_definitions(VAESynth PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)

target_link_libraries(VAESynth
    PRIVATE
        VAESynthAssets
        vae_params
        vae_engine
        vae_retrieval
        vae_joint_experiment
        juce::juce_audio_utils
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)

vae_bundle_onnxruntime(VAESynth)

# Packaging reads the configured build, including local dependency overrides.
configure_file(cmake/release-info.json.in release-info.json @ONLY)

# --- tools ---------------------------------------------------------------------
add_subdirectory(tools/retrieve)
if(VAE_WITH_SURGE)
    add_subdirectory(tools/compare)
    add_subdirectory(tools/p0_eval)
endif()

# --- tests ---------------------------------------------------------------------
if(VAE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/params)
    add_subdirectory(tests/engine)
    add_subdirectory(tests/ml)
    add_subdirectory(tests/plugin)

    find_program(VAE_CLANG_FORMAT NAMES clang-format)
    if(VAE_CLANG_FORMAT)
        add_test(NAME vae_format_check
                 COMMAND ${CMAKE_COMMAND} -D CLANG_FORMAT=${VAE_CLANG_FORMAT} -D ROOT=${CMAKE_CURRENT_SOURCE_DIR}
                         -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VaeFormatCheck.cmake")
    endif()

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
        add_test(NAME vae_retrieve_smoke
                 COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_retrieve.py"
                         $<TARGET_FILE:vae_retrieve>)
        set(_vae_py_args "${VAE_GENERATED_DIR}/schema/ControlSchema.generated.h")
        if(VAE_WITH_SURGE)
            list(APPEND _vae_py_args "${VAE_SURGE_DIR}/resources/data/patches_factory/Templates")
        endif()
        add_test(NAME vae_schema_tools
                 COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_schema_tools.py" ${_vae_py_args})
        if(VAE_WITH_SURGE)
            add_test(NAME vae_compare_smoke
                     COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/test_compare.py"
                             $<TARGET_FILE:vae_compare> "${VAE_SURGE_DIR}/resources/data/patches_factory/Templates")
        endif()
    endif()
endif()
