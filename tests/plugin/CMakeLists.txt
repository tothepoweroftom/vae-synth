juce_add_console_app(vae_plugin_tests PRODUCT_NAME "vae_plugin_tests" VERSION 0.1.0)

target_sources(vae_plugin_tests PRIVATE
    PluginTests.cpp
    ../common/TestMain.cpp
    ${CMAKE_SOURCE_DIR}/source/plugin/PluginProcessor.cpp
    ${CMAKE_SOURCE_DIR}/source/plugin/PluginEditor.cpp
    ${CMAKE_SOURCE_DIR}/source/plugin/ui/VaeLookAndFeel.cpp)

target_include_directories(vae_plugin_tests PRIVATE ${CMAKE_SOURCE_DIR}/source/plugin)

target_compile_definitions(vae_plugin_tests PRIVATE
    VAE_TEST_CATEGORY="vae_plugin"
    JucePlugin_Name="VAE Synth"
    VAE_STUB_DECODER_PATH="${CMAKE_SOURCE_DIR}/models/stub_decoder.onnx"
    VAE_ML_FIXTURES="${CMAKE_SOURCE_DIR}/tests/ml/fixtures"
    JUCE_MODAL_LOOPS_PERMITTED=1
    JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)

target_link_libraries(vae_plugin_tests PRIVATE
    VAESynthAssets
    vae_params
    vae_engine
    vae_retrieval
    juce::juce_audio_utils
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)

vae_copy_onnxruntime_next_to(vae_plugin_tests)
add_test(NAME vae_plugin_tests COMMAND vae_plugin_tests)
# Keep tests away from the user's installed models and caches.
set_tests_properties(vae_plugin_tests PROPERTIES ENVIRONMENT "VAE_DATA_DIR=${CMAKE_CURRENT_BINARY_DIR}/appdata")
