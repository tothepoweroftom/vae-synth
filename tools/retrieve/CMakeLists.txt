juce_add_console_app(vae_retrieve PRODUCT_NAME "vae_retrieve" VERSION 0.1.0)
target_sources(vae_retrieve PRIVATE RetrieveMain.cpp ../compare/Features.cpp)
target_compile_definitions(vae_retrieve PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
target_link_libraries(vae_retrieve PRIVATE vae_engine juce::juce_audio_formats juce::juce_dsp
    juce::juce_recommended_config_flags juce::juce_recommended_warning_flags)

# Optional native feasibility probe; runs the same OnnxModel wrapper as the plug-in.
juce_add_console_app(vae_same_onnx_probe PRODUCT_NAME "vae_same_onnx_probe" VERSION 0.1.0)
target_sources(vae_same_onnx_probe PRIVATE SameOnnxProbe.cpp)
target_compile_definitions(vae_same_onnx_probe PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
target_link_libraries(vae_same_onnx_probe PRIVATE vae_ml juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)
vae_copy_onnxruntime_next_to(vae_same_onnx_probe)

juce_add_console_app(vae_same_search PRODUCT_NAME "vae_same_search" VERSION 0.1.0)
target_sources(vae_same_search PRIVATE NativeSearchMain.cpp)
target_compile_definitions(vae_same_search PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
target_link_libraries(vae_same_search PRIVATE vae_retrieval juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)
vae_copy_onnxruntime_next_to(vae_same_search)

juce_add_console_app(vae_generate PRODUCT_NAME "vae_generate" VERSION 0.1.0)
target_sources(vae_generate PRIVATE NativeGenerateMain.cpp)
target_compile_definitions(vae_generate PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
target_link_libraries(vae_generate PRIVATE vae_retrieval juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)
vae_copy_onnxruntime_next_to(vae_generate)
