juce_add_console_app(vae_ml_tests PRODUCT_NAME "vae_ml_tests" VERSION 0.1.0)

target_sources(vae_ml_tests PRIVATE
    MlTests.cpp
    JointVaeTests.cpp
    XyPadTests.cpp
    LatentOrbitTests.cpp
    PromptWordsTests.cpp
    LiteTranslationTests.cpp
    PromptModelTests.cpp
    ${CMAKE_SOURCE_DIR}/source/experiments/joint_vae/PromptModel.cpp
    ${CMAKE_SOURCE_DIR}/source/experiments/joint_vae/LiteTranslation.cpp
    ${CMAKE_SOURCE_DIR}/source/experiments/joint_vae/PromptWords.cpp
    ${CMAKE_SOURCE_DIR}/source/experiments/joint_vae/XyPad.cpp
    ${CMAKE_SOURCE_DIR}/source/experiments/joint_vae/LatentOrbit.cpp
    ${CMAKE_SOURCE_DIR}/source/experiments/joint_vae/JointVae.cpp
    ${CMAKE_SOURCE_DIR}/source/experiments/joint_vae/StudentCodec.cpp
    ../common/TestMain.cpp)
if(VAE_WITH_SURGE)
    target_sources(vae_ml_tests PRIVATE ${CMAKE_SOURCE_DIR}/source/experiments/joint_vae/SurgePatchBuilder.cpp)
    target_link_libraries(vae_ml_tests PRIVATE vae::surge)
endif()

target_compile_definitions(vae_ml_tests PRIVATE
    VAE_TEST_CATEGORY="vae_ml"
    VAE_STUB_DECODER_PATH="${CMAKE_SOURCE_DIR}/models/stub_decoder.onnx"
    VAE_ML_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
    JUCE_MODAL_LOOPS_PERMITTED=1
    JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)

target_link_libraries(vae_ml_tests PRIVATE
    vae_ml
    vae_tokenizer
    juce::juce_events
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)

vae_copy_onnxruntime_next_to(vae_ml_tests)
add_test(NAME vae_ml_tests COMMAND vae_ml_tests)
# Keep tests away from the user's installed models and caches.
set_tests_properties(vae_ml_tests PROPERTIES ENVIRONMENT "VAE_DATA_DIR=${CMAKE_CURRENT_BINARY_DIR}/appdata")
