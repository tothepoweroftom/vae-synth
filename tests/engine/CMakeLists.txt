juce_add_console_app(vae_engine_tests PRODUCT_NAME "vae_engine_tests" VERSION 0.1.0)

target_sources(vae_engine_tests PRIVATE
    EngineTests.cpp
    ../common/TestMain.cpp)

target_compile_definitions(vae_engine_tests PRIVATE
    VAE_TEST_CATEGORY="vae_engine"
    JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)

target_link_libraries(vae_engine_tests PRIVATE
    vae_engine
    juce::juce_events
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags)

add_test(NAME vae_engine_tests COMMAND vae_engine_tests)
