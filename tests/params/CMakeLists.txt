# vae_params_tests: juce::UnitTest suite for the parameter store (spec section 9).

juce_add_console_app(vae_params_tests
    PRODUCT_NAME "vae_params_tests"
    VERSION 0.1.0)

target_sources(vae_params_tests PRIVATE
    ParamStoreTests.cpp
    TestMain.cpp)

target_link_libraries(vae_params_tests
    PRIVATE
        vae_params
        juce::juce_core
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)

# The attachment tests pump the message loop (MessageManager::runDispatchLoopUntil).
target_compile_definitions(vae_params_tests PRIVATE JUCE_MODAL_LOOPS_PERMITTED=1)

add_test(NAME vae_params_tests COMMAND vae_params_tests)
