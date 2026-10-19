#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>

int main()
{
    // Message manager for the APVTS timer and the attachments' async updates.
    const juce::ScopedJuceInitialiser_GUI juceInit;

    juce::UnitTestRunner runner;
    runner.setAssertOnFailure (false);
    runner.runTestsInCategory ("vae_params");

    int failures = 0;
    int passes = 0;

    for (int i = 0; i < runner.getNumResults(); ++i)
    {
        if (const auto* result = runner.getResult (i))
        {
            failures += result->failures;
            passes += result->passes;
        }
    }

    std::cout << "vae_params_tests: " << passes << " checks passed, " << failures << " failed" << std::endl;
    return failures == 0 ? 0 : 1;
}
