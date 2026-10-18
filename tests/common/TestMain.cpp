// Shared juce::UnitTest runner: runs the category given by VAE_TEST_CATEGORY.
#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>

#include <iostream>

int main()
{
    const juce::ScopedJuceInitialiser_GUI juceInit;

    juce::UnitTestRunner runner;
    runner.setAssertOnFailure (false);
    runner.runTestsInCategory (VAE_TEST_CATEGORY);

    int failures = 0, passes = 0;
    for (int i = 0; i < runner.getNumResults(); ++i)
        if (const auto* r = runner.getResult (i))
        {
            failures += r->failures;
            passes += r->passes;
        }

    std::cout << VAE_TEST_CATEGORY << ": " << passes << " checks passed, " << failures << " failed" << std::endl;
    return failures == 0 ? 0 : 1;
}
