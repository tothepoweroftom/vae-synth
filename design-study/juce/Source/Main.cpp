#include "StudyView.h"
#include "StudyChecks.h"
#include <iostream>
class StudyApplication final : public juce::JUCEApplication
{
public:
    const juce::String getApplicationName() override { return "Pitcher Design Study"; }
    const juce::String getApplicationVersion() override { return "0.1.0"; }
    void initialise(const juce::String& args) override {
        if(args.contains("--checks")) {juce::String report;bool ok=runStudyChecks(report);std::cout<<report<<std::endl;setApplicationReturnValue(ok?0:1);quit();return;}
        window = std::make_unique<Window>();
        if(args.contains("--snapshot"))juce::Timer::callAfterDelay(800,[this]{
            auto image=window->getContentComponent()->createComponentSnapshot(window->getContentComponent()->getLocalBounds());
            juce::File output("/tmp/pitcher-native.png");output.deleteFile();juce::FileOutputStream stream(output);juce::PNGImageFormat().writeImageToStream(image,stream);quit();
        });
    }
    void shutdown() override { window.reset(); }
    class Window final : public juce::DocumentWindow
    {
    public:
        Window() : DocumentWindow("Pitcher - native design study", juce::Colour(0xff252525), allButtons)
        {
            setUsingNativeTitleBar(true);
            setContentOwned(new StudyView(), true);
            setResizable(true, false); setResizeLimits(1000, 720, 2200, 1400);
            centreWithSize(1380, 800); setVisible(true);
        }
        void closeButtonPressed() override { juce::JUCEApplication::getInstance()->systemRequestedQuit(); }
    };
    std::unique_ptr<Window> window;
};
START_JUCE_APPLICATION(StudyApplication)
