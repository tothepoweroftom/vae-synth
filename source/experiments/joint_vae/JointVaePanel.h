#pragma once
#include "JointVae.h"
#include <juce_gui_basics/juce_gui_basics.h>

namespace vae::experimental
{
// Experimental latent explorer. Decoded points play as full Surge patches through onPatch.
class JointVaePanel final : public juce::Component
{
public:
    JointVaePanel();
    void resized() override;
    /** Plays a decoded full Surge patch (set by the editor; message thread). */
    std::function<juce::Result (const juce::MemoryBlock&)> onPatch;

private:
    class Pad final : public juce::Component
    {
    public:
        juce::Point<float> position {};
        std::function<void()> released, moved;
        void paint (juce::Graphics&) override;
        void mouseDown (const juce::MouseEvent& e) override { mouseDrag (e); }
        void mouseDrag (const juce::MouseEvent&) override;
        void mouseUp (const juce::MouseEvent&) override
        {
            if (released)
                released();
        }
    } pad;
    void chooseBundle();
    void importPoint();
    void exportPoint();
    void decode();
    void showOutput (JointVaeWorker::Output);
    void enableActions (bool);
    void setStatus (juce::String);
    void centre();

    JointVaeWorker worker;
    JointLatent anchor {};
    JointVaeWorker::Output current;
    bool loaded = false, outputValid = false;
    juce::TextButton loadButton { "Load VAE bundle..." }, importButton { "Import sound / latent..." },
        randomButton { "Random anchor" }, resetButton { "Centre" }, exportButton { "Export decoded point..." };
    juce::ComboBox category, xAxis, yAxis;
    juce::Slider radius;
    juce::Label title, description, status, axisLabel;
    juce::TextEditor values;
    std::unique_ptr<juce::FileChooser> chooser;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (JointVaePanel)
};
} // namespace vae::experimental
