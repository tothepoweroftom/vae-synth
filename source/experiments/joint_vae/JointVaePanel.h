#pragma once
#include "JointVae.h"
#include "LabAxisList.h"
#include "LabPad.h"
#include <juce_gui_basics/juce_gui_basics.h>

namespace vae::experimental
{
// VAE Lab v2: the student's XY-pad page (vae/pitcher/xypad/page_v2.template.html) as a native screen.
// Every move decodes on the worker; results play on Lite through onLite, Send to Surge plays onPatch.
class JointVaePanel final : public juce::Component
{
public:
    JointVaePanel();
    void resized() override;
    void paint (juce::Graphics&) override;
    /** Plays a decoded point on the Lite engine (set by the editor; message thread). */
    std::function<void (const schema::Snapshot&)> onLite;
    /** Plays a decoded full Surge patch (set by the editor; message thread). */
    std::function<juce::Result (const juce::MemoryBlock&)> onPatch;
    /** The installed bundle loads on construction; tests load another one. */
    void loadBundle (const juce::File&);

private:
#if VAE_WITH_SURGE
    void loadBase (int index);
    void assign (int slot, const juce::String& id);
    void syncPositions();  // page latSyncPos
    void applyPositions(); // page latApplyPos
    void decode();
    void refresh (const LatentAxis* dragged = nullptr);
    void updateDots();
    void setStatus (const juce::String&, bool error = false);
    const LatentAxis* axisOn (int slot) const;

    JointVaeWorker worker;
    std::shared_ptr<const XyPadData> xy;
    JointVaeWorker::Output current;
    bool outputValid = false;
    bool userActed = false; // results play only after the user acts in the Lab (ruling R14)
    bool errorShown = false;
    JointLatent z {};
    std::array<juce::String, 3> axes;
    std::array<float, 3> pos { 0.5f, 0.5f, 0.5f };
    std::optional<std::vector<float>> lockedTail;

    juce::TextButton latentButton { "Latent pad" }, knobButton { "Knob pad" }, lockButton { "Lock structure" },
        resetButton { "Reset" }, dotsButton { "Preset dots" }, sendButton { "Send to Surge" };
    juce::ComboBox base, category;
    LabPad pad;
    juce::Slider zSlider;
    juce::Viewport listView;
    LabAxisList list;
    juce::Label status;
    juce::Rectangle<int> yLabelArea, xLabelArea, zLabelArea, promptArea, readoutArea;
#else
    juce::Label note;
#endif
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (JointVaePanel)
};
} // namespace vae::experimental
