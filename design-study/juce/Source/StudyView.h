#pragma once
#include <juce_gui_extra/juce_gui_extra.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include "StudyData.h"
#include <optional>
#include "StudyOrbit.h"
#include "StudyAxis.h"
#include "ui/PitcherControls.h"

class StudyView final : public juce::Component, private juce::Timer
{
public:
    StudyView();
    ~StudyView() override;
    void resized() override;
    void paint(juce::Graphics&) override;
private:
    friend bool runStudyChecks(juce::String&);
    void timerCallback() override;
    void remember();

    void refreshPosition();
    void refreshCloud();
    void refreshRows();
    void selectTab(bool presets);
    void saveNamed();
    void rebuildFavourites();
    void showTune();
    struct Favourite {juce::String name; std::array<float,32> latent; std::array<int,3> axes; int category;};
    void restore(const Favourite&);
    std::vector<Favourite> history;
    std::optional<Favourite> pinned;
    juce::TextButton back {"Back"}, pin {"Pin A"}, compare {"Swap A/B"}, motion {"Motion: on"};
    bool reducedMotion=false;
    std::array<float,8> displayedMatches {};
    double lastVisual=0, lastInference=0;
    PitcherLookAndFeel look;
    StudyData data;
    std::array<float,32> latent {};
    std::array<int,3> assigned {0,2,5};
    int category=0, highlighted=-1;
    bool dirty=true, presetsVisible=false;
    StudyOrbit orbit;
    Spectrum spectrum;
    juce::TooltipWindow tooltip {this, 350};
    std::array<juce::TextButton,8> categories;
    std::array<float,8> matches {};
    juce::TextButton save {"Save"}, axesTab {"AXES"}, presetsTab {"PRESETS"};
    juce::TextButton zoomOut {"-"}, zoomIn {"+"}, reset {"Reset view"};
    juce::TextEditor prompt;
    juce::TextButton send {">"}, tune {"Shape my space"};
    juce::Label status;
    juce::Viewport axisScroll, presetScroll;
    juce::Component axisContent, presetContent;
    std::vector<std::unique_ptr<StudyAxisRow>> rows;
    std::vector<Favourite> favourites;
    std::vector<std::unique_ptr<juce::TextButton>> favouriteButtons;
    juce::MidiKeyboardState keyboardState;
    PitcherKeyboard keyboard {keyboardState};
    juce::Rectangle<int> left, centre, right;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(StudyView)
};
