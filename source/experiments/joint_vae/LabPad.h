#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include <tuple>
#include <vector>

namespace vae::experimental
{
/** The page's dark-theme category colours --c0..--c8 (jointLabels order). */
inline const juce::Colour labCategoryColours[9] = { juce::Colour (0xff3987e5), juce::Colour (0xffd95926),
                                                    juce::Colour (0xff199e70), juce::Colour (0xffc98500),
                                                    juce::Colour (0xffd55181), juce::Colour (0xff008300),
                                                    juce::Colour (0xff9085e9), juce::Colour (0xffe66767),
                                                    juce::Colour (0xff8a919c) };

/** The pad's well, behind the map and inside the dot. */
inline const juce::Colour labWell { 0xff000000 };
/** Measure map and spectrogram colour (VaeLookAndFeel::accent), and the category map's alpha (page --map-alpha). */
inline const juce::Colour labHeat { 0xffffff07 };
inline constexpr float labMapAlpha = 0.36f;

/** The Lab's XY pad: positions are 0..1 with y up. Arrow keys move the dot by 0.02 (Shift: 0.1). */
class LabPad final : public juce::Component
{
public:
    LabPad();
    std::function<void (juce::Point<float>)> moved;  // drag, click or arrow key
    std::function<void (juce::Point<float>)> tapped; // a click that did not drag
    void setDot (juce::Point<float>);
    /** `labels`: category names at their region centroids (0..1, y up). */
    void setMap (juce::Image, std::vector<std::pair<juce::String, juce::Point<float>>> labels = {});
    void setPresetDots (std::vector<std::tuple<float, float, int>>, int activeLabel);
    /** Find's candidates (0..1, y up), numbered 1.. above the preset dots; `selected` is drawn filled. */
    void setHighlights (std::vector<juce::Point<float>>, int selected);
    const std::vector<juce::Point<float>>& getHighlights() const { return highlights; }
    std::function<void (int)> picked; // a click on highlight i that did not drag
    juce::Point<float> getDot() const { return dot; }
    const juce::Image& getMap() const { return map; }

    void paint (juce::Graphics&) override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseDrag (const juce::MouseEvent&) override;
    void mouseUp (const juce::MouseEvent&) override;
    bool keyPressed (const juce::KeyPress&) override;
    void focusGained (FocusChangeType) override { repaint(); }
    void focusLost (FocusChangeType) override { repaint(); }

private:
    juce::Point<float> toPad (juce::Point<float> pixel) const;
    int highlightAt (juce::Point<float> pixel) const; // nearest highlight within 7 px, else -1
    void move (juce::Point<float>);
    juce::Point<float> dot { 0.5f, 0.5f };
    juce::Image map;
    std::vector<std::pair<juce::String, juce::Point<float>>> mapLabels;
    std::vector<std::tuple<float, float, int>> dots;
    int activeLabel = -1;
    std::vector<juce::Point<float>> highlights;
    int selectedHighlight = -1;
    int downOn = -1; // the highlight a press started on: it picks on release instead of moving
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (LabPad)
};
} // namespace vae::experimental
