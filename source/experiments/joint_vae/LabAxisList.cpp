#include "LabAxisList.h"
#include "plugin/ui/VaeLookAndFeel.h"

namespace vae::experimental
{
using LF = VaeLookAndFeel;
namespace
{
constexpr int kPad = 6, kHeaderH = 24, kButtonW = 24;

juce::TextLayout descLayout (const juce::String& text, int width)
{
    juce::AttributedString s (text);
    s.setFont (juce::Font (LF::labelFont (10.0f)));
    s.setColour (LF::quiet());
    s.setWordWrap (juce::AttributedString::byWord);
    juce::TextLayout layout;
    layout.createLayout (s, (float) juce::jmax (1, width));
    return layout;
}
} // namespace

LabAxisList::Row::Row (const LatentAxis& a, int r, float maxImp)
    : axis (a)
    , rank (r)
    , maxImpact (maxImp)
{
    for (auto& b : slot)
    {
        b.setClickingTogglesState (false);
        addAndMakeVisible (b);
    }
    slider.setRange (0.0, 1.0, 0.001);
    slider.setSliderStyle (juce::Slider::LinearHorizontal);
    slider.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    slider.setTitle (axis.name);
    addAndMakeVisible (slider);
}
int LabAxisList::Row::heightFor (int width) const
{
    const auto desc = (int) std::ceil (descLayout (axis.desc, width - 2 * kPad).getHeight());
    return kPad + 20 + 4 + desc + 4 + 14 + 4 + 20 + kPad;
}
void LabAxisList::Row::resized()
{
    auto r = getLocalBounds().reduced (kPad);
    auto top = r.removeFromTop (20);
    for (int i = 2; i >= 0; --i)
    {
        slot[i].setBounds (top.removeFromRight (kButtonW));
        top.removeFromRight (2);
    }
    slider.setBounds (r.removeFromBottom (20).reduced (64, 0));
}
void LabAxisList::Row::paint (juce::Graphics& g)
{
    const bool on = slot[0].getToggleState() || slot[1].getToggleState() || slot[2].getToggleState();
    if (on)
        g.fillAll (LF::faint.withAlpha (0.6f));
    g.setColour (LF::faint);
    g.drawHorizontalLine (getHeight() - 1, 0, (float) getWidth());

    auto r = getLocalBounds().reduced (kPad);
    auto top = r.removeFromTop (20).withTrimmedRight (3 * kButtonW + 8);
    g.setColour (LF::quiet());
    g.setFont (LF::labelFont (10.0f));
    g.drawText (juce::String (rank), top.removeFromLeft (22), juce::Justification::centredLeft);
    g.setColour (LF::ink);
    g.setFont (LF::labelFont (11.0f, true));
    g.drawFittedText (axis.name, top, juce::Justification::centredLeft, 1, 0.8f);
    r.removeFromTop (4);

    auto layout = descLayout (axis.desc, r.getWidth());
    layout.draw (g, r.removeFromTop ((int) std::ceil (layout.getHeight())).toFloat());
    r.removeFromTop (4);

    auto tags = r.removeFromTop (14);
    g.setFont (LF::labelFont (9.5f));
    juce::String text;
    if (axis.kind == "dim")
    {
        auto meter = tags.removeFromLeft (60).withSizeKeepingCentre (60, 4).toFloat();
        g.setColour (LF::faint);
        g.fillRoundedRectangle (meter, 2);
        g.setColour (axis.minor ? LF::quiet() : LF::ink);
        g.fillRoundedRectangle (meter.withWidth (meter.getWidth() * juce::jlimit (0.0f, 1.0f, axis.impact / maxImpact)),
                                2);
        tags.removeFromLeft (6);
        text = juce::String (axis.impact, 1) + " dB";
        for (size_t i = 0; i < axis.sections.size() && i < 3; ++i)
            text += "   " + axis.sections[i].first + " "
                    + juce::String (juce::roundToInt (axis.sections[i].second * 100)) + "%";
    }
    else
        text = "sound axis";
    g.setColour (LF::quiet());
    g.drawFittedText (text, tags, juce::Justification::centredLeft, 1, 0.8f);
    r.removeFromTop (4);

    auto ends = r.removeFromBottom (20);
    g.setFont (LF::labelFont (9.5f));
    g.drawFittedText (axis.low, ends.removeFromLeft (60), juce::Justification::centredLeft, 1, 0.7f);
    g.drawFittedText (axis.high, ends.removeFromRight (60), juce::Justification::centredRight, 1, 0.7f);
}

void LabAxisList::setAxes (const std::vector<LatentAxis>& axes)
{
    rows.clear();
    float maxImpact = 1e-6f;
    int minor = 0;
    for (const auto& a : axes)
        if (a.kind == "dim")
        {
            maxImpact = juce::jmax (maxImpact, a.impact);
            minor += a.minor ? 1 : 0;
        }
    int sound = 0, dim = 0;
    for (const auto& a : axes)
    {
        auto* row = rows.add (new Row (a, a.kind == "dim" ? ++dim : ++sound, maxImpact));
        for (int i = 0; i < 3; ++i)
            row->slot[i].onClick = [this, i, &a]
            {
                if (assign)
                    assign (i, a.id);
            };
        row->slider.onValueChange = [this, row]
        {
            if (slid)
                slid (row->axis, (float) row->slider.getValue());
        };
        addAndMakeVisible (row);
    }
    moreButton.setButtonText (juce::String (minor) + " more dimensions (< 2 dB)");
    moreButton.setClickingTogglesState (true);
    moreButton.onClick = [this]
    {
        showMinor = moreButton.getToggleState();
        layOut (getWidth());
    };
    addChildComponent (moreButton);
    moreButton.setVisible (minor > 0);
}
void LabAxisList::setAssigned (const std::array<juce::String, 3>& ids)
{
    for (auto* row : rows)
    {
        for (int i = 0; i < 3; ++i)
            row->slot[i].setToggleState (ids[(size_t) i] == row->axis.id, juce::dontSendNotification);
        row->repaint();
    }
}
void LabAxisList::setPositions (const JointLatent& z, const LatentAxis* except)
{
    for (auto* row : rows)
        if (&row->axis != except)
            row->slider.setValue (juce::jlimit (0.0f, 1.0f, axisPosition (row->axis, z)), juce::dontSendNotification);
}
void LabAxisList::layOut (int width)
{
    int y = 0;
    soundHeaderY = dimHeaderY = -1;
    const auto place = [&] (Row* row)
    {
        const int h = row->heightFor (width);
        row->setBounds (0, y, width, h);
        y += h;
    };
    for (auto* row : rows)
        if (row->axis.kind == "sound")
        {
            if (soundHeaderY < 0)
            {
                soundHeaderY = y;
                y += kHeaderH;
            }
            place (row);
        }
    for (auto* row : rows)
        if (row->axis.kind == "dim" && ! row->axis.minor)
        {
            if (dimHeaderY < 0)
            {
                dimHeaderY = y;
                y += kHeaderH;
            }
            place (row);
        }
    if (moreButton.isVisible())
    {
        moreButton.setBounds (0, y + 4, width, 24);
        y += 32;
    }
    for (auto* row : rows)
        if (row->axis.minor)
        {
            row->setVisible (showMinor);
            if (showMinor)
                place (row);
        }
    setSize (width, y);
    repaint();
}
void LabAxisList::paint (juce::Graphics& g)
{
    g.setColour (LF::quiet());
    g.setFont (LF::labelFont (10.0f, true));
    for (auto [y, text] : { std::pair { soundHeaderY, "SOUND AXES" }, std::pair { dimHeaderY, "LATENT DIMENSIONS" } })
        if (y >= 0)
            g.drawText (text, kPad, y, getWidth() - 2 * kPad, kHeaderH, juce::Justification::bottomLeft);
}
} // namespace vae::experimental
