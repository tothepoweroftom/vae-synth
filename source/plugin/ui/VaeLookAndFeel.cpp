#include "VaeLookAndFeel.h"

#include "BinaryData.h"

namespace vae
{

VaeLookAndFeel::VaeLookAndFeel()
{
    regular = juce::Typeface::createSystemTypefaceFor (BinaryData::SpaceMonoRegular_ttf,
                                                       BinaryData::SpaceMonoRegular_ttfSize);
    bold = juce::Typeface::createSystemTypefaceFor (BinaryData::SpaceMonoBold_ttf, BinaryData::SpaceMonoBold_ttfSize);

    setColourScheme (LookAndFeel_V4::getDarkColourScheme());

    setColour (juce::ResizableWindow::backgroundColourId, paper);
    setColour (juce::Label::textColourId, ink);
    setColour (juce::Slider::rotarySliderFillColourId, accent);
    setColour (juce::ComboBox::textColourId, ink);
    setColour (juce::ComboBox::backgroundColourId, paper);
    setColour (juce::ComboBox::outlineColourId, ink);
    setColour (juce::ComboBox::arrowColourId, ink);
    setColour (juce::PopupMenu::backgroundColourId, paper);
    setColour (juce::PopupMenu::textColourId, ink);
    setColour (juce::PopupMenu::highlightedBackgroundColourId, accent);
    setColour (juce::PopupMenu::highlightedTextColourId, paper);
    setColour (juce::TextButton::buttonColourId, paper);
    setColour (juce::TextButton::buttonOnColourId, ink);
    setColour (juce::TextButton::textColourOffId, ink);
    setColour (juce::TextButton::textColourOnId, paper);
    setColour (juce::TextEditor::backgroundColourId, paper);
    setColour (juce::TextEditor::textColourId, ink);
    setColour (juce::TextEditor::highlightColourId, accent.withAlpha (0.25f));
    setColour (juce::CaretComponent::caretColourId, accent);
    setColour (juce::TooltipWindow::backgroundColourId, ink);
    setColour (juce::TooltipWindow::textColourId, paper);
    setColour (juce::TooltipWindow::outlineColourId, ink);
    setColour (juce::AlertWindow::backgroundColourId, paper);
    setColour (juce::AlertWindow::textColourId, ink);
    setColour (juce::AlertWindow::outlineColourId, ink);
    setColour (juce::BubbleComponent::backgroundColourId, ink);
    setColour (juce::BubbleComponent::outlineColourId, ink);

    setColour (juce::MidiKeyboardComponent::whiteNoteColourId, paper);
    setColour (juce::MidiKeyboardComponent::blackNoteColourId, ink);
    setColour (juce::MidiKeyboardComponent::keySeparatorLineColourId, ink);
    setColour (juce::MidiKeyboardComponent::keyDownOverlayColourId, accent);
    setColour (juce::MidiKeyboardComponent::mouseOverKeyOverlayColourId, accent.withAlpha (0.25f));
    setColour (juce::MidiKeyboardComponent::shadowColourId, juce::Colours::transparentBlack);
    setColour (juce::MidiKeyboardComponent::textLabelColourId, quiet());
    setColour (juce::MidiKeyboardComponent::upDownButtonBackgroundColourId, paper);
    setColour (juce::MidiKeyboardComponent::upDownButtonArrowColourId, ink);
}

juce::FontOptions VaeLookAndFeel::labelFont (float size, bool isBold)
{
    return juce::FontOptions (size, isBold ? juce::Font::bold : juce::Font::plain);
}

juce::Typeface::Ptr VaeLookAndFeel::getTypefaceForFont (const juce::Font& font)
{
    if (regular == nullptr || bold == nullptr)
        return LookAndFeel_V4::getTypefaceForFont (font);
    return font.isBold() ? bold : regular;
}

void VaeLookAndFeel::drawRotarySlider (juce::Graphics& g, int x, int y, int w, int h, float pos, float startAngle,
                                       float endAngle, juce::Slider& s)
{
    const auto bounds = juce::Rectangle<int> (x, y, w, h).toFloat();
    const float radius = juce::jmin (bounds.getWidth(), bounds.getHeight()) * 0.5f - 6.0f;
    const auto centre = bounds.getCentre();
    const float angle = startAngle + pos * (endAngle - startAngle);

    // Hairline track, accent value arc, white needle. Nothing else.
    juce::Path track;
    track.addCentredArc (centre.x, centre.y, radius, radius, 0.0f, startAngle, endAngle, true);
    g.setColour (faint);
    g.strokePath (track, juce::PathStrokeType (2.0f));

    juce::Path value;
    value.addCentredArc (centre.x, centre.y, radius, radius, 0.0f, startAngle, angle, true);
    g.setColour (s.isEnabled() ? accent : quiet());
    g.strokePath (value, juce::PathStrokeType (2.0f));

    const juce::Point<float> tip (centre.x + std::sin (angle) * (radius - 4.0f),
                                  centre.y - std::cos (angle) * (radius - 4.0f));
    g.setColour (ink);
    g.drawLine ({ centre, tip }, 1.5f);
    g.fillEllipse (juce::Rectangle<float> (3.0f, 3.0f).withCentre (centre));
}

void VaeLookAndFeel::drawLinearSlider (juce::Graphics& g, int x, int y, int w, int h, float pos, float minPos,
                                       float maxPos, juce::Slider::SliderStyle style, juce::Slider& s)
{
    if (style != juce::Slider::LinearBar)
    {
        LookAndFeel_V4::drawLinearSlider (g, x, y, w, h, pos, minPos, maxPos, style, s);
        return;
    }
    // Thin faint track with a accent fill up to the value (pos is in pixels).
    const auto track =
        juce::Rectangle<float> ((float) x, (float) y, (float) w, (float) h).withSizeKeepingCentre ((float) w, 4.0f);
    g.setColour (faint);
    g.fillRect (track);
    g.setColour (s.isEnabled() ? accent : quiet());
    g.fillRect (track.withRight (pos));
}

void VaeLookAndFeel::drawToggleButton (juce::Graphics& g, juce::ToggleButton& b, bool highlighted, bool)
{
    // A square: outlined when off, solid accent when on.
    const auto box = b.getLocalBounds().toFloat().withSizeKeepingCentre (12.0f, 12.0f);
    if (b.getToggleState())
    {
        g.setColour (accent);
        g.fillRect (box);
    }
    else
    {
        g.setColour (highlighted ? accent : ink);
        g.drawRect (box, 1.0f);
    }
}

void VaeLookAndFeel::drawComboBox (juce::Graphics& g, int w, int h, bool, int, int, int, int, juce::ComboBox& box)
{
    // Underline only, with a small chevron.
    g.setColour (box.isMouseOver (true) ? accent : ink);
    g.fillRect (0.0f, static_cast<float> (h) - 1.0f, static_cast<float> (w), 1.0f);

    const float cx = static_cast<float> (w) - 6.0f, cy = static_cast<float> (h) * 0.5f;
    juce::Path chevron;
    chevron.startNewSubPath (cx - 3.5f, cy - 1.5f);
    chevron.lineTo (cx, cy + 2.0f);
    chevron.lineTo (cx + 3.5f, cy - 1.5f);
    g.strokePath (chevron, juce::PathStrokeType (1.2f));
}

juce::Font VaeLookAndFeel::getComboBoxFont (juce::ComboBox&) { return juce::Font (juce::FontOptions (13.0f)); }

void VaeLookAndFeel::positionComboBoxText (juce::ComboBox& box, juce::Label& label)
{
    label.setBounds (0, 0, box.getWidth() - 14, box.getHeight() - 1);
    label.setBorderSize ({ 0, 0, 0, 0 });
    label.setFont (getComboBoxFont (box));
}

void VaeLookAndFeel::drawButtonBackground (juce::Graphics& g, juce::Button& b, const juce::Colour&, bool highlighted,
                                           bool down)
{
    // Hairline box; hover/press/toggled-on inverts to solid ink (accent while pressed).
    const auto r = b.getLocalBounds().toFloat().reduced (0.5f);
    if (highlighted || down || b.getToggleState())
    {
        g.setColour (down ? accent : ink);
        g.fillRect (r);
    }
    else
    {
        g.setColour (ink);
        g.drawRect (r, 1.0f);
    }
}

juce::Font VaeLookAndFeel::getTextButtonFont (juce::TextButton&, int) { return juce::Font (labelFont (11.0f, true)); }

void VaeLookAndFeel::drawButtonText (juce::Graphics& g, juce::TextButton& b, bool highlighted, bool down)
{
    g.setFont (getTextButtonFont (b, b.getHeight()));
    g.setColour (highlighted || down || b.getToggleState() ? paper : ink);
    g.drawText (b.getButtonText().toUpperCase(), b.getLocalBounds(), juce::Justification::centred);
}

void VaeLookAndFeel::fillTextEditorBackground (juce::Graphics& g, int, int, juce::TextEditor&) { g.fillAll (paper); }

void VaeLookAndFeel::drawTextEditorOutline (juce::Graphics& g, int w, int h, juce::TextEditor& e)
{
    g.setColour (e.hasKeyboardFocus (true) ? accent : ink);
    g.fillRect (0, h - 1, w, 1);
}

void VaeLookAndFeel::drawPopupMenuBackground (juce::Graphics& g, int w, int h)
{
    g.fillAll (paper);
    g.setColour (ink);
    g.drawRect (0, 0, w, h, 1);
}

juce::Font VaeLookAndFeel::getPopupMenuFont() { return juce::Font (juce::FontOptions (13.0f)); }

} // namespace vae
