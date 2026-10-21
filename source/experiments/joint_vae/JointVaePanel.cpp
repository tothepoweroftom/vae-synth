#include "JointVaePanel.h"
#include "plugin/ui/VaeLookAndFeel.h"
#include <cmath>
#include <random>

namespace vae::experimental
{
using LF = VaeLookAndFeel;

void JointVaePanel::Pad::paint (juce::Graphics& g)
{
    auto r = getLocalBounds().toFloat().reduced (8);
    g.setColour (LF::faint);
    g.drawHorizontalLine ((int) r.getCentreY(), r.getX(), r.getRight());
    g.drawVerticalLine ((int) r.getCentreX(), r.getY(), r.getBottom());
    g.setColour (LF::ink);
    g.drawRect (r);
    g.drawText ("Anchor", (int) r.getCentreX() + 8, (int) r.getCentreY() + 8, 80, 20, juce::Justification::centredLeft);
    auto point = juce::Point<float> (r.getCentreX() + position.x * r.getWidth() / 2,
                                     r.getCentreY() - position.y * r.getHeight() / 2);
    g.setColour (LF::accent);
    g.fillEllipse (juce::Rectangle<float> (12, 12).withCentre (point));
}
void JointVaePanel::Pad::mouseDrag (const juce::MouseEvent& e)
{
    auto r = getLocalBounds().toFloat().reduced (8);
    if (r.isEmpty())
        return;
    position = { juce::jlimit (-1.0f, 1.0f, 2 * (e.position.x - r.getCentreX()) / r.getWidth()),
                 juce::jlimit (-1.0f, 1.0f, 2 * (r.getCentreY() - e.position.y) / r.getHeight()) };
    repaint();
    if (moved)
        moved();
}
JointVaePanel::JointVaePanel()
{
    title.setText ("Joint VAE - latent explorer", juce::dontSendNotification);
    title.setFont (juce::FontOptions (20));
    description.setText ("Explore the student's trained decoder separately from SAME retrieval. "
                         "Each decoded point plays as a full Surge patch; text-to-audio encoding is not connected.",
                         juce::dontSendNotification);
    description.setMinimumHorizontalScale (1.0f);
    axisLabel.setText ("X dimension                 Y dimension                 Radius (latent units)",
                       juce::dontSendNotification);
    for (auto* c : std::initializer_list<juce::Component*> {
             &title, &description, &axisLabel, &status, &loadButton, &importButton, &randomButton, &resetButton,
             &exportButton, &category, &xAxis, &yAxis, &radius, &pad, &values, &prompt, &promptButton })
        addAndMakeVisible (c);
    prompt.setTextToShowWhenEmpty ("Describe a sound, e.g. \"dark reese bass\" (label demo: only the noun counts)",
                                   juce::Colours::grey);
    prompt.onReturnKey = [this] { fromPrompt(); };
    promptButton.onClick = [this] { fromPrompt(); };
    for (int i = 0; i < 9; ++i)
        category.addItem (jointLabels[i], i + 1);
    category.setSelectedId (1, juce::dontSendNotification);
    for (int i = 0; i < 32; ++i)
    {
        xAxis.addItem ("z" + juce::String (i + 1), i + 1);
        yAxis.addItem ("z" + juce::String (i + 1), i + 1);
    }
    xAxis.setSelectedId (1, juce::dontSendNotification);
    yAxis.setSelectedId (2, juce::dontSendNotification);
    radius.setRange (0.0, 3.0, 0.05);
    radius.setValue (1.0, juce::dontSendNotification);
    radius.setSliderStyle (juce::Slider::LinearHorizontal);
    radius.setTextBoxStyle (juce::Slider::TextBoxRight, false, 50, 24);
    values.setMultiLine (true);
    values.setReadOnly (true);
    values.setFont (juce::FontOptions (12));
    values.setText (
        "Loads models/joint-vae/bundle.json from the app data folder (scripts/install-models.sh), or pick one.\n\n"
        "The pad is a local 2D slice through a 32D latent. Its axes have no assigned perceptual meaning.\n\n"
        "Release the pad to decode. The keyboard then plays the decoded Surge patch; loading a preset or "
        "generating from a prompt returns the synth to its own controls.");
    loadButton.onClick = [this] { chooseBundle(); };
    importButton.onClick = [this] { importPoint(); };
    exportButton.onClick = [this] { exportPoint(); };
    resetButton.onClick = [this]
    {
        centre();
        decode();
    };
    category.onChange = [this] { decode(); };
    xAxis.onChange = yAxis.onChange = [this]
    {
        centre();
        decode();
    };
    radius.onValueChange = [this] { decode(); };
    pad.released = [this] { decode(); };
    pad.moved = [this]
    {
        worker.cancel();
        outputValid = false;
        exportButton.setEnabled (false);
    };
    randomButton.onClick = [this]
    {
        std::mt19937 rng ((uint32_t) juce::Random::getSystemRandom().nextInt());
        std::normal_distribution<float> normal;
        for (auto& value : anchor)
            value = normal (rng);
        centre();
        decode();
    };
    enableActions (false);
    // Installed by scripts/install-models.sh, like the Prompt tab's models; the button overrides it
    if (const auto installed = ml::appDataDir().getChildFile ("models/joint-vae/bundle.json"); installed.existsAsFile())
        loadBundle (installed);
    else
        setStatus ("No VAE bundle installed: run scripts/install-models.sh, or Load VAE bundle...");
}
void JointVaePanel::fromPrompt()
{
    if (! loaded)
        return;
    const auto match = JointVae::labelFor (prompt.getText());
    // z = 0: the label's typical sound; Random anchor then gives variations within the label
    category.setSelectedId (match.category + 1, juce::dontSendNotification);
    anchor = {};
    centre();
    decode();
    promptNote = match.noun.isEmpty() ? "no label noun found -> Unknown - "
                                      : "\"" + match.noun + "\" -> " + jointLabels[match.category] + " - ";
}
void JointVaePanel::setStatus (juce::String text) { status.setText (text, juce::dontSendNotification); }
void JointVaePanel::centre()
{
    pad.position = {};
    pad.repaint();
}
void JointVaePanel::enableActions (bool enable)
{
    for (auto* c : std::initializer_list<juce::Component*> { &importButton, &randomButton, &resetButton, &category,
                                                             &xAxis, &yAxis, &radius, &pad, &prompt, &promptButton })
        c->setEnabled (enable);
    exportButton.setEnabled (enable && outputValid);
}
void JointVaePanel::chooseBundle()
{
    chooser = std::make_unique<juce::FileChooser> ("Load joint VAE bundle.json", juce::File(), "*.json");
    chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [safe = juce::Component::SafePointer (this)] (const juce::FileChooser& fc)
                          {
                              if (safe && fc.getResult() != juce::File())
                                  safe->loadBundle (fc.getResult());
                          });
}
void JointVaePanel::loadBundle (const juce::File& file)
{
    enableActions (false);
    loadButton.setEnabled (false);
    setStatus ("Loading and checking joint VAE...");
    worker.load (file,
                 [safe = juce::Component::SafePointer (this)] (juce::Result r, JointVaeWorker::Output output)
                 {
                     if (! safe)
                         return;
                     safe->loadButton.setEnabled (true);
                     if (r.failed())
                     {
                         safe->enableActions (safe->loaded);
                         safe->setStatus (r.getErrorMessage());
                         return;
                     }
                     safe->loaded = true;
                     safe->current = std::move (output);
                     safe->anchor = {};
                     safe->centre();
                     safe->enableActions (true);
                     safe->decode();
                 });
}
void JointVaePanel::decode()
{
    if (! loaded)
        return;
    worker.cancel();
    outputValid = false;
    promptNote = {};
    exportButton.setEnabled (false);
    JointLatent point;
    if (auto r = JointVae::slice (anchor, xAxis.getSelectedId() - 1, yAxis.getSelectedId() - 1, pad.position.x,
                                  pad.position.y, (float) radius.getValue(), point);
        r.failed())
    {
        setStatus (r.getErrorMessage());
        return;
    }
    setStatus ("Decoding...");
    worker.decode (point, category.getSelectedId() - 1,
                   [safe = juce::Component::SafePointer (this)] (juce::Result r, JointVaeWorker::Output out)
                   {
                       if (! safe)
                           return;
                       if (r.failed())
                           safe->setStatus (r.getErrorMessage());
                       else
                           safe->showOutput (std::move (out));
                   });
}
void JointVaePanel::showOutput (JointVaeWorker::Output out)
{
    current = std::move (out);
    outputValid = true;
    exportButton.setEnabled (true);
    juce::String text;
    for (int i = 0; i < current.layout.names.size(); ++i)
        text += current.layout.names[i] + " = " + juce::String (current.codes[(size_t) i], 5) + "\n";
    values.setText (text, false);
    auto playback = current.patchResult;
    if (playback.wasOk())
        playback = onPatch ? onPatch (current.patch) : juce::Result::fail ("playback not connected");
    setStatus (promptNote + current.layout.id + " - " + juce::String (current.layout.names.size())
               + " decoded parameters - "
               + (playback.wasOk() ? juce::String ("playing on Surge") : playback.getErrorMessage()));
}
void JointVaePanel::importPoint()
{
    chooser = std::make_unique<juce::FileChooser> ("Import a matching bundle's sound code or latent point",
                                                   juce::File(), "*.json");
    chooser->launchAsync (
        juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
        [safe = juce::Component::SafePointer (this)] (const juce::FileChooser& fc)
        {
            if (! safe || fc.getResult() == juce::File())
                return;
            auto data = juce::JSON::parse (fc.getResult());
            const bool sound = data.hasProperty ("sound");
            auto* array = data[sound ? "sound" : "latent"].getArray();
            const auto cat = data["category"];
            if (data["bundle_id"].toString() != safe->current.layout.id || ! cat.isInt() || (int) cat < 0
                || (int) cat >= 9 || array == nullptr || array->size() != (sound ? 48 : 32))
            {
                safe->setStatus ("Expected matching bundle_id, category 0-8, and sound[48] or latent[32].");
                return;
            }
            for (const auto& v : *array)
                if (! (v.isDouble() || v.isInt() || v.isInt64()) || ! std::isfinite ((float) v))
                {
                    safe->setStatus ("Input must contain finite numbers.");
                    return;
                }
            safe->category.setSelectedId ((int) cat + 1, juce::dontSendNotification);
            safe->centre();
            if (! sound)
            {
                for (int i = 0; i < 32; ++i)
                    safe->anchor[(size_t) i] = (float) (*array)[i];
                safe->decode();
                return;
            }
            SoundCode code;
            for (int i = 0; i < 48; ++i)
                code[(size_t) i] = (float) (*array)[i];
            safe->outputValid = false;
            safe->enableActions (false);
            safe->loadButton.setEnabled (false);
            safe->setStatus ("Encoding sound code...");
            safe->worker.encode (code, (int) cat,
                                 [safe] (juce::Result r, JointVaeWorker::Output out)
                                 {
                                     if (! safe)
                                         return;
                                     safe->loadButton.setEnabled (true);
                                     safe->enableActions (true);
                                     if (r.failed())
                                         safe->setStatus (r.getErrorMessage());
                                     else
                                     {
                                         safe->anchor = out.latent;
                                         safe->showOutput (std::move (out));
                                     }
                                 });
        });
}
void JointVaePanel::exportPoint()
{
    if (! outputValid)
        return;
    auto* object = new juce::DynamicObject();
    juce::var data (object);
    object->setProperty ("bundle_id", current.layout.id);
    object->setProperty ("category", category.getSelectedId() - 1);
    juce::Array<juce::var> latent, codes, names;
    for (auto v : current.latent)
        latent.add (v);
    for (auto v : current.codes)
        codes.add (v);
    for (auto& name : current.layout.names)
        names.add (name);
    object->setProperty ("latent", latent);
    object->setProperty ("codes", codes);
    object->setProperty ("features", names);
    chooser =
        std::make_unique<juce::FileChooser> ("Export decoded point (not a plug-in preset)", juce::File(), "*.json");
    chooser->launchAsync (
        juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles
            | juce::FileBrowserComponent::warnAboutOverwriting,
        [safe = juce::Component::SafePointer (this), data] (const juce::FileChooser& fc)
        {
            if (! safe || fc.getResult() == juce::File())
                return;
            safe->setStatus (fc.getResult().withFileExtension ("json").replaceWithText (juce::JSON::toString (data))
                                 ? "Decoded point exported."
                                 : "Could not save decoded point.");
        });
}
void JointVaePanel::resized()
{
    auto r = getLocalBounds();
    title.setBounds (r.removeFromTop (30));
    description.setBounds (r.removeFromTop (48));
    auto buttons = r.removeFromTop (28);
    loadButton.setBounds (buttons.removeFromLeft (180));
    importButton.setBounds (buttons.removeFromLeft (210));
    randomButton.setBounds (buttons.removeFromLeft (140));
    resetButton.setBounds (buttons.removeFromLeft (80));
    category.setBounds (buttons.removeFromLeft (140));
    exportButton.setBounds (buttons);
    r.removeFromTop (8);
    auto promptRow = r.removeFromTop (28);
    promptButton.setBounds (promptRow.removeFromRight (160));
    prompt.setBounds (promptRow.withTrimmedRight (8));
    r.removeFromTop (8);
    status.setBounds (r.removeFromBottom (28));
    auto left = r.removeFromLeft (r.getWidth() / 2);
    left.removeFromRight (16);
    axisLabel.setBounds (left.removeFromTop (24));
    auto axes = left.removeFromTop (28);
    xAxis.setBounds (axes.removeFromLeft (120));
    axes.removeFromLeft (12);
    yAxis.setBounds (axes.removeFromLeft (120));
    axes.removeFromLeft (12);
    radius.setBounds (axes);
    pad.setBounds (left.reduced (0, 12));
    values.setBounds (r);
}
} // namespace vae::experimental
