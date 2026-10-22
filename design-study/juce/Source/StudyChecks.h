#pragma once
#include "StudyData.h"
#include "OrbitView.h"
#include "BinaryData.h"
#include "ui/PitcherControls.h"
#include "ui/PitcherLayout.h"

// Explicit failures remain active in Release builds. Run on the message thread.
inline bool runStudyChecks(juce::String& report)
{
    report.clear();
    const auto fail = [&](const juce::String& why) { report += "FAIL: " + why + "\n"; return false; };
    const auto close = [](float a, float b) { return std::isfinite(a) && std::abs(a - b) < 1.0e-3f; };
    // Reusable controls render correctly without fonts, model data or an application adapter.
    PitcherLookAndFeel look;
    juce::MidiKeyboardState midi;
    PitcherKeyboard keyboard(midi);
    juce::Image keyImage(juce::Image::RGB,40,60,true);
    juce::Graphics keys(keyImage);
    for(bool black:{false,true}) for(bool down:{false,true}) {
        if(black)keyboard.drawBlackNote(49,keys,{0,0,40,60},down,false,{});
        else keyboard.drawWhiteNote(50,keys,{0,0,40,60},down,false,{},{});
        const juce::Colour expected(down?0xff70ff63:black?0xffb7bcb6:0xff252825);
        if(keyImage.getPixelAt(15,15)!=expected)return fail("inverted keyboard or primary key-down colour");
    }
    juce::Slider slider;slider.setLookAndFeel(&look);
    juce::Image sliderImage(juce::Image::RGB,100,20,true);juce::Graphics sliderGraphics(sliderImage);
    look.drawLinearSlider(sliderGraphics,0,0,100,20,50,0,100,juce::Slider::LinearHorizontal,slider);
    slider.setLookAndFeel(nullptr);
    if(sliderImage.getPixelAt(45,5)!=juce::Colour(0xff70ff63)||sliderImage.getPixelAt(54,14)!=juce::Colour(0xff70ff63))return fail("slider handle must have square green corners");
    for(int width:{1000,1380,1800}) {
        const auto layout=PitcherLayout::calculate({0,0,width,800});
        if(layout.left.getWidth()!=layout.right.getWidth()||layout.left.getHeight()!=layout.right.getHeight()||!layout.centre.contains(layout.keyboard))return fail("shared layout geometry");
    }
    report+="PASS: portable controls (inverted keyboard, green held keys, square slider handles); shared layout at 3 widths.\n";
    StudyData data;
    if (!data.load(BinaryData::study_json, static_cast<size_t>(BinaryData::study_jsonSize)))
        return fail("embedded study data did not load");
    if (data.presets.size() != 1375 || data.labels.size() != data.presets.size())
        return fail("expected 1,375 labelled preset positions");
    if (data.axes.size() < 6) return fail("missing musical axes");
    const std::array<int, 3> assigned {0, 2, 5};
    for (auto target : {std::array<float, 3>{.2f, .4f, .8f}, {.8f, .5f, .3f}, {.5f, .5f, .5f}})
    {
        auto latent = data.initialLatent;
        data.moveTo(latent, assigned, target);
        const auto actual = data.project(latent, assigned);
        for (size_t i = 0; i < 3; ++i)
            if (!close(actual[i], target[i])) return fail("latent constrained move did not reach the requested coordinates");
    }
    const auto spectrum = data.spectrogram(data.initialLatent, data.initialCategory);
    if (!spectrum.isValid() || spectrum.getWidth() < 2 || spectrum.getHeight() < 2)
        return fail("initial predicted spectrogram is empty");
    const auto tooltip = data.soundTooltip();
    for (const auto* measure : {"Brightness:", "Attack:", "Length:", "Noise:", "Movement:"})
        if (!tooltip.contains(measure)) return fail("missing spectrogram measure: " + juce::String(measure));
    if (tooltip.containsIgnoreCase("nan") || tooltip.containsIgnoreCase("inf"))
        return fail("spectrogram measures are non-finite");
    for (int yaw = 0; yaw < 12; ++yaw) for (int pitch = 0; pitch < 6; ++pitch)
    {
        const OrbitMath::Projection p({yaw * .5f, pitch * .5f, 1}, 700, 500);
        const OrbitMath::Position start {.5f, .5f, .5f};
        const auto a = p.project(start);
        auto b = p.project(p.drag(start, 5, -4, OrbitMath::Drag::plane));
        if (!close(b.x - a.x, 5) || !close(b.y - a.y, -4) || !close(b.depth, a.depth))
            return fail("screen-plane inverse projection");
        b = p.project(p.drag(start, 0, -5, OrbitMath::Drag::depth));
        if (!close(b.x, a.x) || !close(b.y, a.y) || !close(b.depth - a.depth, 5 / p.scale))
            return fail("depth inverse projection");
        for (int axis = 0; axis < 3; ++axis)
        {
            const auto moved = p.drag(start, 5, -4,
                static_cast<OrbitMath::Drag>(static_cast<int>(OrbitMath::Drag::x) + axis));
            for (int other = 0; other < 3; ++other)
                if (other != axis && moved[other] != start[other]) return fail("axis constraint leaked into another coordinate");
        }
        const auto bounded = p.drag(start, 1.0e6f, -1.0e6f, OrbitMath::Drag::plane);
        for (auto value : bounded) if (!std::isfinite(value) || value < 0 || value > 1) return fail("cube bounds");
    }
    report += "PASS: 1,375 presets; constrained latent moves; predicted spectrogram and five measures; "
              "72 camera orientations (plane/depth inverse, axis isolation, bounds).\n";

    OrbitView orbit;
    orbit.setSize(700, 500);
    std::vector<PresetPoint> points;
    size_t visible = 0;
    for (size_t i = 0; i < data.presets.size(); ++i)
    {
        const auto p = data.project(data.presets[i], assigned);
        points.push_back({p, data.labels[i]});
        if (std::all_of(p.begin(), p.end(), [](float v) { return std::isfinite(v) && v >= 0 && v <= 1; })) ++visible;
    }
    orbit.setPoints(std::move(points));
    juce::Image target(juce::Image::ARGB, 700, 500, true, juce::SoftwareImageType());
    std::vector<double> times;
    for (int frame = 0; frame < 65; ++frame)
    {
        orbit.setPosition({.45f + .001f * frame, .5f, frame % 2 == 0 ? .4f : .6f});
        juce::Graphics graphics(target);
        orbit.paintEntireComponent(graphics, true);
        if (frame >= 5) times.push_back(orbit.lastPaintMs());
    }
    std::sort(times.begin(), times.end());
    const auto median = (times[29] + times[30]) * .5;
    const auto p95 = times[56];
    report += "CPU software paint, 700 x 500, 5 warmups + 60 measured frames: median "
            + juce::String(median, 3) + " ms; p95 " + juce::String(p95, 3) + " ms. "
            + juce::String(static_cast<int>(visible)) + " visible / "
            + juce::String(static_cast<int>(data.presets.size())) + " total preset points.\n"
              "These are paint callback timings, not presented frame rate or GPU timings.\n";
    return true;
}
