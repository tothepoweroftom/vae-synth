#pragma once
#include <juce_gui_basics/juce_gui_basics.h>

// Shared geometry only. Both hosts can lay out their own controller-bound components.
struct PitcherLayout
{
    juce::Rectangle<int> left, centre, right, orbit, prompt, submit, status, keyboard;
    static PitcherLayout calculate(juce::Rectangle<int> bounds)
    {
        PitcherLayout out;
        auto r=bounds.reduced(24);
        const int side=juce::jlimit(230,300,(int)(bounds.getWidth()*.205f));
        out.left=r.removeFromLeft(side);r.removeFromLeft(24);
        out.right=r.removeFromRight(side);r.removeFromRight(24);out.centre=r;
        out.keyboard=r.removeFromBottom(56);r.removeFromBottom(14);
        out.status=r.removeFromBottom(23);
        auto entry=r.removeFromBottom(42);out.submit=entry.removeFromRight(36);entry.removeFromRight(8);out.prompt=entry;
        r.removeFromBottom(14);out.orbit=r;
        return out;
    }
};
