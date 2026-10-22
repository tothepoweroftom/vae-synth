#include "LabAxisList.h"

namespace vae::experimental
{
void LabAxisList::setAxes (const std::vector<LatentAxis>& a)
{
    rows.clear();
    if (&a != &groupAxes)
        groupAxes.clear();
    axes = &a;
    values.assign (a.size(), 0.5f);
    assigned = { -1, -1, -1 };
    for (size_t i = 0; i < a.size(); ++i)
    {
        auto* row = rows.add (new AxisRow (
            a[i].name, (int) i,
            [this] (int index, int slot)
            {
                if (assign)
                    assign (slot, (*axes)[(size_t) index].id);
            },
            [this] (int index, float pos)
            {
                values[(size_t) index] = pos;
                if (slid)
                    slid ((*axes)[(size_t) index], pos);
            }));
        addAndMakeVisible (row);
        update (i);
    }
    layOut (getWidth());
}
void LabAxisList::setGroups (const KnobGroups& kg)
{
    groupAxes.clear();
    for (const auto& g : kg.groups)
    {
        LatentAxis a;
        a.id = a.name = g.name;
        a.kind = "group";
        groupAxes.push_back (std::move (a));
    }
    setAxes (groupAxes);
}
void LabAxisList::update (size_t row) { rows[(int) row]->update (assigned, juce::jlimit (0.0f, 1.0f, values[row])); }
void LabAxisList::setAssigned (const std::array<juce::String, 3>& ids)
{
    for (size_t s = 0; s < 3; ++s)
    {
        assigned[s] = -1;
        for (size_t i = 0; axes && i < axes->size(); ++i)
            if (ids[s].isNotEmpty() && (*axes)[i].id == ids[s])
                assigned[s] = (int) i;
    }
    for (size_t i = 0; i < values.size(); ++i)
        update (i);
    layOut (getWidth());
}
void LabAxisList::setPositions (const JointLatent& z, const LatentAxis* except)
{
    for (size_t i = 0; axes && i < axes->size(); ++i)
        if (&(*axes)[i] != except)
        {
            values[i] = axisPosition ((*axes)[i], z);
            update (i);
        }
}
void LabAxisList::setSlotPositions (const std::array<float, 3>& pos)
{
    for (size_t s = 0; s < 3; ++s)
        if (assigned[s] >= 0)
        {
            values[(size_t) assigned[s]] = pos[s];
            update ((size_t) assigned[s]);
        }
}
void LabAxisList::layOut (int width)
{
    int y = 0;
    for (auto* row : rows)
    {
        row->setBounds (0, y, width, row->getHeight());
        y += row->getHeight();
    }
    setSize (width, y);
}
} // namespace vae::experimental
