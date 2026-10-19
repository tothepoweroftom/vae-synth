#include "SynthEngine.h"

#include "lite/LiteEngine.h"

#if VAE_WITH_SURGE
 #include "surge/SurgeEngine.h"
#endif

namespace vae
{

std::vector<EngineKind> availableEngines()
{
    std::vector<EngineKind> kinds { EngineKind::Lite };
#if VAE_WITH_SURGE
    kinds.push_back (EngineKind::Surge);
#endif
    return kinds;
}

std::unique_ptr<SynthEngine> createEngine (EngineKind kind)
{
    switch (kind)
    {
        case EngineKind::Lite:
            return std::make_unique<LiteEngine>();
        case EngineKind::Surge:
#if VAE_WITH_SURGE
            return std::make_unique<SurgeEngine>();
#else
            return nullptr;
#endif
    }
    return nullptr;
}

} // namespace vae
