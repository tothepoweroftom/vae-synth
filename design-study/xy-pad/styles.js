(() => {
  const captions = {
    precision: 'Fine lines. Quiet precision.',
    brutalist: 'Heavy rules. Unapologetic controls.',
    swiss: 'Clear type. Rigorous alignment.',
    industrial: 'Tactile controls. Studio equipment.',
    editorial: 'Expressive type. Space to explore.'
  };
  function applyStyle(style) {
    if (!Object.hasOwn(captions, style)) style = 'precision';
    document.documentElement.dataset.style = style;
    document.querySelectorAll('[data-style-choice]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.styleChoice === style));
    });
    document.getElementById('style-caption').textContent = captions[style];
    const url = new URL(location.href);
    url.searchParams.set('style', style);
    history.replaceState(null, '', url);
    try { localStorage.setItem('precision-xypad-style', style); } catch {}
    window.dispatchEvent(new Event('xypad-style-change'));
  }
  let saved;
  try { saved = localStorage.getItem('precision-xypad-style'); } catch {}
  applyStyle(new URL(location.href).searchParams.get('style') || saved || 'precision');
  document.querySelectorAll('[data-style-choice]').forEach(button => {
    button.addEventListener('click', () => applyStyle(button.dataset.styleChoice));
  });
  // A fragment-only About link should reveal the details it points to.
  document.querySelector('.about-link').addEventListener('click', () => {
    document.getElementById('research').open = true;
  });
})();
