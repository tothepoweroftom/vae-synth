(() => {
  const form = document.getElementById('chat-form');
  const input = document.getElementById('chat-prompt');
  const send = document.getElementById('send-chat');
  const status = document.getElementById('prompt-status');
  // A transparent local demo; no request is sent to a chat or text-to-latent model.
  const vocabulary = [
    ['synth strings', 'Orchestral'], ['e-piano', 'Keys'], ['supersaw', 'Lead'],
    ['orchestral', 'Orchestral'], ['atmosphere', 'Atmosphere'], ['strings', 'Orchestral'],
    ['reese', 'Bass'], ['bass', 'Bass'], ['808', 'Bass'], ['pad', 'Pad'],
    ['pluck', 'Pluck'], ['piano', 'Keys'], ['keys', 'Keys'], ['bell', 'Keys'],
    ['lead', 'Lead'], ['drone', 'Atmosphere'], ['synth', 'Synth']
  ];
  function submit() {
    const prompt = input.value.trim();
    if (!prompt) return;
    const match = vocabulary.find(([noun]) => new RegExp('\\b' + noun + '\\b', 'i').test(prompt));
    if (!match) {
      status.textContent = 'Try a category such as bass, pad, pluck or lead. This demo only matches category words; it does not interpret descriptions or follow-up edits.';
    } else {
      const category = match[1];
      const select = document.getElementById('cat');
      const option = [...select.options].find(option => option.textContent === category);
      if (option) {
        if (document.getElementById('m-latent').getAttribute('aria-pressed') !== 'true') document.getElementById('m-latent').click();
        select.value = option.value; select.dispatchEvent(new Event('change', {bubbles: true}));
        status.textContent = category + ' selected · descriptive words aren’t applied in this demo.';
      } else {
        status.textContent = 'The ' + category + ' category is not available in this model.';
      }
    }
    input.focus();
  }
  form.addEventListener('submit', event => { event.preventDefault(); submit(); });
  input.addEventListener('input', () => { send.disabled = !input.value.trim(); });
})();
