(() => {
  const form = document.getElementById('chat-form');
  const input = document.getElementById('chat-prompt');
  const send = document.getElementById('send-chat');
  const log = document.getElementById('chat-log');
  const greeting = log.firstElementChild.cloneNode(true);
  // A transparent local demo; no request is sent to a chat or text-to-latent model.
  const vocabulary = [
    ['synth strings', 'Orchestral'], ['e-piano', 'Keys'], ['supersaw', 'Lead'],
    ['orchestral', 'Orchestral'], ['atmosphere', 'Atmosphere'], ['strings', 'Orchestral'],
    ['reese', 'Bass'], ['bass', 'Bass'], ['808', 'Bass'], ['pad', 'Pad'],
    ['pluck', 'Pluck'], ['piano', 'Keys'], ['keys', 'Keys'], ['bell', 'Keys'],
    ['lead', 'Lead'], ['drone', 'Atmosphere'], ['synth', 'Synth']
  ];
  function message(role, text) {
    const row = document.createElement('div');
    row.className = 'chat-message ' + role;
    const speaker = document.createElement('span'); speaker.className = 'chat-speaker';
    speaker.textContent = role === 'user' ? 'You' : 'VAE';
    const paragraph = document.createElement('p'); paragraph.textContent = text;
    row.append(speaker, paragraph); log.append(row); log.scrollTop = log.scrollHeight;
  }
  function submit() {
    const prompt = input.value.trim();
    if (!prompt) return;
    message('user', prompt); input.value = ''; send.disabled = true;
    const match = vocabulary.find(([noun]) => new RegExp('\\b' + noun + '\\b', 'i').test(prompt));
    if (!match) {
      message('assistant', 'Try a category such as bass, pad, pluck or lead. This demo only matches category words; it does not interpret descriptions or follow-up edits.');
    } else {
      const category = match[1];
      const select = document.getElementById('cat');
      const option = [...select.options].find(option => option.textContent === category);
      if (option) {
        if (document.getElementById('m-latent').getAttribute('aria-pressed') !== 'true') document.getElementById('m-latent').click();
        select.value = option.value; select.dispatchEvent(new Event('change', {bubbles: true}));
        message('assistant', category + ' selected at the current pad position. Explore with X and Y. Descriptive words aren’t applied in this demo.');
      } else {
        message('assistant', 'The ' + category + ' category is not available in this model.');
      }
    }
    input.focus();
  }
  form.addEventListener('submit', event => { event.preventDefault(); submit(); });
  input.addEventListener('input', () => { send.disabled = !input.value.trim(); });
  input.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); submit(); }
  });
  document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => {
    input.value = button.dataset.prompt; send.disabled = false; input.focus();
  }));
  document.getElementById('clear-chat').addEventListener('click', () => {
    log.replaceChildren(greeting.cloneNode(true)); input.value = ''; send.disabled = true; input.focus();
  });
})();
