const API = '/api';
const documentList = document.querySelector('#document-list');
const filePicker = document.querySelector('#file-picker');
const dropzone = document.querySelector('#dropzone');
const form = document.querySelector('#question-form');
const questionInput = document.querySelector('#question-input');
const conversation = document.querySelector('#conversation');
const sourcesBody = document.querySelector('#sources-body');
const sourceCount = document.querySelector('#source-count');
const toast = document.querySelector('#toast');
let toastTimer;

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch { /* The response may not be JSON. */ }
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add('is-visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('is-visible'), 3600);
}

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function relativeDate(value) {
  const date = new Date(value);
  return new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' }).format(date);
}

function formatBytes(value) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

async function loadDocuments() {
  try {
    const documents = await apiRequest('/documents');
    document.querySelector('#document-count').textContent = documents.length;
    documentList.replaceChildren();
    if (!documents.length) {
      documentList.append(createElement('div', 'empty-library', 'Your uploaded documents will appear here.'));
      return;
    }
    for (const document of documents) {
      const item = createElement('article', 'document-item');
      const extension = document.filename.split('.').pop().toUpperCase();
      item.append(createElement('span', 'file-type', extension));
      const meta = createElement('div', 'document-meta');
      meta.append(createElement('p', 'document-name', document.filename));
      meta.append(createElement('p', 'document-detail', `${formatBytes(document.size_bytes)} · ${document.chunk_count} chunks · ${relativeDate(document.uploaded_at)}`));
      item.append(meta);
      const remove = createElement('button', 'delete-button', '×');
      remove.type = 'button';
      remove.title = `Delete ${document.filename}`;
      remove.setAttribute('aria-label', `Delete ${document.filename}`);
      remove.addEventListener('click', async () => {
        if (!window.confirm(`Delete ${document.filename} and its indexed content?`)) return;
        try {
          await apiRequest(`/documents/${encodeURIComponent(document.document_id || document.id)}`, { method: 'DELETE' });
          await loadDocuments();
          showToast('Document and indexed chunks deleted.');
        } catch (error) { showToast(error.message); }
      });
      item.append(remove);
      documentList.append(item);
    }
  } catch (error) {
    documentList.replaceChildren(createElement('div', 'empty-library', error.message));
  }
}

async function uploadFiles(files) {
  for (const file of files) {
    const body = new FormData();
    body.append('file', file);
    showToast(`Indexing ${file.name}…`);
    try {
      const document = await apiRequest('/documents/upload', { method: 'POST', body });
      showToast(`${document.filename} added · ${document.chunk_count} chunks indexed.`);
      await loadDocuments();
    } catch (error) {
      showToast(`${file.name}: ${error.message}`);
    }
  }
}

function addMessage(kind, label, content, extraClass = '') {
  const message = createElement('article', `message ${kind} ${extraClass}`.trim());
  message.append(createElement('span', 'message-label', label));
  const body = createElement('div', 'message-content');
  if (content instanceof Node) body.append(content);
  else body.textContent = content;
  message.append(body);
  conversation.append(message);
  message.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  return message;
}

function renderSources(sources) {
  sourceCount.textContent = sources.length ? String(sources.length).padStart(2, '0') : '—';
  sourcesBody.replaceChildren();
  if (!sources.length) {
    const empty = createElement('div', 'sources-empty');
    empty.append(createElement('span', '', '⌁'));
    empty.append(createElement('p', '', 'No source passages were returned.'));
    sourcesBody.append(empty);
    return;
  }
  for (const source of sources) {
    const item = createElement('article', 'source-item');
    const filename = source.filename || source.document;
    const page = source.page_number ?? source.page;
    const score = source.score ?? source.relevance_score;
    const top = createElement('div', 'source-topline');
    top.append(createElement('span', 'source-document', filename));
    top.append(createElement('span', 'source-score', `${(score * 100).toFixed(0)}%`));
    item.append(top);
    item.append(createElement('span', 'source-page', `${page ? `Page ${page}` : 'Page not available'} · Chunk ${source.chunk_index}`));
    if (source.text) item.append(createElement('p', 'source-excerpt', source.text));
    item.append(createElement('div', 'source-id', `Chunk ${source.chunk_id.slice(0, 8)}`));
    sourcesBody.append(item);
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  const welcome = document.querySelector('#welcome-note');
  welcome?.remove();
  addMessage('question', 'You asked', question);
  questionInput.value = '';
  questionInput.style.height = 'auto';
  const loading = createElement('span', 'typing-indicator');
  loading.setAttribute('aria-label', 'Generating answer');
  loading.append(createElement('i'), createElement('i'), createElement('i'));
  const pending = addMessage('answer', 'Searching your library', loading);
  const send = form.querySelector('button[type="submit"]');
  send.disabled = true;
  try {
    const result = await apiRequest('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    pending.remove();
    addMessage('answer', 'Answer', result.answer);
    renderSources(result.sources);
  } catch (error) {
    pending.remove();
    addMessage('answer', 'Could not answer', error.message, 'error');
    renderSources([]);
  } finally {
    send.disabled = false;
    questionInput.focus();
  }
});

questionInput.addEventListener('input', () => {
  questionInput.style.height = 'auto';
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 140)}px`;
});
questionInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
filePicker.addEventListener('change', () => {
  if (filePicker.files?.length) uploadFiles(filePicker.files);
  filePicker.value = '';
});
for (const eventName of ['dragenter', 'dragover']) {
  dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.add('is-dragging'); });
}
for (const eventName of ['dragleave', 'drop']) {
  dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove('is-dragging'); });
}
dropzone.addEventListener('drop', (event) => {
  if (event.dataTransfer?.files.length) uploadFiles(event.dataTransfer.files);
});
dropzone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); filePicker.click(); }
});
document.querySelector('#refresh-documents').addEventListener('click', loadDocuments);
loadDocuments();
