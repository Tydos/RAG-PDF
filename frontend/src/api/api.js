const API = process.env.REACT_APP_API_PREFIX || '';

async function request(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail ?? j.error ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return res.json();
}

const postJSON = (url, body) =>
  request(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

export const uploadPDF = async (file) => {
  const form = new FormData();
  form.append('file', file);
  return request(`${API}/upload`, { method: 'POST', body: form });
};

export const listDocuments = () =>
  request(`${API}/documents`).then((d) => d.documents);

export const deleteDocument = (filename) =>
  request(`${API}/files/${encodeURIComponent(filename)}`, { method: 'DELETE' });

export const query = (question, { topK = 5, filenames, searchMode = 'hybrid' } = {}) =>
  postJSON(`${API}/query`, {
    question,
    top_k: topK,
    search_mode: searchMode,
    ...(filenames && filenames.length ? { filenames } : {}),
  });

export const getHistory = () =>
  request(`${API}/history`).then((d) => d.messages);

export const chat = (question, { topK = 5, filenames, searchMode = 'hybrid' } = {}) =>
  postJSON(`${API}/chat`, {
    question,
    top_k: topK,
    search_mode: searchMode,
    ...(filenames && filenames.length ? { filenames } : {}),
  });

export const getEvalSummary = () => request(`${API}/eval/summary`);
