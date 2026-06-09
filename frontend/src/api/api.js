import { supabase, BUCKET, SUPABASE_URL } from './supabase';

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

const ragBody = (question, { topK = 5, filenames, searchMode = 'hybrid' } = {}) => ({
  question,
  top_k: topK,
  search_mode: searchMode,
  ...(filenames?.length ? { filenames } : {}),
});

function storagePath(filename) {
  return filename.split('/').map(encodeURIComponent).join('/');
}

function publicBlobUrl(filename) {
  return `${SUPABASE_URL}/storage/v1/object/public/${BUCKET}/${storagePath(filename)}`;
}

function triggerIngest(filename, url) {
  fetch(`${API}/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, url }),
  }).catch((err) => {
    console.warn('[ingest] backend unreachable — file is stored, indexing will run when backend is up:', err.message);
  });
}

async function uploadToStorage(filename, file) {
  const opts = { contentType: file.type || 'application/pdf', upsert: true };
  let { error } = await supabase.storage.from(BUCKET).upload(filename, file, opts);

  // upsert needs UPDATE policy; fall back to remove + upload
  if (error) {
    await supabase.storage.from(BUCKET).remove([filename]);
    ({ error } = await supabase.storage.from(BUCKET).upload(filename, file, opts));
  }

  if (error) {
    const hint = error.message?.includes('policy') || error.statusCode === 400
      ? ' Check Supabase Storage policies for the anon role (see docs/supabase-policies.sql).'
      : '';
    console.error('[supabase] storage upload failed', error);
    throw new Error(`${error.message}${hint}`);
  }
}

// upload is fully independent: Supabase storage + DB row; /ingest is fire-and-forget
export const uploadPDF = async (file) => {
  const filename = file.name;
  if (!filename.toLowerCase().endsWith('.pdf')) {
    throw new Error('Only PDF files are supported.');
  }

  await uploadToStorage(filename, file);

  const publicUrl = publicBlobUrl(filename);

  const { error: dbError } = await supabase.from('uploads').upsert(
    { filename, blob_url: publicUrl, status: 'pending', page_count: 0 },
    { onConflict: 'filename' },
  );
  if (dbError) {
    throw new Error(dbError.message);
  }

  triggerIngest(filename, publicUrl);

  return { status: 'upload recorded', url: publicUrl };
};

// reads uploads table directly from Supabase
export const listDocuments = async () => {
  const { data, error, status, statusText } = await supabase
    .from('uploads')
    .select('filename, blob_url, status, page_count')
    .order('filename');
  if (error) {
    console.error('[supabase] listDocuments failed', { status, statusText, error });
    throw new Error(error.message);
  }
  return data;
};

// deletes from DB (cascades to chunks) and removes from storage
export const deleteDocument = async (filename) => {
  const { error: dbError, status: s1 } = await supabase
    .from('uploads')
    .delete()
    .eq('filename', filename);
  if (dbError) {
    console.error('[supabase] deleteDocument DB failed', { status: s1, error: dbError });
    throw new Error(dbError.message);
  }

  const { error: storageError } = await supabase.storage
    .from(BUCKET)
    .remove([filename]);
  if (storageError) {
    console.error('[supabase] deleteDocument storage failed', { error: storageError });
    throw new Error(storageError.message);
  }
};

// reads messages table directly from Supabase
export const getHistory = async () => {
  const { data, error, status, statusText } = await supabase
    .from('messages')
    .select('role, content, chunks')
    .order('created_at', { ascending: true });
  if (error) {
    console.error('[supabase] getHistory failed', { status, statusText, error });
    throw new Error(error.message);
  }
  return data;
};

// chat and query go through the backend — embeddings and LLM run server-side
export const query = (question, opts) =>
  postJSON(`${API}/query`, ragBody(question, opts));

export const chat = (question, opts) =>
  postJSON(`${API}/chat`, ragBody(question, opts));

export const getHealth = () => request(`${API}/health`);

export const getEvalSummary = () => request(`${API}/eval/summary`);
