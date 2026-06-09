import { createClient } from '@supabase/supabase-js';

const url = process.env.REACT_APP_SUPABASE_URL || 'http://localhost';
const anonKey = process.env.REACT_APP_SUPABASE_ANON_KEY || 'missing-anon-key';

console.info('[supabase] url:', url, '| key set:', anonKey !== 'missing-anon-key');

export const SUPABASE_URL = url;
export const supabase = createClient(url, anonKey);
export const BUCKET = process.env.REACT_APP_SUPABASE_BUCKET || 'files';
