-- Run once in Supabase SQL Editor.
-- The table contains only generated public market/news artifacts.
CREATE TABLE IF NOT EXISTS public.site_data (
    key TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE public.site_data ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.site_data FROM anon, authenticated;
