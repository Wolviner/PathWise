-- pgvector (available on Supabase; no extra database needed)
CREATE EXTENSION IF NOT EXISTS vector;

-- CreateTable
CREATE TABLE "rag_chunks" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "variant" TEXT NOT NULL DEFAULT 'llm',
    "doc_type" TEXT NOT NULL,
    "source" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "month" TEXT,
    "headline" TEXT NOT NULL,
    "summary" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "embedding" vector(768),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "rag_chunks_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "rag_chunks_user_id_variant_idx" ON "rag_chunks"("user_id", "variant");

-- Block Supabase's public REST API (anon/authenticated roles) from this table.
-- Prisma and the RAG service connect as the table owner, which bypasses RLS.
ALTER TABLE "rag_chunks" ENABLE ROW LEVEL SECURITY;
