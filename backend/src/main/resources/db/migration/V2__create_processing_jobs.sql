CREATE TABLE processing_jobs (
    id UUID PRIMARY KEY,
    video_id UUID NOT NULL REFERENCES videos(id),
    status VARCHAR(30) NOT NULL CHECK (status IN ('QUEUED','PROCESSING','COMPLETED','FAILED')),
    created_at TIMESTAMPTZ NOT NULL,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    result_summary VARCHAR(1000),
    error_message VARCHAR(500)
);

CREATE INDEX idx_processing_jobs_video_created
    ON processing_jobs (video_id, created_at DESC);

-- Safety invariant: one active processing request per video (retries permitted after failure).
CREATE UNIQUE INDEX uq_processing_jobs_active_video
    ON processing_jobs(video_id)
    WHERE status IN ('QUEUED','PROCESSING');
