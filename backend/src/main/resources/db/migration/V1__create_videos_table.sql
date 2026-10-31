CREATE TABLE videos (
    id UUID PRIMARY KEY,
    original_filename VARCHAR(255) NOT NULL,
    stored_path VARCHAR(1000) NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    file_size BIGINT NOT NULL,
    status VARCHAR(30) NOT NULL,
    uploaded_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_videos_status
    ON videos(status);
