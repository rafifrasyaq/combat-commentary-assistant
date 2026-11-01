ALTER TABLE processing_jobs
    ADD COLUMN artifact_manifest_path VARCHAR(2048);

-- The manifest references developer-only processing artifacts stored outside the Git repository.
