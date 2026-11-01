package com.combatassistant.backend.processing;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "processing_jobs")
public class ProcessingJob {

    @Id
    private UUID id;

    @Column(name = "video_id", nullable = false)
    private UUID videoId;

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false, length = 30)
    private ProcessingStatus status;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    @Column(name = "started_at")
    private Instant startedAt;

    @Column(name = "finished_at")
    private Instant finishedAt;

    @Column(name = "result_summary", length = 1000)
    private String resultSummary;

    @Column(name = "artifact_manifest_path", length = 2048)
    private String artifactManifestPath;

    @Column(name = "error_message", length = 500)
    private String errorMessage;

    protected ProcessingJob() { }

    public ProcessingJob(UUID id, UUID videoId, Instant createdAt) {
        this.id = id;
        this.videoId = videoId;
        this.createdAt = createdAt;
        this.status = ProcessingStatus.QUEUED;
    }

    public void markProcessing(Instant now) {
        requireStatus(ProcessingStatus.QUEUED);
        status = ProcessingStatus.PROCESSING;
        startedAt = now;
    }

    public void markCompleted(String summary, String manifestPath, Instant now) {
        requireStatus(ProcessingStatus.PROCESSING);
        status = ProcessingStatus.COMPLETED;
        resultSummary = summary;
        artifactManifestPath = manifestPath;
        finishedAt = now;
    }

    public void markFailed(String message, Instant now) {
        if (status != ProcessingStatus.QUEUED && status != ProcessingStatus.PROCESSING) {
            throw new IllegalStateException("Only active jobs can fail");
        }
        status = ProcessingStatus.FAILED;
        errorMessage = message;
        finishedAt = now;
    }

    private void requireStatus(ProcessingStatus expected) {
        if (status != expected) {
            throw new IllegalStateException("Invalid processing transition: " + status + " -> " + expected);
        }
    }

    public UUID getId() { return id; }
    public UUID getVideoId() { return videoId; }
    public ProcessingStatus getStatus() { return status; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getStartedAt() { return startedAt; }
    public Instant getFinishedAt() { return finishedAt; }
    public String getResultSummary() { return resultSummary; }
    public String getArtifactManifestPath() { return artifactManifestPath; }
    public String getErrorMessage() { return errorMessage; }
}
