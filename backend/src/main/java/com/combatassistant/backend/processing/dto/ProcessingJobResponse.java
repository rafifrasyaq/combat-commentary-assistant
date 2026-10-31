package com.combatassistant.backend.processing.dto;

import com.combatassistant.backend.processing.ProcessingJob;
import com.combatassistant.backend.processing.ProcessingStatus;

import java.time.Instant;
import java.util.UUID;

public record ProcessingJobResponse(
        UUID jobId,
        UUID videoId,
        ProcessingStatus status,
        Instant createdAt,
        Instant startedAt,
        Instant finishedAt,
        String resultSummary,
        String errorMessage
) {
    public static ProcessingJobResponse from(ProcessingJob job) {
        return new ProcessingJobResponse(job.getId(), job.getVideoId(), job.getStatus(),
                job.getCreatedAt(), job.getStartedAt(), job.getFinishedAt(),
                job.getResultSummary(), job.getErrorMessage());
    }
}
