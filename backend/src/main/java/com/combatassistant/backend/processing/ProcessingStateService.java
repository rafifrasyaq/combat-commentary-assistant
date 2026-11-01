package com.combatassistant.backend.processing;

import com.combatassistant.backend.video.Video;
import com.combatassistant.backend.video.VideoRepository;
import com.combatassistant.backend.video.VideoStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.Optional;
import java.util.UUID;

@Service
public class ProcessingStateService {

    private final ProcessingJobRepository jobs;
    private final VideoRepository videos;

    public ProcessingStateService(ProcessingJobRepository jobs, VideoRepository videos) {
        this.jobs = jobs;
        this.videos = videos;
    }

    @Transactional
    public Optional<ProcessingInput> start(UUID jobId) {
        ProcessingJob job = lockJob(jobId);
        if (job.getStatus() != ProcessingStatus.QUEUED) {
            return Optional.empty(); // Duplicate dispatch is safe.
        }
        Video video = getVideo(job.getVideoId());
        job.markProcessing(Instant.now());
        video.setStatus(VideoStatus.PROCESSING);
        return Optional.of(new ProcessingInput(job.getId(), video.getId(), video.getStoredPath()));
    }

    @Transactional
    public void succeed(UUID jobId, String summary, String manifestPath) {
        ProcessingJob job = lockJob(jobId);
        if (job.getStatus() != ProcessingStatus.PROCESSING) return;
        job.markCompleted(summary, manifestPath, Instant.now());
        getVideo(job.getVideoId()).setStatus(VideoStatus.COMPLETED);
    }

    @Transactional
    public void fail(UUID jobId, String publicMessage) {
        ProcessingJob job = lockJob(jobId);
        if (job.getStatus() != ProcessingStatus.PROCESSING &&
                job.getStatus() != ProcessingStatus.QUEUED) return;
        job.markFailed(publicMessage, Instant.now());
        getVideo(job.getVideoId()).setStatus(VideoStatus.FAILED);
    }

    private ProcessingJob lockJob(UUID jobId) {
        return jobs.findByIdForUpdate(jobId)
                .orElseThrow(() -> new IllegalStateException("Processing job not found"));
    }

    private Video getVideo(UUID videoId) {
        return videos.findById(videoId)
                .orElseThrow(() -> new IllegalStateException("Video record missing"));
    }
}
