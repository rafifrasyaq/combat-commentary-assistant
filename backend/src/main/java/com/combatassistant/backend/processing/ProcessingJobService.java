package com.combatassistant.backend.processing;

import com.combatassistant.backend.processing.dto.ProcessingJobResponse;
import com.combatassistant.backend.video.Video;
import com.combatassistant.backend.video.VideoRepository;
import com.combatassistant.backend.video.VideoStatus;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.context.ApplicationEventPublisher;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

@Service
public class ProcessingJobService {

    private static final List<ProcessingStatus> ACTIVE =
            List.of(ProcessingStatus.QUEUED, ProcessingStatus.PROCESSING);

    private final VideoRepository videos;
    private final ProcessingJobRepository jobs;
    private final ApplicationEventPublisher events;

    public ProcessingJobService(VideoRepository videos, ProcessingJobRepository jobs,
                                ApplicationEventPublisher events) {
        this.videos = videos;
        this.jobs = jobs;
        this.events = events;
    }

    @Transactional
    public ProcessingJobResponse submit(UUID videoId) {
        // Serializes concurrent submissions for the same video. The partial UNIQUE index
        // in V2 is a second line of defense against concurrent active jobs.
        Video video = videos.findByIdForUpdate(videoId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Video not found"));

        if (video.getStatus() != VideoStatus.UPLOADED && video.getStatus() != VideoStatus.FAILED) {
            throw new ResponseStatusException(HttpStatus.CONFLICT,
                    "Only UPLOADED or FAILED videos can be submitted");
        }
        if (jobs.existsByVideoIdAndStatusIn(videoId, ACTIVE)) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "Video already has an active job");
        }

        ProcessingJob job = jobs.save(new ProcessingJob(UUID.randomUUID(), videoId, Instant.now()));
        // The listener enqueues only AFTER the DB transaction commits.
        events.publishEvent(new ProcessingRequested(job.getId()));
        return ProcessingJobResponse.from(job);
    }

    @Transactional(readOnly = true)
    public ProcessingJobResponse latest(UUID videoId) {
        if (!videos.existsById(videoId)) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Video not found");
        }
        return jobs.findFirstByVideoIdOrderByCreatedAtDesc(videoId)
                .map(ProcessingJobResponse::from)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND,
                        "No processing job for this video"));
    }
}
