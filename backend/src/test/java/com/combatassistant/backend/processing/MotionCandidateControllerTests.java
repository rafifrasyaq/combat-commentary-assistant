package com.combatassistant.backend.processing;

import com.combatassistant.backend.video.VideoRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class MotionCandidateControllerTests {
    @TempDir Path root;

    @Test
    void returnsBoundedMotionJsonForCompletedJob() throws IOException {
        var jobs = mock(ProcessingJobRepository.class);
        var videos = mock(VideoRepository.class);
        var videoId = UUID.randomUUID();
        var jobId = UUID.randomUUID();
        Path jobDir = Files.createDirectories(root.resolve(jobId.toString()));
        Path manifest = jobDir.resolve("manifest.json");
        Files.writeString(manifest, "{}", StandardCharsets.UTF_8);
        Files.writeString(jobDir.resolve("motion_candidates.json"), "{\"motionCandidates\":[]}", StandardCharsets.UTF_8);
        var job = new ProcessingJob(jobId, videoId, Instant.now());
        job.markProcessing(Instant.now());
        job.markCompleted("Complete", manifest.toString(), Instant.now());
        when(videos.existsById(videoId)).thenReturn(true);
        when(jobs.findFirstByVideoIdOrderByCreatedAtDesc(videoId)).thenReturn(Optional.of(job));
        var result = new MotionCandidateController(jobs, videos, root.toString())
                .latestMotionCandidates(videoId);
        assertEquals(HttpStatus.OK, result.getStatusCode());
        assertEquals("{\"motionCandidates\":[]}", result.getBody());
    }

    @Test
    void missingVideoReturns404() {
        var jobs = mock(ProcessingJobRepository.class);
        var videos = mock(VideoRepository.class);
        var id = UUID.randomUUID();
        var error = assertThrows(ResponseStatusException.class, () ->
                new MotionCandidateController(jobs, videos, root.toString())
                        .latestMotionCandidates(id));
        assertEquals(HttpStatus.NOT_FOUND, error.getStatusCode());
    }

    @Test
    void incompleteJobReturns409() {
        var jobs = mock(ProcessingJobRepository.class);
        var videos = mock(VideoRepository.class);
        var id = UUID.randomUUID();
        when(videos.existsById(id)).thenReturn(true);
        when(jobs.findFirstByVideoIdOrderByCreatedAtDesc(id))
                .thenReturn(Optional.of(new ProcessingJob(UUID.randomUUID(), id, Instant.now())));
        var error = assertThrows(ResponseStatusException.class, () ->
                new MotionCandidateController(jobs, videos, root.toString())
                        .latestMotionCandidates(id));
        assertEquals(HttpStatus.CONFLICT, error.getStatusCode());
    }
}
