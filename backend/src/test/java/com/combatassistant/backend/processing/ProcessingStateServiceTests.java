package com.combatassistant.backend.processing;

import com.combatassistant.backend.video.Video;
import com.combatassistant.backend.video.VideoRepository;
import com.combatassistant.backend.video.VideoStatus;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class ProcessingStateServiceTests {

    @Test
    void validJobTransitionsUpdateVideoTogether() {
        ProcessingJobRepository jobs = mock(ProcessingJobRepository.class);
        VideoRepository videos = mock(VideoRepository.class);
        var videoId = UUID.randomUUID();
        var jobId = UUID.randomUUID();
        var job = new ProcessingJob(jobId, videoId, Instant.now());
        var video = new Video();
        video.setId(videoId);
        video.setStatus(VideoStatus.UPLOADED);
        video.setStoredPath("storage/videos/test.mp4");
        when(jobs.findByIdForUpdate(jobId)).thenReturn(Optional.of(job));
        when(videos.findById(videoId)).thenReturn(Optional.of(video));
        var state = new ProcessingStateService(jobs, videos);

        var input = state.start(jobId).orElseThrow();
        assertEquals(videoId, input.videoId());
        assertEquals(ProcessingStatus.PROCESSING, job.getStatus());
        assertEquals(VideoStatus.PROCESSING, video.getStatus());
        assertTrue(state.start(jobId).isEmpty());

        state.succeed(jobId, "Video verified");
        assertEquals(ProcessingStatus.COMPLETED, job.getStatus());
        assertEquals(VideoStatus.COMPLETED, video.getStatus());
        assertEquals("Video verified", job.getResultSummary());
    }

    @Test
    void failingQueuedJobAllowsLaterRetry() {
        ProcessingJobRepository jobs = mock(ProcessingJobRepository.class);
        VideoRepository videos = mock(VideoRepository.class);
        var videoId = UUID.randomUUID();
        var jobId = UUID.randomUUID();
        var job = new ProcessingJob(jobId, videoId, Instant.now());
        var video = new Video();
        video.setId(videoId);
        video.setStatus(VideoStatus.UPLOADED);
        when(jobs.findByIdForUpdate(jobId)).thenReturn(Optional.of(job));
        when(videos.findById(videoId)).thenReturn(Optional.of(video));
        var state = new ProcessingStateService(jobs, videos);

        state.fail(jobId, "Worker unavailable");
        assertEquals(ProcessingStatus.FAILED, job.getStatus());
        assertEquals(VideoStatus.FAILED, video.getStatus());
        assertTrue(state.start(jobId).isEmpty());
    }
}
