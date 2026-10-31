package com.combatassistant.backend.processing;

import com.combatassistant.backend.video.Video;
import com.combatassistant.backend.video.VideoRepository;
import com.combatassistant.backend.video.VideoStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class ProcessingJobServiceTests {

    private final VideoRepository videos = mock(VideoRepository.class);
    private final ProcessingJobRepository jobs = mock(ProcessingJobRepository.class);
    private final ApplicationEventPublisher events = mock(ApplicationEventPublisher.class);
    private final ProcessingJobService service = new ProcessingJobService(videos, jobs, events);
    private UUID videoId;
    private Video video;

    @BeforeEach
    void setUp() {
        videoId = UUID.randomUUID();
        video = new Video();
        video.setId(videoId);
        video.setStatus(VideoStatus.UPLOADED);
        when(videos.findByIdForUpdate(videoId)).thenReturn(Optional.of(video));
        when(jobs.save(any(ProcessingJob.class))).thenAnswer(invocation -> invocation.getArgument(0));
    }

    @Test
    void submitCreatesQueuedJobAndPublishesEvent() {
        var result = service.submit(videoId);
        assertEquals(ProcessingStatus.QUEUED, result.status());
        assertEquals(videoId, result.videoId());
        verify(jobs).save(any(ProcessingJob.class));
        verify(events).publishEvent(any(ProcessingRequested.class));
    }

    @Test
    void duplicateActiveJobRejected() {
        when(jobs.existsByVideoIdAndStatusIn(eq(videoId), anyList())).thenReturn(true);
        assertEquals(409, assertThrows(ResponseStatusException.class,
                () -> service.submit(videoId)).getStatusCode().value());
        verify(jobs, never()).save(any());
    }

    @Test
    void completedVideoCannotBeResubmitted() {
        video.setStatus(VideoStatus.COMPLETED);
        assertEquals(409, assertThrows(ResponseStatusException.class,
                () -> service.submit(videoId)).getStatusCode().value());
    }

    @Test
    void failedVideoCanBeRetried() {
        video.setStatus(VideoStatus.FAILED);
        assertEquals(ProcessingStatus.QUEUED, service.submit(videoId).status());
    }

    @Test
    void missingVideoRejected() {
        when(videos.findByIdForUpdate(videoId)).thenReturn(Optional.empty());
        assertEquals(404, assertThrows(ResponseStatusException.class,
                () -> service.submit(videoId)).getStatusCode().value());
    }
}
