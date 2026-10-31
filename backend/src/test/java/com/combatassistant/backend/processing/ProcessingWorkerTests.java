package com.combatassistant.backend.processing;

import com.combatassistant.backend.processing.ai.AiProcessingClient;
import com.combatassistant.backend.processing.ai.AiProcessingResponse;
import org.junit.jupiter.api.Test;

import java.util.Optional;
import java.util.UUID;

import static org.mockito.Mockito.*;

class ProcessingWorkerTests {

    @Test
    void happyPathCompletesJob() {
        var state = mock(ProcessingStateService.class);
        var client = mock(AiProcessingClient.class);
        var jobId = UUID.randomUUID();
        var input = new ProcessingInput(jobId, UUID.randomUUID(), "video.mp4");
        when(state.start(jobId)).thenReturn(Optional.of(input));
        when(client.analyze(input)).thenReturn(new AiProcessingResponse("stub", 10L, "Analyzed"));

        new ProcessingWorker(state, client).processAsync(jobId);

        verify(state).succeed(jobId, "Analyzed");
        verify(state, never()).fail(any(), anyString());
    }

    @Test
    void aiFailurePersistsFailedState() {
        var state = mock(ProcessingStateService.class);
        var client = mock(AiProcessingClient.class);
        var jobId = UUID.randomUUID();
        var input = new ProcessingInput(jobId, UUID.randomUUID(), "video.mp4");
        when(state.start(jobId)).thenReturn(Optional.of(input));
        when(client.analyze(input)).thenThrow(new IllegalStateException("Connection refused"));

        new ProcessingWorker(state, client).processAsync(jobId);

        verify(state).fail(eq(jobId), contains("failed"));
        verify(state, never()).succeed(any(), anyString());
    }
}
