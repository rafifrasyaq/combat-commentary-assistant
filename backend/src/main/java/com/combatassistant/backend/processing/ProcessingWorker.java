package com.combatassistant.backend.processing;

import com.combatassistant.backend.processing.ai.AiProcessingClient;
import com.combatassistant.backend.processing.ai.AiProcessingResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;

import java.util.UUID;

@Component
public class ProcessingWorker {

    private static final Logger log = LoggerFactory.getLogger(ProcessingWorker.class);
    private static final String GENERIC_FAILURE =
            "Video processing failed. Check backend logs for the job ID.";

    private final ProcessingStateService state;
    private final AiProcessingClient client;

    public ProcessingWorker(ProcessingStateService state, AiProcessingClient client) {
        this.state = state;
        this.client = client;
    }

    @Async("processingExecutor")
    public void processAsync(UUID jobId) {
        try {
            var input = state.start(jobId);
            if (input.isEmpty()) return;
            AiProcessingResponse result = client.analyze(input.get());
            state.succeed(jobId, result.summary(), result.manifestPath());
        } catch (Exception error) {
            log.error("Processing job {} failed", jobId, error);
            try {
                state.fail(jobId, GENERIC_FAILURE);
            } catch (Exception updateError) {
                log.error("Failed to persist failure for processing job {}", jobId, updateError);
            }
        }
    }
}
