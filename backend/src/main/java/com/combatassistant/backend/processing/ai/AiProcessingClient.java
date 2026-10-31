package com.combatassistant.backend.processing.ai;

import com.combatassistant.backend.processing.ProcessingInput;

public interface AiProcessingClient {
    AiProcessingResponse analyze(ProcessingInput input);
}
