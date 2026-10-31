package com.combatassistant.backend.processing;

import java.util.UUID;

public record ProcessingInput(UUID jobId, UUID videoId, String storedPath) { }
