package com.combatassistant.backend.processing.ai;

public record AiProcessingResponse(String engine, long fileSizeBytes, String summary) { }
