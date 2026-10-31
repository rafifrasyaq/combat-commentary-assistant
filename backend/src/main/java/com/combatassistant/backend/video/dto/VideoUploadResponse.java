package com.combatassistant.backend.video.dto;

import com.combatassistant.backend.video.VideoStatus;

import java.util.UUID;

public record VideoUploadResponse(
        UUID videoId,
        String filename,
        VideoStatus status
) {
}
