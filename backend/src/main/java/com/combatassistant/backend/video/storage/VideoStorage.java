package com.combatassistant.backend.video.storage;

import org.springframework.web.multipart.MultipartFile;

import java.util.UUID;

public interface VideoStorage {

    String store(UUID videoId, MultipartFile file);
}
