package com.combatassistant.backend.video.storage;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.UUID;

@Service
public class LocalVideoStorage implements VideoStorage {

    private final Path storageDirectory;

    public LocalVideoStorage(
            @Value("${app.storage.video-dir}") String storageDirectory) {

        this.storageDirectory = Paths.get(storageDirectory)
                .toAbsolutePath()
                .normalize();

        try {
            Files.createDirectories(this.storageDirectory);
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Unable to create video storage directory", e);
        }
    }

    @Override
    public String store(UUID videoId, MultipartFile file) {

        String originalFilename = StringUtils.cleanPath(
                file.getOriginalFilename() != null
                        ? file.getOriginalFilename()
                        : "video"
        );

        if (originalFilename.contains("..")) {
            throw new IllegalArgumentException("Invalid filename");
        }

        String storedFilename = videoId + "_" + originalFilename;

        Path target = storageDirectory
                .resolve(storedFilename)
                .normalize();

        if (!target.startsWith(storageDirectory)) {
            throw new IllegalArgumentException("Invalid storage path");
        }

        try {
            Files.copy(
                    file.getInputStream(),
                    target,
                    StandardCopyOption.REPLACE_EXISTING
            );

            return target.toString();

        } catch (IOException e) {
            throw new IllegalStateException(
                    "Unable to store video", e);
        }
    }
}
