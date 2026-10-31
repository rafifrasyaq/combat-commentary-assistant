package com.combatassistant.backend.video;

import com.combatassistant.backend.video.dto.VideoUploadResponse;
import com.combatassistant.backend.video.storage.VideoStorage;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.time.Instant;
import java.util.UUID;

@Service
public class VideoService {

    private final VideoRepository videoRepository;
    private final VideoStorage videoStorage;

    public VideoService(
            VideoRepository videoRepository,
            VideoStorage videoStorage) {

        this.videoRepository = videoRepository;
        this.videoStorage = videoStorage;
    }

    public VideoUploadResponse upload(MultipartFile file) {

        validateVideo(file);

        UUID videoId = UUID.randomUUID();

        String storedPath =
                videoStorage.store(videoId, file);

        Video video = new Video();

        video.setId(videoId);
        video.setOriginalFilename(file.getOriginalFilename());
        video.setStoredPath(storedPath);
        video.setContentType(file.getContentType());
        video.setFileSize(file.getSize());
        video.setStatus(VideoStatus.UPLOADED);
        video.setUploadedAt(Instant.now());

        Video savedVideo =
                videoRepository.save(video);

        return new VideoUploadResponse(
                savedVideo.getId(),
                savedVideo.getOriginalFilename(),
                savedVideo.getStatus()
        );
    }

    private void validateVideo(MultipartFile file) {

        if (file == null || file.isEmpty()) {
            throw new IllegalArgumentException(
                    "Video file cannot be empty");
        }

        String contentType = file.getContentType();

        if (contentType == null ||
                !contentType.startsWith("video/")) {

            throw new IllegalArgumentException(
                    "Only video files are allowed");
        }
    }
}
