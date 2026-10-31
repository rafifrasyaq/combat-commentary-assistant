package com.combatassistant.backend.video;

import com.combatassistant.backend.video.dto.VideoUploadResponse;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/videos")
public class VideoController {

    private final VideoService videoService;

    public VideoController(VideoService videoService) {
        this.videoService = videoService;
    }

    @PostMapping(
            consumes = MediaType.MULTIPART_FORM_DATA_VALUE
    )
    public ResponseEntity<VideoUploadResponse> upload(
            @RequestPart("file") MultipartFile file) {

        VideoUploadResponse response =
                videoService.upload(file);

        return ResponseEntity
                .status(HttpStatus.CREATED)
                .body(response);
    }
}
