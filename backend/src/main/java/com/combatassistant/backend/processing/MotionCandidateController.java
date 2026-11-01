package com.combatassistant.backend.processing;

import com.combatassistant.backend.video.VideoRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

/** Returns the generated motion-candidate JSON, not internal filesystem paths. */
@RestController
public class MotionCandidateController {
    private static final long MAX_RESULT_BYTES = 65_536;
    private final ProcessingJobRepository jobs;
    private final VideoRepository videos;
    private final Path artifactRoot;

    public MotionCandidateController(ProcessingJobRepository jobs, VideoRepository videos,
            @Value("${app.storage.analysis-dir:storage/analysis}") String analysisDir) {
        this.jobs = jobs;
        this.videos = videos;
        this.artifactRoot = Path.of(analysisDir).toAbsolutePath().normalize();
    }

    @GetMapping("/api/videos/{videoId}/moments")
    public ResponseEntity<String> latestMotionCandidates(@PathVariable UUID videoId) {
        if (!videos.existsById(videoId)) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Video not found");
        }
        ProcessingJob job = jobs.findFirstByVideoIdOrderByCreatedAtDesc(videoId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "No processing job"));
        if (job.getStatus() != ProcessingStatus.COMPLETED) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "Processing job is not completed");
        }
        Path jobFolder = artifactRoot.resolve(job.getId().toString());
        Path expectedManifest = jobFolder.resolve("manifest.json");
        if (job.getArtifactManifestPath() == null ||
                !Path.of(job.getArtifactManifestPath()).toAbsolutePath().normalize().equals(expectedManifest)) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Motion candidates not available");
        }
        Path result = jobFolder.resolve("motion_candidates.json");
        try {
            if (!Files.isRegularFile(result) || Files.isSymbolicLink(result) ||
                    !result.toRealPath().startsWith(artifactRoot.toRealPath()) ||
                    Files.size(result) > MAX_RESULT_BYTES) {
                throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Motion candidates not available");
            }
            return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON)
                    .body(Files.readString(result, StandardCharsets.UTF_8));
        } catch (IOException exception) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Motion candidates not available", exception);
        }
    }
}
