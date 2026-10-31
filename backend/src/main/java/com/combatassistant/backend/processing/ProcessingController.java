package com.combatassistant.backend.processing;

import com.combatassistant.backend.processing.dto.ProcessingJobResponse;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.UUID;

@RestController
@RequestMapping("/api/videos/{videoId}/processing")
public class ProcessingController {

    private final ProcessingJobService jobs;

    public ProcessingController(ProcessingJobService jobs) { this.jobs = jobs; }

    @PostMapping
    public ResponseEntity<ProcessingJobResponse> submit(@PathVariable UUID videoId) {
        return ResponseEntity.status(HttpStatus.ACCEPTED).body(jobs.submit(videoId));
    }

    @GetMapping
    public ProcessingJobResponse latest(@PathVariable UUID videoId) {
        return jobs.latest(videoId);
    }
}
