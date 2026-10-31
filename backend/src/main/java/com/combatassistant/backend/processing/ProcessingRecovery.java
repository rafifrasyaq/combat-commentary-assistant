package com.combatassistant.backend.processing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

@Component
public class ProcessingRecovery {

    private static final Logger log = LoggerFactory.getLogger(ProcessingRecovery.class);
    private final ProcessingJobRepository jobs;
    private final ProcessingStateService state;
    private final ProcessingWorker worker;

    public ProcessingRecovery(ProcessingJobRepository jobs,
                              ProcessingStateService state, ProcessingWorker worker) {
        this.jobs = jobs;
        this.state = state;
        this.worker = worker;
    }

    @EventListener(ApplicationReadyEvent.class)
    public void recover() {
        // Single backend instance milestone: in-flight work interrupted by restart is failed;
        // pending queued jobs are restarted. Multi-instance requires leased durable workers.
        for (ProcessingJob job : jobs.findByStatus(ProcessingStatus.PROCESSING)) {
            log.warn("Marking interrupted job {} as failed", job.getId());
            state.fail(job.getId(), "Processing interrupted by backend restart; resubmit to retry.");
        }
        for (ProcessingJob job : jobs.findByStatus(ProcessingStatus.QUEUED)) {
            worker.processAsync(job.getId());
        }
    }
}
