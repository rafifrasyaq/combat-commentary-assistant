package com.combatassistant.backend.processing;

import org.springframework.stereotype.Component;
import org.springframework.transaction.event.TransactionPhase;
import org.springframework.transaction.event.TransactionalEventListener;

@Component
public class ProcessingEventHandler {

    private final ProcessingWorker worker;

    public ProcessingEventHandler(ProcessingWorker worker) {
        this.worker = worker;
    }

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void onSubmitted(ProcessingRequested event) {
        worker.processAsync(event.jobId());
    }
}
