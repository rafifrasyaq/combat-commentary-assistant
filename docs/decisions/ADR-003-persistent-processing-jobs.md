# ADR-003: Persistent Processing Jobs and Python AI Boundary

**Status:** Accepted for Milestone 2 (single backend instance). **Date:** 2026-10-31.

## Decision

Store a `processing_jobs` row in PostgreSQL for each attempt, separate from the existing `videos` table. Expose an internal processing submit endpoint and status polling endpoint. After the job transaction commits, dispatch it to a bounded Spring `@Async` executor, call the Python AI service via a typed HTTP contract, and persist terminal status. Keep Python inference logic out of Spring Boot.

## Responsibilities

- `ProcessingJob`: persistent domain state and valid transitions.
- `ProcessingJobService`: submission, row locking, concurrency rule, job lookup.
- `ProcessingStateService`: atomic job/video status updates in transactions.
- `ProcessingWorker`: asynchronous orchestration and failure handling.
- `AiProcessingClient`: replaceable external service boundary; HTTP adapter is its current implementation.
- `ProcessingRecovery`: on single-instance restart, rerun queued jobs and fail interrupted processing jobs.
- `ai-service`: initial mock service validates access to the video path and returns *file metadata only*.

## Invariants and trade-offs

- Only one active job per video, enforced by row lock plus PostgreSQL partial unique index.
- FAILED videos can be retried with a new job; COMPLETED videos cannot be submitted again in this milestone.
- HTTP API returns `202 Accepted` and the worker runs asynchronously; client polls job status.
- No retry storms: transient external errors mark the job FAILED, and explicit retry is required.
- No timestamps, fighter detection, or generated commentary are faked by the Python mock.
- This is a **single-process development design**, not a multi-replica distributed queue. For horizontal scaling add durable leasing/heartbeats or a message broker, delivery guarantees, and observability.
- Both services currently run on the same host and use shared local video storage; for separate hosts, use object storage with signed access instead of sharing arbitrary filesystem paths.
- Endpoints are not yet authenticated; bind Python to localhost and do not deploy APIs publicly before role-based authorization, rate limits and credential management.
- The manual POST to submit is an internal integration-control endpoint; the final upload-only UX can enqueue automatically after downstream AI behavior is validated.
