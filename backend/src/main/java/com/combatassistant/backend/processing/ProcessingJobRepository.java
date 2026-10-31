package com.combatassistant.backend.processing;

import jakarta.persistence.LockModeType;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface ProcessingJobRepository extends JpaRepository<ProcessingJob, UUID> {

    boolean existsByVideoIdAndStatusIn(UUID videoId, Collection<ProcessingStatus> statuses);

    Optional<ProcessingJob> findFirstByVideoIdOrderByCreatedAtDesc(UUID videoId);

    List<ProcessingJob> findByStatus(ProcessingStatus status);

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("select j from ProcessingJob j where j.id = :id")
    Optional<ProcessingJob> findByIdForUpdate(@Param("id") UUID id);
}
