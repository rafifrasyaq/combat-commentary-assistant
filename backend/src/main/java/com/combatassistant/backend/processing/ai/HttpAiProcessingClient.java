package com.combatassistant.backend.processing.ai;

import com.combatassistant.backend.processing.ProcessingInput;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.net.http.HttpClient;
import java.time.Duration;
import java.util.UUID;

@Component
public class HttpAiProcessingClient implements AiProcessingClient {

    private final RestClient http;

    public HttpAiProcessingClient(@Value("${app.ai.service-url}") String baseUrl) {
        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();
        JdkClientHttpRequestFactory requestFactory = new JdkClientHttpRequestFactory(client);
        requestFactory.setReadTimeout(Duration.ofSeconds(120));
        this.http = RestClient.builder()
                .baseUrl(baseUrl)
                .requestFactory(requestFactory)
                .build();
    }

    @Override
    public AiProcessingResponse analyze(ProcessingInput input) {
        AiProcessingResponse response = http.post()
                .uri("/analyze")
                .body(new AiProcessingRequest(input.jobId(), input.videoId(), input.storedPath()))
                .retrieve()
                .body(AiProcessingResponse.class);
        if (response == null || response.summary() == null || response.summary().isBlank() ||
                response.summary().length() > 1000 || response.fileSizeBytes() <= 0 ||
                response.manifestPath() == null || response.manifestPath().isBlank() ||
                response.manifestPath().length() > 2048) {
            throw new IllegalStateException("Invalid AI service response");
        }
        return response;
    }

    private record AiProcessingRequest(UUID jobId, UUID videoId, String storagePath) { }
}
