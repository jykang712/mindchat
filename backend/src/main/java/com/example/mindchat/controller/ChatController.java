package com.example.mindchat.controller;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/chat")
public class ChatController {

    @PostMapping("/messages")
    public ChatResponse reply(@Valid @RequestBody ChatRequest request) {
        return new ChatResponse(
            "이야기해 줘서 고마워요. 지금 느끼는 감정을 골라볼까요?",
            "불안",
            "DEMO"
        );
    }

    public record ChatRequest(
        @NotBlank @Size(max = 1000) String message
    ) {}

    public record ChatResponse(
        String reply,
        String suggestedEmotion,
        String mode
    ) {}
}