package com.example.mindchat.controller;

import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Set;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;

import org.springframework.http.HttpStatus;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

@RestController
@RequestMapping("/api/mood-entries")
public class MoodEntryController {

    private static final Set<String> ALLOWED_EMOTIONS = Set.of(
        "기쁨", "불안", "슬픔", "분노", "당황", "상처", "중립", "잘 모르겠어요"
    );

    private final JdbcTemplate jdbcTemplate;

    public MoodEntryController(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public void save(@Valid @RequestBody CreateMoodEntryRequest request) {
        if (!ALLOWED_EMOTIONS.contains(request.confirmedEmotion())) {
            throw new ResponseStatusException(
                HttpStatus.BAD_REQUEST, "선택할 수 없는 감정입니다."
            );
        }

        jdbcTemplate.update("""
            INSERT INTO mood_entries (confirmed_emotion, created_at)
            VALUES (?, UTC_TIMESTAMP(3))
            """, request.confirmedEmotion());
    }

    @GetMapping
    public List<MoodEntryResponse> findRecent() {
        return jdbcTemplate.query("""
            SELECT id, confirmed_emotion, created_at
            FROM mood_entries
            ORDER BY id DESC
            LIMIT 20
            """, (row, rowNumber) -> new MoodEntryResponse(
                row.getLong("id"),
                row.getString("confirmed_emotion"),
                row.getObject("created_at", LocalDateTime.class)
                    .toInstant(ZoneOffset.UTC)
                    .toString()
            ));
    }

    public record CreateMoodEntryRequest(@NotBlank String confirmedEmotion) {}

    public record MoodEntryResponse(long id, String confirmedEmotion, String createdAt) {}
}