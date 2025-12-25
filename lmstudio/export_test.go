package lmstudio

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestParseOllamaModelfile(t *testing.T) {
	modelfile := `
TEMPLATE """{{ if .System }}{{ .System }}{{ end }}<think>
</think>{{ .Prompt }}"""
SYSTEM "You are a helpful assistant."
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "User:"
PARAMETER stop "Assistant:"
`
	parsed, err := ParseOllamaModelfile(modelfile)
	if err != nil {
		t.Fatalf("ParseOllamaModelfile failed: %v", err)
	}

	if !strings.Contains(parsed.Template, "<think>") {
		t.Errorf("Template did not contain expected content: %s", parsed.Template)
	}

	if parsed.System != "You are a helpful assistant." {
		t.Errorf("Expected system prompt %q, got %q", "You are a helpful assistant.", parsed.System)
	}

	if len(parsed.Parameters["stop"]) != 2 {
		t.Errorf("Expected 2 stop strings, got %d", len(parsed.Parameters["stop"]))
	}

	if parsed.Parameters["temperature"][0] != "0.7" {
		t.Errorf("Expected temperature 0.7, got %s", parsed.Parameters["temperature"][0])
	}
}

func TestConvertToLMStudioFormat(t *testing.T) {
	parsed := &ParsedModelfile{
		Template: "{{ .Prompt }}",
		System:   "System prompt",
		Parameters: map[string][]string{
			"temperature": {"0.8"},
			"stop":        {"Stop1", "Stop2"},
			"num_ctx":     {"4096"},
		},
	}

	config, err := ConvertToLMStudioFormat(parsed)
	if err != nil {
		t.Fatalf("ConvertToLMStudioFormat failed: %v", err)
	}

	// Verify JSON structure
	data, _ := json.Marshal(config)
	jsonStr := string(data)

	if !strings.Contains(jsonStr, "System prompt\\n{{ user_message }}") {
		t.Errorf("JSON did not contain prepended system prompt: %s", jsonStr)
	}

	if !strings.Contains(jsonStr, "llm.prediction.temperature") || !strings.Contains(jsonStr, "0.8") {
		t.Errorf("JSON missing temperature: %s", jsonStr)
	}

	if !strings.Contains(jsonStr, "Stop1") || !strings.Contains(jsonStr, "Stop2") {
		t.Errorf("JSON missing stop strings: %s", jsonStr)
	}

	if !strings.Contains(jsonStr, "llm.load.contextLength") || !strings.Contains(jsonStr, "4096") {
		t.Errorf("JSON missing context length: %s", jsonStr)
	}
}
