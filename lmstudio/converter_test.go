package lmstudio

import "testing"

func TestConvertGoTemplateToJinja(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "Basic Prompt",
			input:    "{{ .Prompt }}",
			expected: "{{ user_message }}",
		},
		{
			name:     "System and Prompt",
			input:    "{{ .System }} {{ .Prompt }}",
			expected: "{{ system_message }} {{ user_message }}",
		},
		{
			name:     "Conditional System",
			input:    "{{ if .System }}System: {{ .System }}{{ end }} User: {{ .Prompt }}",
			expected: "{% if system_message %}System: {{ system_message }}{% endif %} User: {{ user_message }}",
		},
		{
			name:     "GLM-4 Style",
			input:    "[gMASK]<sop><|user|> \n {{ .Prompt }} <|assistant|> ",
			expected: "[gMASK]<sop><|user|> \n {{ user_message }} <|assistant|> ",
		},
		{
			name:     "Nothink Variant",
			input:    "<|user|>{{ .Prompt }}<|assistant|><think>\n</think>",
			expected: "<|user|>{{ user_message }}<|assistant|><think>\n</think>",
		},
		{
			name:     "Complex Spacing",
			input:    "{{if .System}} {{.System}} {{end}}",
			expected: "{% if system_message %} {{ system_message }} {% endif %}",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertGoTemplateToJinja(tt.input)
			if result != tt.expected {
				t.Errorf("convertGoTemplateToJinja() = %v, want %v", result, tt.expected)
			}
		})
	}
}
