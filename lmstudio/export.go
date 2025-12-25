package lmstudio

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/sammcj/gollama/logging"
)

// LMStudioConfig represents the complete configuration for a model
type LMStudioConfig struct {
	PredictionConfig PredictionConfig `json:"predictionConfig"`
	LoadModelConfig  LoadModelConfig  `json:"loadModelConfig"`
}

// PredictionConfig contains inference-time parameters
type PredictionConfig struct {
	Fields []ConfigField `json:"fields"`
}

// LoadModelConfig contains model loading parameters
type LoadModelConfig struct {
	Fields []ConfigField `json:"fields"`
}

// ConfigField represents a key-value configuration field
type ConfigField struct {
	Key   string      `json:"key"`
	Value interface{} `json:"value"`
}

// JinjaPromptTemplate represents LM Studio's Jinja template format
type JinjaPromptTemplate struct {
	Type                string                 `json:"type"`
	JinjaPromptTemplate JinjaTemplateDetails   `json:"jinjaPromptTemplate"`
	StopStrings         []string               `json:"stopStrings"`
}

// JinjaTemplateDetails contains the actual template content
type JinjaTemplateDetails struct {
	Template    string      `json:"template"`
	BosToken    string      `json:"bosToken"`
	EosToken    string      `json:"eosToken"`
	InputConfig InputConfig `json:"inputConfig"`
}

// InputConfig defines input format configuration
type InputConfig struct {
	MessagesConfig MessagesConfig `json:"messagesConfig"`
	UseTools       bool           `json:"useTools"`
}

// MessagesConfig defines message content configuration
type MessagesConfig struct {
	ContentConfig ContentConfig `json:"contentConfig"`
}

// ContentConfig defines content type
type ContentConfig struct {
	Type string `json:"type"`
}

// ParsedModelfile contains extracted Modelfile components
type ParsedModelfile struct {
	Template   string
	System     string
	Parameters map[string][]string // Parameter name -> values (supports multiple values like stop)
}

// ExportModelConfig exports Ollama Modelfile configuration to LM Studio format
func ExportModelConfig(modelName, outputPath string, client *api.Client) error {
	// Get the Modelfile from Ollama
	ctx := context.Background()
	req := &api.ShowRequest{
		Name: modelName,
	}

	resp, err := client.Show(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to get model info: %w", err)
	}

	// Parse the Modelfile
	parsed, err := ParseOllamaModelfile(resp.Modelfile)
	if err != nil {
		logging.ErrorLogger.Printf("Warning: Failed to parse Modelfile for %s: %v\n", modelName, err)
		// Continue with defaults
		parsed = &ParsedModelfile{
			Parameters: make(map[string][]string),
		}
	}

	// Convert to LM Studio format
	config, err := ConvertToLMStudioFormat(parsed)
	if err != nil {
		return fmt.Errorf("failed to convert config: %w", err)
	}

	// Write the configuration file
	return WriteLMStudioConfig(config, outputPath)
}

// LMStudioPreset represents the LM Studio preset format (new v0.3+ format)
type LMStudioPreset struct {
	Identifier string              `json:"identifier"`
	Name       string              `json:"name"`
	Changed    bool                `json:"changed"`
	Operation  PresetOperation     `json:"operation"`
	Load       PresetLoad          `json:"load"`
}

// PresetOperation contains prediction configuration fields
type PresetOperation struct {
	Fields []PresetField `json:"fields"`
}

// PresetLoad contains model loading configuration fields
type PresetLoad struct {
	Fields []PresetField `json:"fields"`
}

// PresetField represents a configuration field
type PresetField struct {
	Key   string      `json:"key"`
	Value interface{} `json:"value"`
}

// ManualPromptTemplateValue represents the prompt template configuration
type ManualPromptTemplateValue struct {
	Type                  string                `json:"type"`
	StopStrings           []string              `json:"stopStrings"`
	ManualPromptTemplate  ManualPromptTemplate  `json:"manualPromptTemplate"`
}

// ManualPromptTemplate defines the template structure
type ManualPromptTemplate struct {
	BeforeSystem    string `json:"beforeSystem"`
	AfterSystem     string `json:"afterSystem"`
	BeforeUser      string `json:"beforeUser"`
	AfterUser       string `json:"afterUser"`
	BeforeAssistant string `json:"beforeAssistant"`
	AfterAssistant  string `json:"afterAssistant"`
}

// LMStudioPresetLegacy represents the old LM Studio preset format (deprecated)
type LMStudioPresetLegacy struct {
	Name            string               `json:"name"`
	InferenceParams InferenceParams      `json:"inference_params"`
}

// InferenceParams contains LM Studio's legacy preset parameters
type InferenceParams struct {
	InputPrefix      string   `json:"input_prefix"`
	InputSuffix      string   `json:"input_suffix"`
	PrePrompt        string   `json:"pre_prompt,omitempty"`
	PrePromptPrefix  string   `json:"pre_prompt_prefix,omitempty"`
	PrePromptSuffix  string   `json:"pre_prompt_suffix,omitempty"`
	Antiprompt       []string `json:"antiprompt,omitempty"`
}

// ExportModelPreset exports Ollama Modelfile as an LM Studio preset
// to ~/.lmstudio/config-presets/ for manual loading via LM Studio UI
func ExportModelPreset(modelName, lmStudioModelName string, client *api.Client) error {
	// Get the Modelfile from Ollama
	ctx := context.Background()
	req := &api.ShowRequest{
		Name: modelName,
	}

	resp, err := client.Show(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to get model info: %w", err)
	}

	// Parse the Modelfile
	parsed, err := ParseOllamaModelfile(resp.Modelfile)
	if err != nil {
		logging.ErrorLogger.Printf("Warning: Failed to parse Modelfile for %s: %v\n", modelName, err)
		// Continue with defaults
		parsed = &ParsedModelfile{
			Parameters: make(map[string][]string),
		}
	}

	// Convert to LM Studio preset format
	preset, err := ConvertToLMStudioPreset(parsed, lmStudioModelName)
	if err != nil {
		return fmt.Errorf("failed to convert to preset: %w", err)
	}

	// Determine preset directory path
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get home directory: %w", err)
	}

	presetDir := fmt.Sprintf("%s/.lmstudio/config-presets", homeDir)

	// Ensure directory exists
	if err := os.MkdirAll(presetDir, 0755); err != nil {
		return fmt.Errorf("failed to create preset directory: %w", err)
	}

	// Write preset file
	presetPath := fmt.Sprintf("%s/%s.preset.json", presetDir, lmStudioModelName)
	return WriteLMStudioPreset(preset, presetPath)
}

// ParseOllamaModelfile extracts TEMPLATE, SYSTEM, and PARAMETER directives
func ParseOllamaModelfile(modelfileContent string) (*ParsedModelfile, error) {
	parsed := &ParsedModelfile{
		Parameters: make(map[string][]string),
	}

	if modelfileContent == "" {
		return parsed, nil
	}

	lines := strings.Split(modelfileContent, "\n")
	var templateLines []string
	var systemLines []string
	inTemplate := false
	inMultilineTemplate := false
	inDoubleQuotedTemplate := false
	inSystem := false
	inMultilineSystem := false
	inDoubleQuotedSystem := false

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Handle TEMPLATE directive
		if strings.HasPrefix(trimmed, "TEMPLATE") && !inTemplate {
			if strings.Contains(trimmed, `"""`) {
				// Triple-quoted multi-line template
				templateContent := strings.TrimPrefix(trimmed, "TEMPLATE ")
				templateContent = strings.TrimSpace(templateContent)
				if strings.HasPrefix(templateContent, `"""`) {
					templateContent = strings.TrimPrefix(templateContent, `"""`)
				}
				inTemplate = true
				inMultilineTemplate = true
				if templateContent != "" {
					templateLines = append(templateLines, templateContent)
				}
			} else if strings.HasPrefix(trimmed, `TEMPLATE "`) {
				// Could be single-line or double-quoted multi-line
				templateContent := strings.TrimPrefix(trimmed, "TEMPLATE ")
				templateContent = strings.TrimSpace(templateContent)
				templateContent = strings.TrimPrefix(templateContent, `"`)

				// Check if closing quote is on the same line
				if strings.HasSuffix(templateContent, `"`) {
					// Single-line template
					parsed.Template = strings.TrimSuffix(templateContent, `"`)
				} else {
					// Multi-line double-quoted template
					inTemplate = true
					inDoubleQuotedTemplate = true
					if templateContent != "" {
						templateLines = append(templateLines, templateContent)
					}
				}
			}
		} else if inTemplate {
			if inMultilineTemplate && strings.HasSuffix(trimmed, `"""`) {
				// End of triple-quoted template
				line = strings.TrimSuffix(line, `"""`)
				if line != "" {
					templateLines = append(templateLines, line)
				}
				inTemplate = false
				inMultilineTemplate = false
			} else if inDoubleQuotedTemplate && strings.HasSuffix(trimmed, `"`) {
				// End of double-quoted template
				line = strings.TrimSuffix(line, `"`)
				if line != "" {
					templateLines = append(templateLines, line)
				}
				inTemplate = false
				inDoubleQuotedTemplate = false
			} else {
				// Continue collecting template lines
				templateLines = append(templateLines, line)
			}
		} else if strings.HasPrefix(trimmed, "SYSTEM") && !inSystem {
			// Handle SYSTEM directive (similar to TEMPLATE)
			if strings.Contains(trimmed, `"""`) {
				// Triple-quoted multi-line system
				systemContent := strings.TrimPrefix(trimmed, "SYSTEM ")
				systemContent = strings.TrimSpace(systemContent)
				if strings.HasPrefix(systemContent, `"""`) {
					systemContent = strings.TrimPrefix(systemContent, `"""`)
				}
				inSystem = true
				inMultilineSystem = true
				if systemContent != "" {
					systemLines = append(systemLines, systemContent)
				}
			} else if strings.HasPrefix(trimmed, `SYSTEM "`) {
				// Could be single-line or double-quoted multi-line
				systemContent := strings.TrimPrefix(trimmed, "SYSTEM ")
				systemContent = strings.TrimSpace(systemContent)
				systemContent = strings.TrimPrefix(systemContent, `"`)

				// Check if closing quote is on the same line
				if strings.HasSuffix(systemContent, `"`) {
					// Single-line system
					parsed.System = strings.TrimSuffix(systemContent, `"`)
				} else {
					// Multi-line double-quoted system
					inSystem = true
					inDoubleQuotedSystem = true
					if systemContent != "" {
						systemLines = append(systemLines, systemContent)
					}
				}
			}
		} else if inSystem {
			if inMultilineSystem && strings.HasSuffix(trimmed, `"""`) {
				// End of triple-quoted system
				line = strings.TrimSuffix(line, `"""`)
				if line != "" {
					systemLines = append(systemLines, line)
				}
				inSystem = false
				inMultilineSystem = false
			} else if inDoubleQuotedSystem && strings.HasSuffix(trimmed, `"`) {
				// End of double-quoted system
				line = strings.TrimSuffix(line, `"`)
				if line != "" {
					systemLines = append(systemLines, line)
				}
				inSystem = false
				inDoubleQuotedSystem = false
			} else {
				// Continue collecting system lines
				systemLines = append(systemLines, line)
			}
		} else if strings.HasPrefix(trimmed, "PARAMETER ") {
			// Parse parameter directive
			paramStr := strings.TrimPrefix(trimmed, "PARAMETER ")
			parts := strings.SplitN(paramStr, " ", 2)
			if len(parts) == 2 {
				key := parts[0]
				value := strings.Trim(parts[1], `"`)
				parsed.Parameters[key] = append(parsed.Parameters[key], value)
			}
		}
	}

	if len(templateLines) > 0 {
		parsed.Template = strings.Join(templateLines, "\n")
	}

	if len(systemLines) > 0 {
		parsed.System = strings.Join(systemLines, "\n")
	}

	return parsed, nil
}

// ConvertToLMStudioFormat maps Ollama config to LM Studio JSON structure
func ConvertToLMStudioFormat(parsed *ParsedModelfile) (*LMStudioConfig, error) {
	config := &LMStudioConfig{
		PredictionConfig: PredictionConfig{Fields: []ConfigField{}},
		LoadModelConfig:  LoadModelConfig{Fields: []ConfigField{}},
	}

	// Handle template if present
	if parsed.Template != "" {
		// Extract BOS and EOS tokens from template (heuristic)
		bosToken := extractBOSToken(parsed.Template)
		eosToken := extractEOSToken(parsed.Template)

		// Get stop strings from parameters
		stopStrings := parsed.Parameters["stop"]

		// Convert Go template to Jinja for LM Studio
		// IMPORTANT: Do NOT prepend system prompt - it should be handled separately
		jinjaTemplate := convertGoTemplateToJinja(parsed.Template)

		// Log the template for debugging
		logging.DebugLogger.Printf("Original template length: %d characters\n", len(parsed.Template))
		logging.DebugLogger.Printf("Converted template length: %d characters\n", len(jinjaTemplate))

		templateValue := JinjaPromptTemplate{
			Type: "jinja",
			JinjaPromptTemplate: JinjaTemplateDetails{
				Template:    jinjaTemplate,
				BosToken:    bosToken,
				EosToken:    eosToken,
				InputConfig: InputConfig{
					MessagesConfig: MessagesConfig{
						ContentConfig: ContentConfig{
							Type: "string",
						},
					},
					UseTools: false,
				},
			},
			StopStrings: stopStrings,
		}

		config.PredictionConfig.Fields = append(config.PredictionConfig.Fields, ConfigField{
			Key:   "llm.prediction.promptTemplate",
			Value: templateValue,
		})
	}

	// Map parameters
	for paramKey, values := range parsed.Parameters {
		if paramKey == "stop" {
			// Already handled in template
			continue
		}

		// Take the first value for single-value parameters
		if len(values) > 0 {
			value := values[0]

			switch paramKey {
			case "temperature":
				if temp, err := strconv.ParseFloat(value, 64); err == nil {
					config.PredictionConfig.Fields = append(config.PredictionConfig.Fields, ConfigField{
						Key:   "llm.prediction.temperature",
						Value: temp,
					})
				}
			case "top_p":
				if topP, err := strconv.ParseFloat(value, 64); err == nil {
					config.PredictionConfig.Fields = append(config.PredictionConfig.Fields, ConfigField{
						Key:   "llm.prediction.topP",
						Value: topP,
					})
				}
			case "top_k":
				if topK, err := strconv.Atoi(value); err == nil {
					config.PredictionConfig.Fields = append(config.PredictionConfig.Fields, ConfigField{
						Key:   "llm.prediction.topK",
						Value: topK,
					})
				}
			case "repeat_penalty":
				if penalty, err := strconv.ParseFloat(value, 64); err == nil {
					config.PredictionConfig.Fields = append(config.PredictionConfig.Fields, ConfigField{
						Key:   "llm.prediction.repeatPenalty",
						Value: penalty,
					})
				}
			case "min_p":
				if minP, err := strconv.ParseFloat(value, 64); err == nil {
					config.PredictionConfig.Fields = append(config.PredictionConfig.Fields, ConfigField{
						Key:   "llm.prediction.minP",
						Value: minP,
					})
				}
			case "num_ctx":
				if ctx, err := strconv.Atoi(value); err == nil {
					config.LoadModelConfig.Fields = append(config.LoadModelConfig.Fields, ConfigField{
						Key:   "llm.load.contextLength",
						Value: ctx,
					})
				}
			default:
				logging.DebugLogger.Printf("Warning: Unsupported parameter '%s' (value: %s) - skipping\n", paramKey, value)
			}
		}
	}

	return config, nil
}

// WriteLMStudioConfig writes the configuration to a JSON file
func WriteLMStudioConfig(config *LMStudioConfig, path string) error {
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	err = os.WriteFile(path, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// ConvertToLMStudioPreset converts parsed Modelfile to LM Studio preset format (new v0.3+ format)
func ConvertToLMStudioPreset(parsed *ParsedModelfile, modelName string) (*LMStudioPreset, error) {
	// Generate identifier from model name
	identifier := "@local:" + strings.ToLower(strings.ReplaceAll(modelName, " ", "-"))

	preset := &LMStudioPreset{
		Identifier: identifier,
		Name:       modelName,
		Changed:    true,
		Operation:  PresetOperation{Fields: []PresetField{}},
		Load:       PresetLoad{Fields: []PresetField{}},
	}

	if parsed.Template == "" {
		// No template - use simple defaults
		return preset, nil
	}

	// Extract components from the template
	template := parsed.Template

	// Extract BOS token (beginning of sequence)
	bosToken := extractBOSToken(template)

	// Get stop strings from parameters
	stopStrings := parsed.Parameters["stop"]
	if stopStrings == nil {
		stopStrings = []string{}
	}

	// Parse the template structure
	// Common pattern: [BOS]<|system|>{{ .System }}<|user|>{{ .Prompt }}<|assistant|>[response]

	// Find system marker
	beforeSystem := ""
	afterSystem := ""
	if strings.Contains(template, "<|system|>") {
		beforeSystem = bosToken + "<|system|>\n"
		afterSystem = ""
	}

	// Find user marker
	beforeUser := ""
	if strings.Contains(template, "<|user|>") {
		beforeUser = "<|user|>\n"
	}

	// Find assistant marker
	afterUser := ""
	if strings.Contains(template, "<|assistant|>") {
		afterUser = "<|assistant|>\n"
	}

	// Check if template has <think></think> tags
	hasThinkTags := strings.Contains(template, "<think></think>")
	if hasThinkTags {
		afterUser += "<think></think>\n"
	}

	// Build prompt template field
	templateValue := ManualPromptTemplateValue{
		Type:        "manual",
		StopStrings: []string{}, // Empty in template, set separately
		ManualPromptTemplate: ManualPromptTemplate{
			BeforeSystem:    beforeSystem,
			AfterSystem:     afterSystem,
			BeforeUser:      beforeUser,
			AfterUser:       afterUser,
			BeforeAssistant: "",
			AfterAssistant:  "",
		},
	}

	preset.Operation.Fields = append(preset.Operation.Fields, PresetField{
		Key:   "llm.prediction.promptTemplate",
		Value: templateValue,
	})

	// Add stop strings as separate field
	preset.Operation.Fields = append(preset.Operation.Fields, PresetField{
		Key:   "llm.prediction.stopStrings",
		Value: stopStrings,
	})

	// Add system prompt if available
	// IMPORTANT: For nothink variants, skip the system prompt as it often emphasizes
	// brevity which causes overly short responses. The pre-filled <think></think> tags
	// are sufficient to suppress thinking without additional instructions.
	if parsed.System != "" && !hasThinkTags {
		preset.Operation.Fields = append(preset.Operation.Fields, PresetField{
			Key:   "llm.prediction.systemPrompt",
			Value: parsed.System,
		})
	}

	return preset, nil
}

// WriteLMStudioPreset writes a preset to a JSON file
func WriteLMStudioPreset(preset *LMStudioPreset, path string) error {
	data, err := json.MarshalIndent(preset, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal preset JSON: %w", err)
	}

	err = os.WriteFile(path, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write preset file: %w", err)
	}

	return nil
}

// extractBOSToken attempts to extract the BOS token from the template
func extractBOSToken(template string) string {
	// Common BOS tokens
	bosTokens := []string{"[gMASK]", "<s>", "<|begin_of_text|>", "<|im_start|>"}
	for _, token := range bosTokens {
		if strings.HasPrefix(template, token) {
			return token
		}
	}
	return ""
}

// extractEOSToken attempts to extract the EOS token from the template
func extractEOSToken(template string) string {
	// Common EOS tokens
	eosTokens := []string{"<|endoftext|>", "</s>", "<|end_of_text|>", "<|im_end|>", "<|user|>"}
	for _, token := range eosTokens {
		if strings.Contains(template, token) {
			return token
		}
	}
	return ""
}

// convertGoTemplateToJinja converts Ollama's Go template syntax to LM Studio's Jinja syntax
func convertGoTemplateToJinja(goTemplate string) string {
	// Replace variables
	// {{ .System }} -> {{ system_message }}
	// {{ .Prompt }} -> {{ user_message }}
	// {{ .Response }} -> {{ model_response }} (though usually implied)
	
	result := goTemplate
	
	// Handle conditional blocks first
	// {{ if .System }} -> {% if system_message %}
	// {{if .System}} -> {% if system_message %}
	reIfSystem := regexp.MustCompile(`{{\s*if\s+\.System\s*}}`)
	result = reIfSystem.ReplaceAllString(result, "{% if system_message %}")

	// {{ if .Prompt }} -> {% if user_message %}
	// {{if .Prompt}} -> {% if user_message %}
	reIfPrompt := regexp.MustCompile(`{{\s*if\s+\.Prompt\s*}}`)
	result = reIfPrompt.ReplaceAllString(result, "{% if user_message %}")

	// {{ end }} -> {% endif %}
	// {{end}} -> {% endif %}
	reEnd := regexp.MustCompile(`{{\s*end\s*}}`)
	result = reEnd.ReplaceAllString(result, "{% endif %}")
	
	// Replace variable tags
	reSystem := regexp.MustCompile(`{{\s*\.System\s*}}`)
	result = reSystem.ReplaceAllString(result, "{{ system_message }}")
	
	rePrompt := regexp.MustCompile(`{{\s*\.Prompt\s*}}`)
	result = rePrompt.ReplaceAllString(result, "{{ user_message }}")
	
	reResponse := regexp.MustCompile(`{{\s*\.Response\s*}}`)
	result = reResponse.ReplaceAllString(result, "{{ model_response }}")
	
	return result
}
