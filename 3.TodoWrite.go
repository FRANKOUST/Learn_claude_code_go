package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const maxOutputLen = 50000

// =========================
// .env loading
// =========================

func loadEnvFile(path string, override bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}

	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "export ") {
			line = strings.TrimSpace(strings.TrimPrefix(line, "export "))
		}

		idx := strings.Index(line, "=")
		if idx <= 0 {
			continue
		}

		key := strings.TrimSpace(line[:idx])
		val := strings.TrimSpace(line[idx+1:])

		if len(val) >= 2 {
			if (val[0] == '"' && val[len(val)-1] == '"') || (val[0] == '\'' && val[len(val)-1] == '\'') {
				val = val[1 : len(val)-1]
			}
		}

		if !override {
			if _, exists := os.LookupEnv(key); exists {
				continue
			}
		}
		_ = os.Setenv(key, val)
	}
}

// =========================
// TodoManager
// =========================

type TodoItem struct {
	ID     string `json:"id"`
	Text   string `json:"text"`
	Status string `json:"status"`
}

type TodoManager struct {
	Items []TodoItem
}

func (t *TodoManager) Update(items []TodoItem) (string, error) {
	if len(items) > 20 {
		return "", errors.New("max 20 todos allowed")
	}

	validated := make([]TodoItem, 0, len(items))
	inProgressCount := 0

	for i, item := range items {
		text := strings.TrimSpace(item.Text)
		status := strings.ToLower(strings.TrimSpace(item.Status))
		itemID := strings.TrimSpace(item.ID)
		if itemID == "" {
			itemID = fmt.Sprintf("%d", i+1)
		}

		if text == "" {
			return "", fmt.Errorf("item %s: text required", itemID)
		}

		switch status {
		case "pending", "in_progress", "completed":
		default:
			return "", fmt.Errorf("item %s: invalid status %q", itemID, status)
		}

		if status == "in_progress" {
			inProgressCount++
		}

		validated = append(validated, TodoItem{
			ID:     itemID,
			Text:   text,
			Status: status,
		})
	}

	if inProgressCount > 1 {
		return "", errors.New("only one task can be in_progress at a time")
	}

	t.Items = validated
	return t.Render(), nil
}

func (t *TodoManager) Render() string {
	if len(t.Items) == 0 {
		return "No todos."
	}

	lines := make([]string, 0, len(t.Items)+1)
	done := 0

	for _, item := range t.Items {
		marker := map[string]string{
			"pending":     "[ ]",
			"in_progress": "[>]",
			"completed":   "[x]",
		}[item.Status]

		if item.Status == "completed" {
			done++
		}

		lines = append(lines, fmt.Sprintf("%s #%s: %s", marker, item.ID, item.Text))
	}

	lines = append(lines, fmt.Sprintf("\n(%d/%d completed)", done, len(t.Items)))
	return strings.Join(lines, "\n")
}

// =========================
// Anthropic API types
// =========================

type ToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type ContentBlock struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

type MessagesRequest struct {
	Model     string        `json:"model"`
	System    string        `json:"system"`
	Messages  []ChatMessage `json:"messages"`
	Tools     []ToolDef     `json:"tools"`
	MaxTokens int           `json:"max_tokens"`
}

type MessagesResponse struct {
	Content    []ContentBlock `json:"content"`
	StopReason string         `json:"stop_reason"`
}

type APIErrorResponse struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// =========================
// Anthropic client
// =========================

type AnthropicClient struct {
	BaseURL string
	APIKey  string
	Client  *http.Client
}

func NewAnthropicClient() (*AnthropicClient, error) {
	baseURL := strings.TrimSpace(os.Getenv("ANTHROPIC_BASE_URL"))
	if baseURL == "" {
		baseURL = "https://api.anthropic.com"
	}

	apiKey := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY"))
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv("ANTHROPIC_AUTH_TOKEN"))
	}
	if apiKey == "" {
		return nil, errors.New("missing ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN")
	}

	if !strings.HasPrefix(baseURL, "http://") && !strings.HasPrefix(baseURL, "https://") {
		return nil, fmt.Errorf("ANTHROPIC_BASE_URL must start with http:// or https://, got: %s", baseURL)
	}

	return &AnthropicClient{
		BaseURL: strings.TrimRight(baseURL, "/"),
		APIKey:  apiKey,
		Client: &http.Client{
			Timeout: 180 * time.Second,
		},
	}, nil
}

func (c *AnthropicClient) messagesURL() string {
	if strings.HasSuffix(c.BaseURL, "/v1/messages") {
		return c.BaseURL
	}
	if strings.HasSuffix(c.BaseURL, "/v1") {
		return c.BaseURL + "/messages"
	}
	return c.BaseURL + "/v1/messages"
}

func (c *AnthropicClient) CreateMessages(reqBody MessagesRequest) (*MessagesResponse, error) {
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, c.messagesURL(), bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.APIKey)
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := c.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		var apiErr APIErrorResponse
		if err := json.Unmarshal(body, &apiErr); err == nil && apiErr.Error.Message != "" {
			return nil, fmt.Errorf("api error (%d): %s", resp.StatusCode, apiErr.Error.Message)
		}
		return nil, fmt.Errorf("api error (%d): %s", resp.StatusCode, string(body))
	}

	var out MessagesResponse
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, fmt.Errorf("decode response failed: %w; raw=%s", err, string(body))
	}

	return &out, nil
}

// =========================
// Tool input structs
// =========================

type BashInput struct {
	Command string `json:"command"`
}

type ReadFileInput struct {
	Path  string `json:"path"`
	Limit *int   `json:"limit,omitempty"`
}

type WriteFileInput struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

type EditFileInput struct {
	Path    string `json:"path"`
	OldText string `json:"old_text"`
	NewText string `json:"new_text"`
}

type TodoInput struct {
	Items []TodoItem `json:"items"`
}

// =========================
// Tool implementations
// =========================

func truncateString(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max]
}

func safePath(workdir, p string) (string, error) {
	absWorkdir, err := filepath.Abs(workdir)
	if err != nil {
		return "", err
	}

	joined := filepath.Join(absWorkdir, p)
	absPath, err := filepath.Abs(joined)
	if err != nil {
		return "", err
	}

	rel, err := filepath.Rel(absWorkdir, absPath)
	if err != nil {
		return "", err
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes workspace: %s", p)
	}

	return absPath, nil
}

func runBash(workdir, command string) string {
	dangerous := []string{"rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"}
	for _, d := range dangerous {
		if strings.Contains(command, d) {
			return "Error: Dangerous command blocked"
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.CommandContext(ctx, "cmd", "/C", command)
	} else {
		cmd = exec.CommandContext(ctx, "bash", "-lc", command)
	}
	cmd.Dir = workdir

	out, err := cmd.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		return "Error: Timeout (120s)"
	}

	result := strings.TrimSpace(string(out))
	if result == "" {
		result = "(no output)"
	}

	if err != nil && result == "" {
		result = "Error: " + err.Error()
	}

	return truncateString(result, maxOutputLen)
}

func runRead(workdir, path string, limit *int) string {
	fp, err := safePath(workdir, path)
	if err != nil {
		return "Error: " + err.Error()
	}

	data, err := os.ReadFile(fp)
	if err != nil {
		return "Error: " + err.Error()
	}

	lines := strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n")
	if limit != nil && *limit < len(lines) {
		remaining := len(lines) - *limit
		lines = append(lines[:*limit], fmt.Sprintf("... (%d more)", remaining))
	}

	return truncateString(strings.Join(lines, "\n"), maxOutputLen)
}

func runWrite(workdir, path, content string) string {
	fp, err := safePath(workdir, path)
	if err != nil {
		return "Error: " + err.Error()
	}

	if err := os.MkdirAll(filepath.Dir(fp), 0o755); err != nil {
		return "Error: " + err.Error()
	}

	if err := os.WriteFile(fp, []byte(content), 0o644); err != nil {
		return "Error: " + err.Error()
	}

	return fmt.Sprintf("Wrote %d bytes", len(content))
}

func runEdit(workdir, path, oldText, newText string) string {
	fp, err := safePath(workdir, path)
	if err != nil {
		return "Error: " + err.Error()
	}

	data, err := os.ReadFile(fp)
	if err != nil {
		return "Error: " + err.Error()
	}

	content := string(data)
	if !strings.Contains(content, oldText) {
		return fmt.Sprintf("Error: Text not found in %s", path)
	}

	updated := strings.Replace(content, oldText, newText, 1)
	if err := os.WriteFile(fp, []byte(updated), 0o644); err != nil {
		return "Error: " + err.Error()
	}

	return fmt.Sprintf("Edited %s", path)
}

func handleTool(block ContentBlock, todo *TodoManager, workdir string) string {
	switch block.Name {
	case "bash":
		var in BashInput
		if err := json.Unmarshal(block.Input, &in); err != nil {
			return "Error: invalid bash input: " + err.Error()
		}
		return runBash(workdir, in.Command)

	case "read_file":
		var in ReadFileInput
		if err := json.Unmarshal(block.Input, &in); err != nil {
			return "Error: invalid read_file input: " + err.Error()
		}
		return runRead(workdir, in.Path, in.Limit)

	case "write_file":
		var in WriteFileInput
		if err := json.Unmarshal(block.Input, &in); err != nil {
			return "Error: invalid write_file input: " + err.Error()
		}
		return runWrite(workdir, in.Path, in.Content)

	case "edit_file":
		var in EditFileInput
		if err := json.Unmarshal(block.Input, &in); err != nil {
			return "Error: invalid edit_file input: " + err.Error()
		}
		return runEdit(workdir, in.Path, in.OldText, in.NewText)

	case "todo":
		var in TodoInput
		if err := json.Unmarshal(block.Input, &in); err != nil {
			return "Error: invalid todo input: " + err.Error()
		}
		out, err := todo.Update(in.Items)
		if err != nil {
			return "Error: " + err.Error()
		}
		return out

	default:
		return "Unknown tool: " + block.Name
	}
}

func buildTools() []ToolDef {
	return []ToolDef{
		{
			Name:        "bash",
			Description: "Run a shell command.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"command": map[string]any{"type": "string"},
				},
				"required": []string{"command"},
			},
		},
		{
			Name:        "read_file",
			Description: "Read file contents.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path":  map[string]any{"type": "string"},
					"limit": map[string]any{"type": "integer"},
				},
				"required": []string{"path"},
			},
		},
		{
			Name:        "write_file",
			Description: "Write content to file.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path":    map[string]any{"type": "string"},
					"content": map[string]any{"type": "string"},
				},
				"required": []string{"path", "content"},
			},
		},
		{
			Name:        "edit_file",
			Description: "Replace exact text in file.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path":     map[string]any{"type": "string"},
					"old_text": map[string]any{"type": "string"},
					"new_text": map[string]any{"type": "string"},
				},
				"required": []string{"path", "old_text", "new_text"},
			},
		},
		{
			Name:        "todo",
			Description: "Update task list. Track progress on multi-step tasks.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"items": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"id":     map[string]any{"type": "string"},
								"text":   map[string]any{"type": "string"},
								"status": map[string]any{"type": "string", "enum": []string{"pending", "in_progress", "completed"}},
							},
							"required": []string{"id", "text", "status"},
						},
					},
				},
				"required": []string{"items"},
			},
		},
	}
}

// =========================
// Agent loop
// =========================

func agentLoop(history *[]ChatMessage, client *AnthropicClient, model, system, workdir string, todo *TodoManager) error {
	roundsSinceTodo := 0
	tools := buildTools()

	for {
		resp, err := client.CreateMessages(MessagesRequest{
			Model:     model,
			System:    system,
			Messages:  *history,
			Tools:     tools,
			MaxTokens: 8000,
		})
		if err != nil {
			return err
		}

		*history = append(*history, ChatMessage{
			Role:    "assistant",
			Content: resp.Content,
		})

		if resp.StopReason != "tool_use" {
			return nil
		}

		results := make([]map[string]any, 0)
		usedTodo := false

		for _, block := range resp.Content {
			if block.Type != "tool_use" {
				continue
			}

			output := handleTool(block, todo, workdir)
			fmt.Printf("> %s: %s\n", block.Name, truncateString(output, 200))

			results = append(results, map[string]any{
				"type":        "tool_result",
				"tool_use_id": block.ID,
				"content":     output,
			})

			if block.Name == "todo" {
				usedTodo = true
			}
		}

		if usedTodo {
			roundsSinceTodo = 0
		} else {
			roundsSinceTodo++
		}

		if roundsSinceTodo >= 3 {
			results = append([]map[string]any{
				{
					"type": "text",
					"text": "<reminder>Update your todos.</reminder>",
				},
			}, results...)
		}

		*history = append(*history, ChatMessage{
			Role:    "user",
			Content: results,
		})
	}
}

func printAssistantText(content any) {
	blocks, ok := content.([]ContentBlock)
	if !ok {
		return
	}

	for _, block := range blocks {
		if block.Type == "text" && block.Text != "" {
			fmt.Println(block.Text)
		}
	}
}

// =========================
// main
// =========================

func main() {
	loadEnvFile(".env", true)

	workdir, err := os.Getwd()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	model := strings.TrimSpace(os.Getenv("MODEL_ID"))
	if model == "" {
		fmt.Println("Error: MODEL_ID is required")
		return
	}

	client, err := NewAnthropicClient()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	system := fmt.Sprintf(
		"You are a coding agent at %s.\nUse the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.\nPrefer tools over prose.",
		workdir,
	)

	todo := &TodoManager{}
	history := []ChatMessage{}

	fmt.Println("Go Todo Agent 启动成功！输入 q/exit 退出")

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for {
		fmt.Print("\033[36ms03 >> \033[0m")
		if !scanner.Scan() {
			break
		}

		query := scanner.Text()
		trimmed := strings.ToLower(strings.TrimSpace(query))
		if trimmed == "" || trimmed == "q" || trimmed == "exit" {
			break
		}

		history = append(history, ChatMessage{
			Role:    "user",
			Content: query,
		})

		if err := agentLoop(&history, client, model, system, workdir, todo); err != nil {
			fmt.Println("Error:", err)
			continue
		}

		if len(history) > 0 {
			printAssistantText(history[len(history)-1].Content)
		}
		fmt.Println()
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("Error:", err)
	}
}
