package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/joho/godotenv"
)

var (
	workdir      string
	modelID      string
	systemPrompt string
)

type toolHandler func(json.RawMessage) string

var toolHandlers = map[string]toolHandler{
	"bash":       handleBash,
	"read_file":  handleReadFile,
	"write_file": handleWriteFile,
	"edit_file":  handleEditFile,
}

func main() {
	_ = godotenv.Overload()

	wd, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	workdir, err = filepath.Abs(wd)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// 对齐原 Python 逻辑：
	// if os.getenv("ANTHROPIC_BASE_URL"):
	//     os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
	if strings.TrimSpace(os.Getenv("ANTHROPIC_BASE_URL")) != "" {
		_ = os.Unsetenv("ANTHROPIC_AUTH_TOKEN")
	}

	modelID = mustGetEnv("MODEL_ID")
	systemPrompt = fmt.Sprintf(
		"You are a coding agent at %s. Use tools to solve tasks. Act, don't explain.",
		workdir,
	)

	client := buildClient()
	tools := buildTools()

	history := make([]anthropic.MessageParam, 0, 16)

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for {
		fmt.Printf("Go AI Agent 启动成功！输入 q/exit 退出")
		fmt.Print("\033[36mgo-agent >> \033[0m")
		if !scanner.Scan() {
			break
		}

		query := scanner.Text()
		trimmed := strings.ToLower(strings.TrimSpace(query))
		if trimmed == "" || trimmed == "q" || trimmed == "exit" {
			break
		}

		history = append(history, anthropic.NewUserMessage(
			anthropic.NewTextBlock(query),
		))

		resp, err := agentLoop(context.Background(), client, &history, tools)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n\n", err)
			continue
		}

		for _, block := range resp.Content {
			switch b := block.AsAny().(type) {
			case anthropic.TextBlock:
				fmt.Println(b.Text)
			}
		}
		fmt.Println()
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func buildClient() anthropic.Client {
	baseURL := strings.TrimSpace(os.Getenv("ANTHROPIC_BASE_URL"))
	apiKey := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY"))

	opts := make([]option.RequestOption, 0, 2)

	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}

	// 官方 Go SDK 默认使用 ANTHROPIC_API_KEY。
	// 这里显式加上，避免完全依赖环境推断。
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}

	return anthropic.NewClient(opts...)
}

func buildTools() []anthropic.ToolUnionParam {
	toolParams := []anthropic.ToolParam{
		{
			Name:        "bash",
			Description: anthropic.String("Run a shell command."),
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: map[string]any{
					"command": map[string]any{
						"type": "string",
					},
				},
				Required: []string{"command"},
			},
		},
		{
			Name:        "read_file",
			Description: anthropic.String("Read file contents."),
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: map[string]any{
					"path": map[string]any{
						"type": "string",
					},
					"limit": map[string]any{
						"type": "integer",
					},
				},
				Required: []string{"path"},
			},
		},
		{
			Name:        "write_file",
			Description: anthropic.String("Write content to file."),
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: map[string]any{
					"path": map[string]any{
						"type": "string",
					},
					"content": map[string]any{
						"type": "string",
					},
				},
				Required: []string{"path", "content"},
			},
		},
		{
			Name:        "edit_file",
			Description: anthropic.String("Replace exact text in file."),
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: map[string]any{
					"path": map[string]any{
						"type": "string",
					},
					"old_text": map[string]any{
						"type": "string",
					},
					"new_text": map[string]any{
						"type": "string",
					},
				},
				Required: []string{"path", "old_text", "new_text"},
			},
		},
	}

	tools := make([]anthropic.ToolUnionParam, 0, len(toolParams))
	for i := range toolParams {
		tp := toolParams[i]
		tools = append(tools, anthropic.ToolUnionParam{OfTool: &tp})
	}

	return tools
}

func agentLoop(
	ctx context.Context,
	client anthropic.Client,
	messages *[]anthropic.MessageParam,
	tools []anthropic.ToolUnionParam,
) (*anthropic.Message, error) {
	for {
		resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     anthropic.Model(modelID),
			System:    []anthropic.TextBlockParam{{Text: systemPrompt}},
			Messages:  *messages,
			Tools:     tools,
			MaxTokens: 8000,
		})
		if err != nil {
			return nil, err
		}

		*messages = append(*messages, resp.ToParam())

		if string(resp.StopReason) != "tool_use" {
			return resp, nil
		}

		results := make([]anthropic.ContentBlockParamUnion, 0)

		for _, block := range resp.Content {
			switch b := block.AsAny().(type) {
			case anthropic.ToolUseBlock:
				handler := toolHandlers[b.Name]

				var output string
				if handler == nil {
					output = "Unknown tool: " + b.Name
				} else {
					output = handler(json.RawMessage(b.JSON.Input.Raw()))
				}

				fmt.Printf("> %s: %s\n", b.Name, truncateString(output, 200))
				results = append(results, anthropic.NewToolResultBlock(b.ID, output, false))
			}
		}

		if len(results) == 0 {
			return resp, nil
		}

		*messages = append(*messages, anthropic.NewUserMessage(results...))
	}
}

func safePath(p string) (string, error) {
	full := filepath.Clean(filepath.Join(workdir, p))

	rel, err := filepath.Rel(workdir, full)
	if err != nil {
		return "", err
	}

	if rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "", fmt.Errorf("Path escapes workspace: %s", p)
	}

	return full, nil
}

func runBash(command string) string {
	dangerous := []string{
		"rm -rf /",
		"sudo",
		"shutdown",
		"reboot",
		"> /dev/",
	}

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
		cmd = exec.CommandContext(ctx, "sh", "-c", command)
	}
	cmd.Dir = workdir

	out, err := cmd.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		return "Error: Timeout (120s)"
	}

	result := strings.TrimSpace(string(out))
	if result == "" {
		return "(no output)"
	}

	_ = err
	return truncateString(result, 50000)
}

func runRead(path string, limit *int) string {
	fp, err := safePath(path)
	if err != nil {
		return "Error: " + err.Error()
	}

	data, err := os.ReadFile(fp)
	if err != nil {
		return "Error: " + err.Error()
	}

	text := string(data)
	lines := splitLines(text)

	if limit != nil && *limit > 0 && *limit < len(lines) {
		more := len(lines) - *limit
		lines = append(lines[:*limit], fmt.Sprintf("... (%d more lines)", more))
	}

	return truncateString(strings.Join(lines, "\n"), 50000)
}

func runWrite(path, content string) string {
	fp, err := safePath(path)
	if err != nil {
		return "Error: " + err.Error()
	}

	if err := os.MkdirAll(filepath.Dir(fp), 0o755); err != nil {
		return "Error: " + err.Error()
	}

	if err := os.WriteFile(fp, []byte(content), 0o644); err != nil {
		return "Error: " + err.Error()
	}

	return fmt.Sprintf("Wrote %d bytes to %s", len(content), path)
}

func runEdit(path, oldText, newText string) string {
	fp, err := safePath(path)
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

type bashInput struct {
	Command string `json:"command"`
}

func handleBash(raw json.RawMessage) string {
	var in bashInput
	if err := json.Unmarshal(raw, &in); err != nil {
		return "Error: " + err.Error()
	}
	return runBash(in.Command)
}

type readFileInput struct {
	Path  string `json:"path"`
	Limit *int   `json:"limit,omitempty"`
}

func handleReadFile(raw json.RawMessage) string {
	var in readFileInput
	if err := json.Unmarshal(raw, &in); err != nil {
		return "Error: " + err.Error()
	}
	return runRead(in.Path, in.Limit)
}

type writeFileInput struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

func handleWriteFile(raw json.RawMessage) string {
	var in writeFileInput
	if err := json.Unmarshal(raw, &in); err != nil {
		return "Error: " + err.Error()
	}
	return runWrite(in.Path, in.Content)
}

type editFileInput struct {
	Path    string `json:"path"`
	OldText string `json:"old_text"`
	NewText string `json:"new_text"`
}

func handleEditFile(raw json.RawMessage) string {
	var in editFileInput
	if err := json.Unmarshal(raw, &in); err != nil {
		return "Error: " + err.Error()
	}
	return runEdit(in.Path, in.OldText, in.NewText)
}

func splitLines(s string) []string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")

	if s == "" {
		return []string{}
	}

	lines := strings.Split(s, "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

func truncateString(s string, max int) string {
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max])
}

func mustGetEnv(key string) string {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		fmt.Fprintf(os.Stderr, "Error: missing required env %s\n", key)
		os.Exit(1)
	}
	return v
}
