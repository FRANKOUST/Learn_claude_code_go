/*
s01_agent_loop.py - The Agent Loop
The entire secret of an AI coding agent in one pattern:
while stop_reason == "tool_use":
response = LLM(messages, tools)
execute tools
append results
+----------+      +-------+      +---------+
|   User   | ---> |  LLM  | ---> |  Tool   |
|  prompt  |      |       |      | execute |
+----------+      +---+---+      +----+----+
^               |
|   tool_result |
+---------------+
(loop continues)
This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
*/

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/joho/godotenv"
)

var (
	model        string
	systemPrompt string

	// 简单危险命令拦截
	dangerousCmds = []string{
		"rm -rf /",
		"sudo ",
		"shutdown",
		"reboot",
		"> /dev/",
		":(){:|:&};:", // fork bomb
	}
)

// 初始化：加载环境变量并校验配置
func init() {
	_ = godotenv.Load(".env")

	workDir, err := os.Getwd()
	if err != nil {
		fmt.Println("获取工作目录失败:", err)
		os.Exit(1)
	}

	systemPrompt = fmt.Sprintf(
		"You are a coding agent working in %s. You may use the bash tool to solve tasks. Act directly and be concise.",
		workDir,
	)

	requiredEnv := map[string]string{
		"MODEL_ID":           "未配置 MODEL_ID",
		"ANTHROPIC_API_KEY":  "未配置 ANTHROPIC_API_KEY",
		"ANTHROPIC_BASE_URL": "未配置ANTHROPIC_BASE_URL",
	}

	for key, msg := range requiredEnv {
		if strings.TrimSpace(os.Getenv(key)) == "" {
			fmt.Println(msg)
			os.Exit(1)
		}
	}

	model = strings.TrimSpace(os.Getenv("MODEL_ID"))
}

func isDangerousCommand(command string) bool {
	normalized := strings.ToLower(strings.TrimSpace(command))
	for _, bad := range dangerousCmds {
		if strings.Contains(normalized, strings.ToLower(bad)) {
			return true
		}
	}
	return false
}

// RunBash 执行 bash 命令，带危险拦截和超时控制
func RunBash(command string) string {
	command = strings.TrimSpace(command)
	if command == "" {
		return "ERROR: 空命令，未执行"
	}

	if isDangerousCommand(command) {
		return "ERROR: 禁止执行危险命令"
	}

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "bash", "-c", command)

	// 让命令在当前工作目录执行
	if wd, err := os.Getwd(); err == nil {
		cmd.Dir = wd
	}

	output, err := cmd.CombinedOutput()

	if ctx.Err() == context.DeadlineExceeded {
		return "ERROR: 命令执行超时（120s）"
	}

	result := strings.TrimSpace(string(output))
	if result == "" {
		result = "(无输出)"
	}

	if len(result) > 50000 {
		result = result[:50000] + "\n...(输出已截断)"
	}

	if err != nil {
		return fmt.Sprintf("ERROR: 命令执行失败: %v\n%s", err, result)
	}

	return result
}

// getTools 定义 bash 工具
func getTools() []anthropic.ToolUnionParam {
	bashTool := anthropic.ToolParam{
		Name:        "bash",
		Description: anthropic.String("Run a shell command and return stdout/stderr."),
		InputSchema: anthropic.ToolInputSchemaParam{
			Properties: map[string]any{
				"command": map[string]any{
					"type":        "string",
					"description": "The shell command to execute.",
				},
			},
			Required: []string{"command"},
		},
	}

	return []anthropic.ToolUnionParam{
		{OfTool: &bashTool},
	}
}

// AgentLoop 核心循环：直到模型不再请求工具
func AgentLoop(messages *[]anthropic.MessageParam) (string, error) {
	client := anthropic.NewClient()
	tools := getTools()

	for {
		resp, err := client.Messages.New(context.Background(), anthropic.MessageNewParams{
			Model:     anthropic.Model(model),
			MaxTokens: 4096,
			System: []anthropic.TextBlockParam{
				{Text: systemPrompt},
			},
			Messages: *messages,
			Tools:    tools,
		})
		if err != nil {
			return "", fmt.Errorf("调用大模型失败: %w", err)
		}

		// 官方 SDK 提供了 ToParam，直接把模型回复回写到历史里
		*messages = append(*messages, resp.ToParam())

		var finalText strings.Builder
		var toolResults []anthropic.ContentBlockParamUnion

		for _, block := range resp.Content {
			switch variant := block.AsAny().(type) {
			case anthropic.TextBlock:
				finalText.WriteString(variant.Text)

			case anthropic.ToolUseBlock:
				var input struct {
					Command string `json:"command"`
				}

				if err := json.Unmarshal([]byte(variant.JSON.Input.Raw()), &input); err != nil {
					errMsg := fmt.Sprintf("ERROR: 解析工具参数失败: %v", err)
					toolResults = append(toolResults, anthropic.NewToolResultBlock(variant.ID, errMsg, true))
					continue
				}

				command := strings.TrimSpace(input.Command)
				if command == "" {
					errMsg := "ERROR:缺少 command 参数"
					toolResults = append(toolResults, anthropic.NewToolResultBlock(variant.ID, errMsg, true))
					continue
				}

				fmt.Printf("\033[33m$ %s\033[0m\n", command)

				output := RunBash(command)
				if len(output) > 300 {
					fmt.Println(output[:300] + "...")
				} else {
					fmt.Println(output)
				}

				isErr := strings.HasPrefix(output, "ERROR:")
				toolResults = append(toolResults, anthropic.NewToolResultBlock(variant.ID, output, isErr))
			}
		}

		// 没有工具调用，说明这是最终文本回复
		if len(toolResults) == 0 {
			return strings.TrimSpace(finalText.String()), nil
		}

		// 把工具结果作为 user 消息继续喂回去
		*messages = append(*messages, anthropic.NewUserMessage(toolResults...))
	}
}

func main() {
	fmt.Println("Go AI Agent 启动成功！输入 q/exit 退出")

	history := []anthropic.MessageParam{}
	scanner := bufio.NewScanner(os.Stdin)

	// 放大 scanner 缓冲，避免长输入报错
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for {
		fmt.Print("\033[36mgo-agent >> \033[0m")
		if !scanner.Scan() {
			break
		}

		query := strings.TrimSpace(scanner.Text())

		if query == "" {
			continue
		}
		if query == "q" || query == "exit" {
			break
		}

		history = append(history, anthropic.NewUserMessage(anthropic.NewTextBlock(query)))

		answer, err := AgentLoop(&history)
		if err != nil {
			fmt.Println(err)
			fmt.Println("------------------------")
			continue
		}

		fmt.Println("\n------------------------")
		if answer == "" {
			fmt.Println("(模型未返回文本)")
		} else {
			fmt.Println(answer)
		}
		fmt.Println("------------------------\n")
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("读取输入失败:", err)
	}

	fmt.Println(" 已退出")
}
