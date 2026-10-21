#!/bin/sh
# A stand-in for `pi --mode rpc` used by the tests: it answers a few commands in Pi's JSONL format.
field() { printf '%s' "$2" | sed -n "s/.*\"$1\":\"\([^\"]*\)\".*/\1/p"; }
while IFS= read -r line; do
  id=$(field id "$line")
  type=$(field type "$line")
  case "$type" in
    get_state)
      printf '{"type":"response","id":"%s","command":"get_state","success":true,"data":{"isStreaming":false,"thinkingLevel":"medium","model":{"provider":"opencode-go","id":"deepseek-v4.1-flash"}}}\n' "$id" ;;
    prompt)
      if [ "$(field message "$line")" = "FAIL" ]; then
        printf '{"type":"response","id":"%s","command":"prompt","success":false,"error":"Cannot submit a prompt while compaction is in progress"}\n' "$id"
        continue
      fi
      # A prompt with attachments says what it carried: how many images, and its message (with the attachment lines).
      case "$line" in *attached*)
        images=$(printf '%s' "$line" | grep -o '"type":"image"' | wc -l | tr -d ' ')
        printf '{"type":"prompt_seen","images":%s,"message":"%s"}\n' "$images" "$(field message "$line")" ;;
      esac
      behavior=$(field streamingBehavior "$line")
      [ -n "$behavior" ] && printf '{"type":"queued_as","behavior":"%s","message":"%s"}\n' "$behavior" "$(field message "$line")"
      printf '{"type":"response","id":"%s","command":"prompt","success":true,"data":{"disposition":"started"}}\n' "$id"
      printf '{"type":"agent_start"}\n'
      printf '{"type":"message_update","assistantMessageEvent":{"type":"text_delta","contentIndex":0,"delta":"hi"}}\n'
      printf '{"type":"agent_settled","aborted":false}\n' ;;
    reject)
      printf '{"type":"response","id":"%s","command":"reject","success":false,"error":"nope"}\n' "$id" ;;
    ask)
      printf '{"type":"extension_ui_request","id":"u1","method":"select","title":"Allow?","options":["Allow","Block"]}\n' ;;
    ask_timeout)
      printf '{"type":"extension_ui_request","id":"u2","method":"confirm","title":"Switch?","timeout":150}\n' ;;
    notify)
      printf '{"type":"extension_ui_request","id":"n1","method":"notify","message":"careful","notifyType":"warning"}\n' ;;
    get_commands)
      printf '{"type":"response","id":"%s","command":"get_commands","success":true,"data":{"commands":[{"name":"effort","description":"Set effort","source":"extension"}]}}\n' "$id" ;;
    get_available_models)
      printf '{"type":"response","id":"%s","command":"get_available_models","success":true,"data":{"models":[{"provider":"opencode-go","id":"glm-5.3","name":"GLM 5.3","input":["text"]},{"provider":"opencode-go","id":"kimi-k3","name":"Kimi K3","input":["text","image"]}]}}\n' "$id" ;;
    get_messages)
      printf '{"type":"response","id":"%s","command":"get_messages","success":true,"data":{"messages":[{"role":"user","content":"earlier"}]}}\n' "$id" ;;
    get_session_stats)
      printf '{"type":"response","id":"%s","command":"get_session_stats","success":true,"data":{"contextUsage":{"tokens":60000,"contextWindow":200000,"percent":30}}}\n' "$id" ;;
    compact)
      printf '{"type":"response","id":"%s","command":"compact","success":true,"data":{"tokensBefore":150000,"estimatedTokensAfter":32000,"instructions":"%s"}}\n' "$id" "$(field customInstructions "$line")" ;;
    export_html)
      printf '{"type":"response","id":"%s","command":"export_html","success":false,"error":"Nothing to export"}\n' "$id" ;;
    clear_queue)
      printf '{"type":"response","id":"%s","command":"clear_queue","success":true,"data":{"steering":[],"followUp":["x"]}}\n' "$id"
      printf '{"type":"queue_update","steering":[],"followUp":[]}\n' ;;
    set_model)
      if [ "$(field modelId "$line")" = "rejected" ]; then
        printf '{"type":"response","id":"%s","command":"set_model","success":false,"error":"Model not found"}\n' "$id"
        continue
      fi
      printf '{"type":"response","id":"%s","command":"set_model","success":true,"data":{}}\n' "$id" ;;
    extension_ui_response)
      printf '{"type":"answered","value":"%s"}\n' "$(field value "$line")" ;;
    start)
      printf '{"type":"agent_start"}\n' ;;
    tool_start)
      printf '{"type":"tool_execution_start","toolCallId":"t1","toolName":"bash","args":{"command":"cargo build"}}\n' ;;
    tool_end)
      printf '{"type":"tool_execution_end","toolCallId":"t1","toolName":"bash","result":{},"isError":false}\n' ;;
    abort)
      printf '{"type":"abort_seen"}\n' ;;
    settle)
      printf '{"type":"agent_settled","aborted":true}\n' ;;
    ask_card)
      printf '{"type":"extension_ui_request","id":"c1","method":"select","title":"{\\"aria\\":1,\\"kind\\":\\"sandbox\\",\\"summary\\":\\"Wants to run rm -rf build\\"}","options":["Allow once","Block"]}\n' ;;
    args)
      printf '{"type":"args","args":"%s"}\n' "$ARIA_TEST_ARGS" ;;
    background)
      sleep 300 >/dev/null 2>&1 &
      printf '{"type":"background","pid":"%s"}\n' "$!" ;;
    env_projects)
      printf '{"type":"env","projects":"%s"}\n' "$ARIA_PROJECTS_DIR" ;;
    env_settings)
      printf '{"type":"env","settings":"%s"}\n' "$ARIA_SETTINGS" ;;
    env_watch)
      printf '{"type":"env","watch":"%s"}\n' "$PI_WATCH_SOCK" ;;
    env_jobs)
      printf '{"type":"env","jobs":"%s"}\n' "$ARIA_JOBS_SOCK" ;;
    parallel_call)
      printf '{"type":"tool_execution_start","toolCallId":"p1","toolName":"parallel","args":{"tasks":[{"agent":"security","task":"Review the policy"}]}}\n' ;;
    crash) exit 3 ;;
    silent) : ;;
  esac
done
