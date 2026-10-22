#!/bin/sh
# A stand-in for `pi --mode rpc` used by the tests: it answers a few commands in Pi's JSONL format.
field() { printf '%s' "$2" | sed -n "s/.*\"$1\":\"\([^\"]*\)\".*/\1/p"; }
# The session it runs on, as Pi takes it from `--session-dir`/`--session-id`; `/aria-rotate` moves it to `rotated-<n>`
# and, as Pi's new session does, back to the default model and thinking level.
sid=""; sdir=""; rotations=0; rotate_late=""
provider="opencode-go"; model="deepseek-v4.1-flash"; thinking="medium"
while [ $# -gt 0 ]; do
  case "$1" in
    --session-id) sid="$2"; shift ;;
    --session-dir) sdir="$2"; shift ;;
  esac
  shift
done
while IFS= read -r line; do
  id=$(field id "$line")
  type=$(field type "$line")
  case "$type" in
    get_state)
      printf '{"type":"response","id":"%s","command":"get_state","success":true,"data":{"isStreaming":false,"thinkingLevel":"%s","model":{"provider":"%s","id":"%s"},"sessionId":"%s","sessionFile":"%s"}}\n' "$id" "$thinking" "$provider" "$model" "$sid" "${sdir:+$sdir/$sid.jsonl}" ;;
    prompt)
      # The rotate extension's command: no model turn; the session moves to a new id.
      # It answers after a moment, as Pi does once the new session is up; a prompt that comes meanwhile says so.
      flag="${sdir:-.}/.fake-rotating-$$"
      if [ "$(field message "$line")" = "/aria-rotate" ]; then
        rotations=$((rotations + 1)); sid="rotated-$rotations"
        provider="opencode-go"; model="deepseek-v4.1-flash"; thinking="medium"
        mkdir -p "$(dirname "$flag")" && : > "$flag"
        printf '{"type":"rotate_seen","session":"%s"}\n' "$sid"
        # After `rotate_late`, the answer comes after the caller stopped waiting.
        ( sleep "${rotate_late:-0.3}"; rm -f "$flag"
          printf '{"type":"response","id":"%s","command":"prompt","success":true,"data":{"disposition":"handled"}}\n' "$id" ) &
        continue
      fi
      [ -e "$flag" ] && printf '{"type":"prompt_during_rotation"}\n'
      # Pi refuses every prompt while it compacts.
      if [ -e "${sdir:-.}/.fake-compacting-$$" ]; then
        printf '{"type":"prompt_during_compaction"}\n'
        printf '{"type":"response","id":"%s","command":"prompt","success":false,"error":"Cannot submit a prompt while compaction is in progress. Wait for compaction to finish and retry."}\n' "$id"
        continue
      fi
      # A run that dies half-way (Pi crashed).
      case "$(field message "$line")" in *CRASH*)
        printf '{"type":"response","id":"%s","command":"prompt","success":true,"data":{"disposition":"started"}}\n' "$id"
        printf '{"type":"agent_start"}\n'
        exit 3 ;;
      esac
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
      # A scheduled task's prompt (it says TASKS) gets a whole answer, as a run's summary is its first line.
      case "$(field message "$line")" in *TASKS*)
        printf '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"Here are 3 tasks for today\\nthe first"}],"stopReason":"stop"}}\n' ;;
      esac
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
      printf '{"type":"response","id":"%s","command":"get_commands","success":true,"data":{"commands":[{"name":"effort","description":"Set effort","source":"extension"},{"name":"aria-rotate","description":"Rotate","source":"extension"}]}}\n' "$id" ;;
    get_available_models)
      printf '{"type":"response","id":"%s","command":"get_available_models","success":true,"data":{"models":[{"provider":"opencode-go","id":"glm-5.3","name":"GLM 5.3","input":["text"]},{"provider":"opencode-go","id":"kimi-k3","name":"Kimi K3","input":["text","image"]}]}}\n' "$id" ;;
    get_messages)
      printf '{"type":"response","id":"%s","command":"get_messages","success":true,"data":{"messages":[{"role":"user","content":"earlier"}]}}\n' "$id" ;;
    get_session_stats)
      printf '{"type":"response","id":"%s","command":"get_session_stats","success":true,"data":{"contextUsage":{"tokens":60000,"contextWindow":200000,"percent":30}}}\n' "$id" ;;
    compact)
      # Pi's manual compaction says so as events before it answers; it takes a moment, and refuses prompts meanwhile.
      printf '{"type":"compact_seen"}\n'
      printf '{"type":"compaction_start","reason":"manual"}\n'
      compacting="${sdir:-.}/.fake-compacting-$$"
      mkdir -p "$(dirname "$compacting")" && : > "$compacting"
      instructions=$(field customInstructions "$line")
      ( sleep 0.3; rm -f "$compacting"
        printf '{"type":"compaction_end","reason":"manual","result":{"summary":"S"},"aborted":false,"willRetry":false}\n'
        printf '{"type":"response","id":"%s","command":"compact","success":true,"data":{"tokensBefore":150000,"estimatedTokensAfter":32000,"instructions":"%s"}}\n' "$id" "$instructions" ) & ;;
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
      provider=$(field provider "$line"); model=$(field modelId "$line")
      printf '{"type":"model_set","model":"%s/%s"}\n' "$provider" "$model"
      printf '{"type":"response","id":"%s","command":"set_model","success":true,"data":{}}\n' "$id" ;;
    set_thinking_level)
      thinking=$(field level "$line")
      printf '{"type":"thinking_set","level":"%s"}\n' "$thinking"
      printf '{"type":"response","id":"%s","command":"set_thinking_level","success":true}\n' "$id" ;;
    rotate_late)
      rotate_late=1 ;;
    move_session)
      # Pi switched sessions and Aria was not told (a rotation whose answer was lost).
      sid="moved-1"
      printf '{"type":"moved","session":"%s"}\n' "$sid" ;;
    extension_ui_response)
      printf '{"type":"answered","value":"%s"}\n' "$(field value "$line")" ;;
    start)
      printf '{"type":"agent_start"}\n' ;;
    tool_start)
      printf '{"type":"tool_execution_start","toolCallId":"t1","toolName":"bash","args":{"command":"cargo build"}}\n' ;;
    tool_end)
      printf '{"type":"tool_execution_end","toolCallId":"t1","toolName":"bash","result":{},"isError":false}\n' ;;
    tool_start_b)
      printf '{"type":"tool_execution_start","toolCallId":"t2","toolName":"bash","args":{"command":"bun test"}}\n' ;;
    tool_end_b)
      printf '{"type":"tool_execution_end","toolCallId":"t2","toolName":"bash","result":{},"isError":false}\n' ;;
    compaction_start)
      printf '{"type":"compaction_start","reason":"threshold"}\n' ;;
    compaction_end)
      printf '{"type":"compaction_end","reason":"threshold","result":{"summary":"S"},"aborted":false,"willRetry":false}\n' ;;
    compaction_failed)
      printf '{"type":"compaction_end","reason":"threshold","aborted":false,"willRetry":false,"errorMessage":"no model"}\n' ;;
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
    env_grants)
      printf '{"type":"env","grants":"%s"}\n' "$(printf '%s' "$ARIA_GRANTS" | base64 | tr -d '\n')" ;;
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
