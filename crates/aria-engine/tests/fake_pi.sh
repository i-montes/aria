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
      printf '{"type":"response","id":"%s","command":"get_available_models","success":true,"data":{"models":[{"provider":"opencode-go","id":"glm-5.3","name":"GLM 5.3"}]}}\n' "$id" ;;
    get_messages)
      printf '{"type":"response","id":"%s","command":"get_messages","success":true,"data":{"messages":[{"role":"user","content":"earlier"}]}}\n' "$id" ;;
    set_model)
      printf '{"type":"response","id":"%s","command":"set_model","success":true,"data":{}}\n' "$id" ;;
    extension_ui_response)
      printf '{"type":"answered","value":"%s"}\n' "$(field value "$line")" ;;
    crash) exit 3 ;;
    silent) : ;;
  esac
done
