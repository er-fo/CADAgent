# CADAgent Backend Flow Schema

This document maps the main CADAgent backend inputs, outputs, realization paths,
runtime stores, external services, and deployment flow.

Scope: `cadagent-backend-legacy` plus the Fusion add-in message surfaces that
feed or consume backend flows.

## Top-Level Runtime Graph

```text
CADAgent palette / Fusion 360 add-in
  |
  | user text, image, attachments, model, reasoning effort,
  | timeline_state, visual_context, selection_context, spatial_context,
  | entity_context, Supabase JWT, per-session BYOK provider keys
  v
FastAPI backend
  |
  +-- HTTP
  |     GET /       -> service metadata
  |     GET /health -> process + WebSocket capacity status
  |
  +-- WebSocket /ws/{session_id}
        |
        +-- authenticate / update_api_keys
        |     -> validate/store Supabase JWT
        |     -> store user_id for rate limiting
        |     -> store per-session LLM provider keys
        |
        +-- execute_request
        |     -> normalize attachments
        |     -> optional image/sketch vision translation
        |     -> create message checkpoint
        |     -> prepopulate entity refs
        |     -> ensure feature snapshot
        |     -> prompt routing + prompt/tool assembly
        |     -> LLM/tool loop
        |     -> Fusion or build123d realization
        |
        +-- planning_request
        |     -> clear active workflow state
        |     -> stream planning LLM output
        |     -> wait for plan_approval
        |     -> execute approved plan through execute_request
        |
        +-- result/callback messages
        |     <- execution_result / execution_error
        |     <- feature_snapshot
        |     <- entity_context_response
        |     <- plan_approval
        |     <- question_tree_completed
        |     <- design_selected
        |
        +-- control messages
              -> cancel_request
              -> revert_request
              -> resume_operation_request
              -> iteration_feedback
```

## Primary Inputs

```text
User-facing request payload
  |
  +-- Required or primary
  |     user_request
  |     image_data + image_format
  |     attachments
  |
  +-- Context from Fusion/add-in
  |     timeline_state
  |     entity_context
  |       -> bodies
  |       -> faces
  |       -> edges
  |       -> spatial_context
  |     selection_context
  |     visual_context
  |     client_capabilities
  |
  +-- Execution controls
  |     type = execute_request | planning_request
  |     execution_target = fusion | build123d | studio
  |     model_name
  |     reasoning_effort
  |     max_iterations
  |     request_id
  |
  +-- Auth and provider access
        Supabase access token
        llm_api_keys / api_keys
```

## Session State Graph

```text
ConnectionManager session_id
  |
  +-- transport
  |     active WebSocket connection
  |     pending Fusion result queue
  |     pending entity context queue
  |
  +-- auth/provider state
  |     authenticated flag
  |     user_token
  |     user_id
  |     llm_api_keys
  |
  +-- conversation/workflow state
  |     conversation_history
  |     message_checkpoints
  |     operation_checkpoints
  |     active_build_plan
  |     reasoning_context
  |
  +-- CAD context
  |     latest_entity_context
  |     EntityStore
  |       short refs: face_0, e0, body_0, v0
  |       opaque Fusion entity tokens
  |       geometry fingerprints
  |     SketchEntityStore
  |     feature_snapshot
  |
  +-- target-independent model state
        IRDocumentState
          operations[]
          entities{}
          metadata.source = fusion | studio
```

## Execute Request Flow

```text
execute_request
  |
  +-- request validation
  |     require user_request or image_data or attachments
  |     require authenticated session unless AUTH_BYPASS
  |
  +-- normalize input
  |     attachments.py
  |       -> normalized_attachments
  |       -> attachments_context
  |       -> first image promoted to image_data when needed
  |
  +-- optional vision translation
  |     vision_translator.py
  |       image_data + user text + spatial/entity context
  |       -> expanded user_request
  |
  +-- session logging
  |     runs/<timestamp>_session_<session_id>/
  |
  +-- checkpoint and context
  |     capture message checkpoint
  |     send checkpoint_created
  |     prepopulate EntityStore from entity_context
  |     request or cache feature_snapshot
  |     append user message with current CAD context
  |
  +-- workflow loop
        |
        +-- route prompt
        |     prompt_router.py
        |       user_request + history + active_build_plan
        |       -> selected tool clusters
        |
        +-- build prompt
        |     prompt_builder.py + prompt_structure.py
        |       selected clusters
        |       -> system prompt
        |       -> tool schemas
        |
        +-- sync runtime state
        |     request_entity_context when stale or required
        |     update latest_entity_context and EntityStore
        |
        +-- call LLM
        |     llm_client.py
        |       messages + tools + system prompt
        |       -> text response or tool_use blocks
        |       -> optional reasoning_chunk stream
        |
        +-- execute each tool call
        |     validate/ref-resolve/preflight
        |     map to IR where supported
        |     realize on target
        |     append tool_result
        |     refresh/enrich CAD context
        |     capture operation checkpoint
        |
        +-- terminal response
              llm_message
              completed
              conversation_history persisted in memory
```

## LLM Provider and Quota Flow

```text
call_claude_with_tools / generate_plan
  |
  +-- model normalization
  |     Claude / Anthropic
  |     OpenAI Responses or Chat
  |     Gemini
  |     managed Bedrock OpenAI-compatible models
  |
  +-- provider key resolution
  |     per-session BYOK keys first
  |     environment fallback
  |
  +-- usage-gateway path for managed free-tier models
        |
        v
SupabaseAPIGateway
  |
  +-- POST /functions/v1/api-generate
        headers:
          Authorization: Bearer <user JWT or service role>
          x-user-token: <user JWT>
          apikey: <publishable/anon/service key>
        body:
          provider
          model
          request_id
          input.messages
          input.system
          input.tools
          input.max_tokens
          input.reasoning_effort
        |
        v
Supabase Edge Function api-generate
  |
  +-- validate user token
  +-- ensure/read user_entitlements
  +-- count recent usage_ledger rows
  +-- estimate request token budget
  +-- reject over hourly or period quota
  +-- call Bedrock OpenAI-compatible /chat/completions
  +-- convert OpenAI Chat response to Anthropic-shaped result
  +-- insert usage_ledger row
  +-- return result + usage
```

## Tool Cluster Graph

```text
Prompt/tool cluster selection
  |
  +-- core
  |     final text response / one clarifying question
  |
  +-- sketch_tools
  |     create_sketch
  |     add_circle
  |     add_line
  |     add_arc
  |     add_rectangle
  |     add_sketch_geometry_batch
  |     list_sketch_profiles
  |
  +-- construction
  |     create_construction_plane
  |
  +-- 3d_modeling
  |     extrude_profile
  |     revolve_profile
  |     create_loft
  |
  +-- inspection
  |     list_features
  |
  +-- selection
  |     select_edges / select_faces / select_bodies
  |     clear_edge_selection / clear_face_selection / clear_body_selection
  |
  +-- modification
  |     apply_fillet
  |     apply_chamfer
  |     create_shell
  |
  +-- holes
  |     create_simple_hole
  |     create_counterbore_hole
  |     create_tapped_hole
  |
  +-- threading
  |     create_tapped_hole
  |     create_external_thread
  |
  +-- patterns
  |     create_pattern_feature
  |
  +-- timeline
  |     delete_feature
  |     adjust_feature_parameters
  |     suppress_feature
  |     unsuppress_feature
  |
  +-- design_exploration
        generate_question_tree
        propose_designs
        output_build_plan
```

## Realization Flow: Fusion Target

```text
LLM tool_use
  |
  +-- map/validate/preflight
  |     entity refs -> Fusion tokens
  |     stale refs -> candidate recovery
  |     sketch/profile/hole/timeline-specific guards
  |
  +-- IR-supported operation
  |     ir/mapper.py
  |       -> IROperation
  |     backends/fusion/translator.py
  |       -> Fusion tool call shape
  |     backends/fusion/executor.py
  |       -> execute_code or feature/selection payload
  |
  +-- legacy/direct operation
  |     code_generator.py
  |       tool name + parameters
  |       -> Fusion Python code
  |
  v
Backend -> add-in
  execute_code | feature_operation | edge_operation | face_operation | body_operation
  |
  v
Fusion add-in main thread
  |
  +-- execute_code
  |     CodeExecutor.execute_code()
  |     exec(code) with adsk, design, rootComp, registries, plane_manager
  |     -> actual Fusion geometry/timeline mutation
  |
  +-- feature_operation
  |     feature_tools.py
  |     -> timeline feature delete/edit/suppress/unsuppress/pattern/hole/etc.
  |
  +-- edge/face/body operation
  |     edge_tools.py / face_tools.py / body_tools.py
  |     -> selection and entity operations
  |
  v
Add-in -> backend
  execution_result | execution_error
  |
  v
Backend post-processing
  |
  +-- summarize result for LLM
  +-- register sketch entities
  +-- request fresh entity_context when needed
  +-- request feature_snapshot when needed
  +-- enrich result text
  +-- append committed IR operation
  +-- create operation checkpoint
```

## Realization Flow: build123d / Studio Target

```text
execute_request with execution_target = build123d | studio
  |
  +-- LLM tool_use
  |
  +-- IR mapping
  |     ir/mapper.py
  |       -> IROperation
  |     IRDocumentState
  |       -> committed operation list
  |
  +-- build123d target adapter
        backends/build123d/translator.py
          IRDocument -> executable build123d Python source
        backends/build123d/executor.py
          exec generated program server-side
          cache part by session_id
          extract entities from real geometry
        backends/build123d/entity_extractor.py
          -> entity metadata
        |
        v
      backend -> UI
        studio_geometry_update
        ir_operation_committed

Studio export
  |
  +-- request format=step
  +-- Build123dTargetExecutor.export_step()
  +-- backends/build123d/exporter.py
  +-- runs/studio_exports/<session>_<timestamp>.step
  +-- studio_export_complete { format, path }
```

## Planning Flow

```text
planning_request
  |
  +-- clear conversation, checkpoints, operation checkpoints, IR state
  +-- prepopulate entity refs from entity_context
  +-- append formatted entity context to planning request
  |
  v
generate_plan()
  |
  +-- model provider path:
  |     managed Bedrock gateway
  |     managed Bedrock direct
  |     OpenAI Responses streaming
  |     Gemini streaming
  |     Anthropic streaming
  |
  +-- stream to add-in:
        reasoning_chunk
        plan_chunk
  |
  v
plan_complete
  |
  +-- full_plan
  +-- display_plan
  +-- display_plan_plain
  |
  v
Add-in plan approval dialog
  |
  +-- plan_approval approved=false
  |     -> llm_message rejection text
  |
  +-- plan_approval approved=true
        -> handle_execute_request with plan_text
```

## Revert and Resume Flow

```text
Message checkpoint
  |
  +-- captured before a new user message
  +-- stores timeline marker/count, conversation index, runtime context
  +-- emits checkpoint_created

Operation checkpoint
  |
  +-- captured after successful operation
  +-- stores tool info, conversation slice, IR state, feature/entity context
  +-- emits operation_checkpoint_created

revert_request { message_id }
  |
  +-- lookup message checkpoint
  +-- backend -> add-in: revert_timeline
  +-- add-in mutates Fusion timeline when available
  +-- add-in -> backend: execution_result
  +-- backend trims conversation/checkpoints
  +-- restores runtime state when geometry was reverted
  +-- emits revert_applied + log

resume_operation_request { checkpoint_id }
  |
  +-- lookup operation checkpoint
  +-- backend -> add-in: revert_timeline
  +-- add-in reverts Fusion timeline
  +-- backend restores entity store, conversation, IR state
  +-- emits operation_resume_applied + log
```

## Context Refresh Flow

```text
Backend needs current geometry context
  |
  +-- request_entity_context
  |     context_request_id / message_id correlation
  |     reason
  |
  v
Fusion add-in
  |
  +-- extract_entity_context()
  |     bodies
  |     faces
  |     edges
  |     spatial relationships
  |
  +-- retry short transient empty contexts
  |
  v
entity_context_response
  |
  v
Backend
  |
  +-- pending_entity_context queue
  +-- latest_entity_context cache
  +-- EntityStore short-ref mapping
  +-- stale token pruning
```

```text
Backend needs timeline feature metadata
  |
  +-- feature_snapshot_request
  |     message_id
  |     timeline_count
  |     marker_position
  |     max_features
  |
  v
Fusion add-in
  |
  +-- feature_tools.capture_feature_snapshot()
  |
  v
feature_snapshot
  |
  v
Backend
  |
  +-- feature_snapshot cache
  +-- list_features result source
  +-- timeline tool validation
```

## Message Inventory

Inbound to backend:

```text
authenticate
update_api_keys
execute_request
planning_request
execution_result
execution_error
result
error
feature_snapshot
entity_context_response
plan_approval
question_tree_completed
design_selected
status
cancel_request
revert_request
resume_operation_request
iteration_feedback
```

Outbound from backend:

```text
authentication_ack
authentication_error
auth_error
api_keys_updated
rate_limit_error
checkpoint_created
operation_checkpoint_created
reasoning_chunk
plan_chunk
plan_complete
llm_message
completed
cancelled
error
log
activity_log
runtime_fallback
execute_code
feature_operation
edge_operation
face_operation
body_operation
request_entity_context
feature_snapshot_request
ir_operation
ir_operation_committed
studio_geometry_update
studio_export_complete
question_tree_generated
designs_proposed
build_plan_generated
build_step_completed
build_plan_completed
revert_timeline
revert_applied
operation_resume_applied
iteration_feedback_ack
```

## Persistence and External Data Stores

```text
In-memory backend process state
  |
  +-- WebSocket sessions
  +-- queues
  +-- conversation history
  +-- checkpoints
  +-- entity refs
  +-- reasoning context
  +-- IR document state
  +-- build123d session part cache

Filesystem
  |
  +-- runs/
  |     session logs
  |     session_context.json
  |     llm_response.json
  |
  +-- runs/studio_exports/
  |     STEP exports
  |
  +-- deployment bundle files
        infra/appspec.yml
        infra/hooks/
        apps/backend/backend/
        apps/backend/requirements.txt
        apps/backend/start_backend.py

Supabase
  |
  +-- Auth users/JWTs
  +-- user_entitlements
  +-- usage_ledger
  +-- marketing_preferences
  +-- export_feedback

External providers
  |
  +-- Anthropic
  +-- OpenAI
  +-- Google Gemini
  +-- AWS Bedrock OpenAI-compatible endpoint
  +-- Brevo contact list API
  +-- AWS S3 and CodeDeploy
```

## Deployment and Ops Flow

```text
Developer / GitHub main
  |
  v
GitHub Actions deploy.yml
  |
  +-- checkout
  +-- Python 3.11
  +-- AWS OIDC credentials
  +-- install requirements-dev.txt
  +-- pytest
  +-- zip deployment bundle
  +-- upload to S3
  +-- create CodeDeploy deployment
  +-- wait for deployment-successful
  |
  v
AWS CodeDeploy on EC2
  |
  +-- before_install.sh
  +-- after_install.sh
  +-- start_server.sh
  +-- validate_service.sh
  |
  v
EC2 runtime
  |
  +-- /opt/backend-legacy
  +-- systemd cadagent-backend-legacy.service
  +-- uvicorn/FastAPI on 127.0.0.1:8001
  +-- nginx routes ws.cadagentpro.com to backend
  |
  v
Health checks
  |
  +-- https://ws.cadagentpro.com/health
  +-- http://127.0.0.1:8001/health
```

## Source Anchors

- `apps/backend/backend/main.py`: FastAPI routes, WebSocket message dispatch, auth gate.
- `apps/backend/backend/websocket_manager.py`: session stores, queues, connection lifecycle.
- `apps/backend/backend/agent_workflow.py`: execute/planning/revert/resume workflows.
- `apps/backend/backend/attachments.py`: attachment normalization and prompt context.
- `apps/backend/backend/vision_translator.py`: image/sketch-to-text translation.
- `apps/backend/backend/prompt_router.py`: request-to-cluster routing.
- `apps/backend/backend/prompt_builder.py`: prompt and tool assembly.
- `apps/backend/backend/prompt_structure.py`: cluster docs and tool usage contract.
- `apps/backend/backend/llm_client.py`: provider selection, streaming, response normalization.
- `apps/backend/backend/supabase_client.py`: usage-gateway client.
- `supabase/functions/api-generate/index.ts`: quota-enforced managed model gateway.
- `apps/backend/backend/code_generator.py`: legacy Fusion Python code generation.
- `apps/backend/backend/ir/`: shared target-independent IR types, mapper, validator.
- `apps/backend/backend/backends/fusion/`: Fusion target adapter.
- `apps/backend/backend/backends/build123d/`: build123d target adapter and STEP export.
- `apps/backend/backend/entity_store.py`: short ref to Fusion token mapping.
- `apps/backend/backend/session_logger.py`: session/API-call logging.
- `apps/backend/ops/sync_brevo_users.py`: Supabase Auth to Brevo operational list sync.
- `.github/workflows/deploy.yml`: CI/CD path to EC2 production.
- `apps/addin/mac/CADAgent/websocket_client.py`: add-in WebSocket transport.
- `apps/addin/mac/CADAgent/CADAgent.py`: add-in request assembly and message handling.
- `apps/addin/mac/CADAgent/code_executor.py`: Fusion-side Python execution.
