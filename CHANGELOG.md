# Changelog

All notable changes to this project will be documented in this file.

# [1.9.10] - 2025-06-16

### Added

- N/A

### Changed

- Removed LLMessage generated after tool abortion.

# [1.9.9] - 2025-06-13

### Added

- N/A

### Changed

- Made `abort_after_execution` end message as default `False`.

# [1.9.8] - 2025-06-13

### Added

- Added test version of `abort_after_execution`

### Changed

- N/A

# [1.9.7] - 2025-06-11

### Added

- N/A

### Changed

- N/A

### Fixed

- Added more fiels to `LLMessage` type.
- Removed useless checks on `tool` engine.

# [1.9.6] - 2025-06-06

### Added

- N/A

### Changed

- N/A

### Fixed

- Added more defensive checks against `dict` conversion to `LLMessages`.

# [1.9.5] - 2025-05-29

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed random chain_id assigment on graph execution.

# [1.9.4] - 2025-05-12

### Added

- N/A

### Changed

- N/A

### Fixed

- Removed uncessary dependencies from the core list.

# [1.9.3] - 2025-05-12

### Added

- N/A

### Changed

- N/A

### Fixed

- Improved type checking on `tool` signature param check.

# [1.9.2] - 2025-05-12

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed the parameters type checking on `tool` runtime.

# [1.9.1] - 2025-04-28

### Added

- Added more logging on how message history is being passed to the `ToolGraph` on each run.

### Changed

- N/A

### Fixed

- Fixed case where empty strings ("") where being added to message history on `ToolGraph` loops.

# [1.9.0] - 2025-04-28

### Added

- Option to hide parameters from `tool` function schema.

### Changed

- N/A

### Fixed

- N/A

# [1.8.5] - 2025-04-28

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed `ToolState` inheritance not loading converting pydantic models of new added fields.

# [1.8.4] - 2025-04-27

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed `ToolState` inheritance not loading checkpoint data correctly on load.

# [1.8.3] - 2025-04-02

### Added

- N/A

### Changed

- Added id and other fields to `raw_response_history` and `LLLMessage`.

### Fixed

- N/A

# [1.8.2] - 2025-04-01

### Added

- N/A

### Changed

- Added new event to `StreamingType` : `message_start`

### Fixed

- N/A

# [1.8.1] - 2025-03-31

### Added

- N/A

### Changed

- Allowed the user the change the `type` value for the Redis message object.

### Fixed

- N/A

# [1.8.0] - 2025-03-30

### Added

- Added Redis-based streaming functionality for real-time event processing
- New `StreamingConfig` and `StreamingEventType` classes for configuring streaming capabilities
- Support for multiple event types including text fragments and content block delimiters
- Comprehensive Redis integration for streaming LLM responses
- Added examples for consuming streaming events from Redis

### Changed

- Enhanced documentation for setting up and using Redis with primeGraph
- Updated Docker configuration to better support streaming features

### Fixed

- N/A

# [1.7.2] - 2025-03-20

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed how state is passed to the graph instance on `ToolGraph` and removed it from execute method.

# [1.7.1] - 2025-03-20

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed chain_id assigment on `ToolGraph` execution. Graph should keep the chain_id unless execution assigned a new chain_id.

# [1.7.0] - 2025-03-20

### Added

- Added `should_show_to_user` flag to `LLMMessage` class to allow easy identification of messages that should be displayed to end users.
- Automatically sets `should_show_to_user=False` for system messages, tool results, and LLM-internal messages.
- Final assistant messages are automatically marked with `should_show_to_user=True`.
- Enhanced state management in tool functions - tools can now directly access and modify custom state fields by accepting a `state` parameter
- Support for state-aware tools that can keep track of their own context across multiple executions

### Changed

- Removed the exclude flag on `raw_response_history` so it's also saved with checkpoints

### Fixed

- Fixed redundant wrapper function definition in `tool` decorator, removing linter warning about redefinition of unused variable.
- Improved state parameter handling in tool functions by simplifying the wrapper implementation.

# [1.6.1] - 2025-03-17

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed save_checkpoint parsing errors from ToolGraph.

# [1.6.0] - 2025-03-15

### Added

- N/A

### Changed

- Moved engine execution to run inside the ToolGraph class as we currently have with Graph, for consistency and easy of use.

### Fixed

- N/A

# [1.5.0] - 2025-03-15

### Added

- Added proper ChainStatus synchronization with tool pause states in ToolGraph and ToolEngine
- Enhanced ChainStatus handling in ToolEngine's execute, resume, and \_execute_all methods
- Added safeguards to ensure ToolState.is_paused flag and ChainStatus are always consistent

### Changed

- Updated ToolEngine.resume to properly set ChainStatus.FAILED on error
- Improved ChainStatus state management in ToolEngine execution and checkpointing

### Fixed

- Fixed issue where ChainStatus wasn't properly updated to PAUSE when tools paused execution
- Fixed state restoration to ensure ChainStatus matches is_paused state after loading from checkpoint

# [1.4.0] - 2025-03-14

### Added

- Added `on_message` callback for LLM messages in ToolNode
- Enhanced checkpointing for ToolEngine to properly save and restore paused tool state
- Added support for preserving tool call and message history in PostgreSQL checkpoints
- Added examples for LLM message callbacks and PostgreSQL checkpointing with paused tools
- Added comprehensive tests for PostgreSQL checkpointing with tool pauses

### Changed

- Improved state serialization/deserialization for complex objects in checkpoints

### Fixed

- Fixed issue where pause state attributes weren't being properly stored in checkpoints
- Fixed restoration of tool execution state and message history from checkpoints

# [1.3.0] - 2025-03-13

### Added

- Added `pause_after_execution` feature for LLM tools, complementing the existing `pause_before_execution` feature
- Enhanced ToolState with new fields to support post-execution pauses
- Updated the `tool` decorator to accept the new `pause_after_execution` parameter
- Extended `resume_from_pause` method to handle both pre-execution and post-execution pauses
- Added comprehensive tests for the new feature

### Changed

- The `resume_from_pause` method now differentiates between pre-execution and post-execution pauses

### Fixed

- N/A

# [1.2.4] - 2025-03-12

### Added

- N/A

### Changed

- N/A

### Fixed

- Moved away from token usage calculations and started logging all the raw responses from LLM calls.

# [1.2.3] - 2025-03-12

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed LLM calls token usage reporting.

# [1.2.2] - 2025-03-12

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed LLM client model setting and other api kwargs.

# [1.2.1] - 2025-03-12

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed Anthropic tool schema format errors.

# [1.2.0] - 2025-03-11

### Added

- Added support for ToolGraphs: let you create graphs (with 1 or more nodes) that accecpt tools and run tool loops automatically.

### Changed

- N/A

### Fixed

- N/A

---

# [1.1.5] - 2025-02-20

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed Any typing on all the buffers.

---

# [1.1.4] - 2025-02-20

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed deep nested model serialization issues with state.

---

# [1.1.3] - 2025-02-20

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed missing checkpoint when the END node is reached.

---

# [1.1.2] - 2025-02-20

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed issue with state serialization with nested BaseModels on loading state from database.

---

# [1.1.1] - 2025-02-20

### Added

- N/A

### Changed

- N/A

### Fixed

- Engine graph state was being saved as a duplicate and loaded from there. Now it's not saved anymore and it's loaded from the graph.state latest checkpoint.

---

# [1.1.0] - 2025-02-17

### Added

- Now users can pass a list of node names to the `add_repeating_edge` method to assign custom names to the repeated nodes.
- Added `is_repeat` metadata to nodes that are part of a repeating edge.
- Added `original_node` metadata to nodes that are part of a repeating edge.
- Allowed the user to retrieve Node information at execution on the node function body by using the `self.node` attribute.

### Changed

- N/A

### Fixed

- N/A

---

# [1.0.0] - 2025-03-01

### Added

- Introduction of new asynchronous engine methods `execute()` and `resume()` which replace the old synchronous methods such as `start()` and `resume_async()`.
- New examples and documentation updates demonstrating how to run workflows using asyncio (e.g., using `asyncio.run(...)`).
- Enhanced checkpoint persistence with improved saving and loading of engine state.

### Changed

- Refactored engine internals for better handling of parallel execution, flow control, and convergence points.
- Updated ChainStatus to be a string (for enhanced debugging clarity).
- Updated state management and buffer validation error messages.

### Fixed

- Addressed issues with buffer type validation and provided clearer error messages.
- Fixed several issues in the engine related to node routing and cyclical workflows.

---

# [0.2.6] - 2025-02-02

### Added

- N/A

### Changed

- N/A

### Fixed

- Changed ChainStatus enum to be a string instead of an integer so it's easier to debug.

# [0.2.5] - 2025-02-02

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed base.py on \_analyze_router_paths method that was not checking if the router node has any possible routes.

# [0.2.4] - 2025-02-02

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed base.py build_path when working with cyclical graphs.

# [0.2.3] - 2025-01-23

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed variable cleaning on resume_async method that was making it get stuck.

# [0.2.2] - 2025-01-23

### Added

- N/A

### Changed

- N/A

### Fixed

- Fixed the `END` node not flagging the chain as done.
- Added better error messages for buffer type validation.

# [0.2.1] - 2025-01-14

### Added

- N/A

### Changed

- N/A

### Fixed

- Added ChainStatus.DONE when END node is reached as before it was not flagging the chain as done.

## [0.2.0] - 2025-01-09

### Added

- Added `set_state_and_checkpoint` method to `Graph` class.
- Added `update_state_and_checkpoint` method to `Graph` class.

### Changed

- N/A

### Fixed

- N/A
