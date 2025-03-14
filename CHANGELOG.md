# Changelog

All notable changes to this project will be documented in this file.

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
