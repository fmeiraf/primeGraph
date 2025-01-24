# Changelog

All notable changes to this project will be documented in this file.

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
