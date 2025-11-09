# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Metacycle is an open-source cycling simulator that pairs with smart indoor cycling accessories (smart trainers and steering plates) to offer realistic in-game control and feedback. It's built on top of CARLA Simulator 0.9.15 (Unreal Engine 4) and uses Bluetooth Low Energy to communicate with cycling equipment.

**Key Technologies:**
- CARLA Simulator 0.9.15 (Unreal Engine 4 based)
- Python 3.10 (required for CARLA compatibility)
- Pygame for rendering and UI
- Bleak for Bluetooth Low Energy communication
- Pycycling for cycling equipment protocols (FTMS, Sterzo)

## Development Commands

### Setup
```bash
# Install UV package manager first (see https://docs.astral.sh/uv/getting-started/installation/)
uv python install 3.10
uv run --python 3.10 metacycle
```

### Running the Application
```bash
# Run with default map (Town07)
uv run --python 3.10 metacycle

# Run with specific map
uv run --python 3.10 metacycle --map Town10HD

# Run with custom resolution
uv run --python 3.10 metacycle --res 2560x1440

# Run with custom FPS limit
uv run --python 3.10 metacycle --max-fps 120
```

### Building and Publishing
```bash
# Clean previous builds
rm -r dist/

# Build package
uv build

# Publish to PyPI (requires PYPI_TOKEN environment variable)
uv publish --token $PYPI_TOKEN
```

## Architecture

### Single-File Architecture
The entire application is contained in `metacycle.py` (~1670 lines). This is intentional for simplicity and ease of distribution as a PyPI package.

### Main Components

**1. Bluetooth Management (`BluetoothManager`, `PycyclingInput`)**
- Runs in a separate thread with its own asyncio event loop
- Scans for BLE cycling accessories (Sterzo steering, FTMS trainers)
- Updates `LiveControlState` with real-time inputs from equipment
- Located in metacycle.py:387-473

**2. Control Systems**
- `PycyclingControl` (metacycle.py:685-748): Translates BLE inputs to CARLA vehicle controls
  - Speed control via throttle/brake based on wheel speed from smart trainer
  - Steering with speed-dependent sensitivity (exponential decay)
  - Road gradient simulation for downhill speed gains
- `KeyboardControl` (metacycle.py:755-875): Fallback keyboard controls for testing
- Users can toggle between control modes via button or keyboard

**3. Vehicle Physics and Simulation**
- `World` class (metacycle.py:480-680): Manages CARLA world, vehicle spawning, sensors
- `RoadGradientEstimator` (metacycle.py:119-198): Calculates road gradient using moving average filter
  - Uses frame skipping to avoid numerical instability from small time deltas
  - Configured with window_size=5, ignore_first_n=30 updates
- Resistance calculation: Smart trainer resistance is dynamically set based on road gradient + simulated wind resistance

**4. Sensors and Telemetry**
- `GnssSensor`: GPS location tracking (metacycle.py:1035-1055)
- `IMUSensor`: Accelerometer, gyroscope, compass (metacycle.py:1063-1094)
- `CameraManager`: Handles camera views and rendering (metacycle.py:1103-1192)
  - Two camera positions: chase cam and rider POV
  - Toggle with Tab key or UI button

**5. GPX Export**
- `GPXCreator` (metacycle.py:1289-1377): Records ride data and exports to GPX format
- Creates trackpoints at ~1Hz (every 60 frames)
- Includes power and cadence data in Garmin TrackPointExtension format
- GNSS coordinates offset to Galapagos Islands to avoid distance calculation issues
- Auto-saves to Downloads folder when application exits
- Strava-compatible output format

**6. HUD and UI**
- `HUD` class (metacycle.py:881-997): Displays telemetry overlay and bottom panel
- `Button` class (metacycle.py:1198-1237): Interactive UI buttons
- `NumericInput` (metacycle.py:1239-1280): Gradient offset control widget
- Snapshot feature: Press 'p' or click button to save screenshots to Downloads

### Data Flow

1. **Input Flow**: BLE devices → `BluetoothManager` → `PycyclingInput` → `LiveControlState` → `PycyclingControl` → CARLA vehicle
2. **Simulation Flow**: CARLA physics → sensors → `SimulationOutputs` → HUD + resistance calculation
3. **Resistance Feedback**: Road gradient + speed → resistance level → FTMS trainer

### BLE Cycling Services

- **Sterzo**: `347b0001-7635-408b-8918-8ff3949ce592` (Elite proprietary steering)
- **FTMS**: `00001826-0000-1000-8000-00805f9b34fb` (Fitness Machine Service for smart trainers)
- **Power Meter**: `00001818-0000-1000-8000-00805f9b34fb` (Cycling Power Service)

### Key Configuration Values

- Vehicle blueprint: `vehicle.diamondback.century` (bicycle model)
- Steering sensitivity: `math.exp(-speed/40)` - decreases exponentially with speed
- Resistance calculation: `(gradient * 200 / 15) + (speed * 1.3)` where 200 is max resistance at 15% gradient
- GPX sampling: Every 60 frames (~1Hz at 60fps)
- Road gradient window: 5 samples with 15-frame skip between calculations

## Important Notes

- Python 3.10 is strictly required due to CARLA 0.9.15 dependency
- CARLA Simulator must be running separately before launching metacycle
- Bluetooth Low Energy (BLE) support required - not ANT+
- On Linux, the app automatically restarts system bluetooth on startup to handle ungraceful shutdowns
- The application uses `uv` as package manager (not pip or poetry)
- GPX files are automatically saved on exit, even if the app crashes

## Platform-Specific Behavior

- **Linux**: Auto-restarts bluetooth via `bluetoothctl` on startup (metacycle.py:399-406)
- **Windows**: Skips bluetooth restart, uses different default font rendering
- **macOS**: Uses standard Unix-style paths and XDG directory lookup

## Testing Without Hardware

Set `keyboard_override = True` to enable keyboard controls (WASD/arrow keys) for testing without BLE cycling equipment. Toggle during runtime with the UI button or by modifying the initial value in `game_loop()` (metacycle.py:1511).
