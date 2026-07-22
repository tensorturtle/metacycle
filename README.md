# metacycle

**Metacycle** is an open source cycling simulator. It pairs with smart indoor cycling accessories to offer the highest degree of in-game control and feedback.

`metacycle` is to **Zwift** as **Project Cars** is to **Mario Kart**.

In other words, it's a serious simulator, not an arcade game. Metacycle has lifelike physics, sophisticated graphics, and deeply modifiable maps and environment.


## Requirements

1. Your Bike
2. Smart steering plate
3. Smart Trainer
4. Gaming computer running Windows 10+ or Ubuntu 20.04 / 22.04

> **Linux/Ubuntu version matters.** CARLA (UE4 4.26) is only supported on Ubuntu 20.04 and 22.04. It does **not** run on Ubuntu 24.04 / Pop!_OS 24.04: the newer glibc (2.39) makes the CARLA server crash on startup with heap corruption (`malloc(): unsorted double linked list corrupted`, Signal 6) or an intermittent `Signal 11` segfault. Use 20.04/22.04, or run the CARLA server in a container based on a supported Ubuntu (see Troubleshooting).

Currently, the only smart steering plate on the market is [Elite Sterzo](https://www.elite-it.com/en/products/home-trainers/ecosystem-accessories/sterzo-smart). Any smart trainer supporting Bluetooth FTMS by Elite, JetBlack, Wahoo, Tacx, etc. are compatible.

*No affiliation with any manufacturers.*

## Quickstart

### 1. Install Carla

Download [Carla Simulator 0.9.15](https://github.com/carla-simulator/carla/releases/tag/0.9.15/) from the Github Release.

#### Windows

Open the downloaded ZIP file. Uncompress it to a location of your choice and find `CarlaUE4.exe`. Double click to launch it.

> **Note:** On a brand-new Windows installation, the first launch may prompt you to install DirectX. Accept and complete that install, then launch CARLA again.

#### Linux (Ubuntu)

Un-tar it:
```
tar -xvf CARLA_0.9.15.tar.gz
```

Run the launch script:
```
./CarlaUE4.sh
```

### 2. Install Metacycle

Install UV, a python package manager (instructions [here](https://docs.astral.sh/uv/getting-started/installation/))

Then, download the required python version and metacycle app. From the terminal or powershell:
```
uv python install 3.10
uv tool install metacycle --python 3.10
```
This will install `metacycle` to your system, so that it can be launched directly.

Launch metacycle
```
metacycle
```

You can pass in a `--map ` argument from available CARLA maps. For example:
```
metacycle --map Town10HD
```
A list and description of the maps are available from [CARLA docs](https://carla.readthedocs.io/en/latest/core_map/#carla-maps)


## Read More

+ [STORY.md](STORY.md) - The history of this project.
+ [CONTRIBUTING.md](CONTRIBUTING.md) - For developers who want to contribute to this project.

## FAQ

### How do I upgrade to the latest version?

Sometimes, the cache needs to be cleared in order to get the latest version.
```
uv cache clean
```

The following command will find and install the latest published version.
```
uv tool upgrade metacycle
```

CARLA must be re-installed manually. Please refer to [CARLA docs](https://carla.readthedocs.io/en/latest/start_quickstart/#updating-carla)

### How do I remove/uninstall the program?

```
uv tool uninstall metacycle
```

CARLA must be uninstall manually by deleting the program yourself.

### What is BLE? How do I know if my cycling accessories or computer support it?

Most modern 'smart' cycling accessories communicate using Bluetooth Low Energy. It is sometimes abbreviated to 'BLE'. Bluetooth 4 may or may not support 'Low Energy', so check with the computer manufacturer. Bluetooth 5 and 6 always includes support for 'Low Energy'. Most cycling accessories are not compatible with older, non-'Low Energy' versions of Bluetooth.

### My cycling accessories (such as powermeter) support ANT+ but not Bluetooth. Can I use it with metacycle?

No, ANT+ is a completely different protocol that is not supported for this project.

### Why do I need a high-performance gaming computer to run metacycle?

Metacycle is based Unreal Engine 4, a serious 3D game engine with full customizability, so it'll be more demanding than typical indoor cycling games.

+ CPU: Intel Core i5 6th-gen or better.
+ GPU: NVIDIA RTX 2070 or better (2080, 3060, 3070, 3080, 4060, 4070, 4080 etc.)
+ Hard drive: 30GB of free space.
+ Bluetooth Low Energy (BLE) support.
+ Internet connection required for installation, not required to run the game.

There is work being done upstream (in Carla) to upgrade to Unreal Engine 5, which will have potentially higher system requirements (16GB+ VRAM)

## Troubleshooting

### CARLA server crashes on startup (Ubuntu 24.04 / newer glibc)

Symptoms: `./CarlaUE4.sh` dies during startup with `malloc(): unsorted double linked list corrupted` (Signal 6) or an intermittent `Signal 11` segfault. This is **not** a GPU problem — it's a binary-compatibility issue. CARLA (UE4 4.26) is built for Ubuntu 20.04/22.04 and is incompatible with the glibc 2.39 shipped in Ubuntu 24.04 / Pop!_OS 24.04.

Options:
1. **Use Ubuntu 20.04 or 22.04** (recommended — this is what CARLA officially supports). The native install then works as documented above.
2. **Run the CARLA server in Docker** on a supported base image, keeping the metacycle client on the host:
   ```bash
   docker run --rm -d --name carla --gpus all --net=host \
     carlasim/carla:0.9.16 ./CarlaUE4.sh -RenderOffScreen -nosound
   ```
   This requires Docker + the NVIDIA Container Toolkit. Note: Docker 29.x has a CDI/GPU-passthrough regression on Ubuntu 24.04 — if `--gpus all` fails with "no known GPU vendor found" / "unresolvable CDI devices", use Docker 28.x or the classic `--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all` instead.

### Map 'Town07' not found

The default map requires CARLA's *additional maps* pack. Base packages ship Town01–05 and Town10HD. Either install the additional maps (place `AdditionalMaps_<ver>.tar.gz` in the CARLA package's `Import/` folder and run `./ImportAssets.sh`), or pass an available map, e.g. `--map Town10HD`.


