# metacycle

**Metacycle** is an open source cycling simulator. It pairs with smart indoor cycling accessories to offer the highest degree of in-game control and feedback, providing the ultimate interactive indoor cycling experience.

`metacycle` is to **Zwift** as **Project Cars** is to **Mario Kart**.

In other words, it's a serious simulator, not an arcade game. Metacycle has lifelike physics, sophisticated graphics, and deeply modifiable maps and environment.


## Requirements

1. Your Bike
2. Smart steering plate
3. Smart Trainer
4. Gaming computer running Windows 10+ or Ubuntu 22.04+

Currently, the only smart steering plate on the market is [Elite Sterzo](https://www.elite-it.com/en/products/home-trainers/ecosystem-accessories/sterzo-smart). Any smart trainer supporting Bluetooth FTMS by Elite, JetBlack, Wahoo, Tacx, etc. are compatible.

*No affiliation with any manufacturers.*

## Quickstart

### 1. Install Carla

Download [Carla Simulator 0.9.15](https://github.com/carla-simulator/carla/releases/tag/0.9.15/) from the Github Release.

#### Windows

Open the downloaded ZIP file. Uncompress it to a location of your choice and find `CarlaUE4.exe`. Double click to launch it.

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


