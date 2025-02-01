# metacycle

**Metacycle** is an open source cycling simulator. It pairs with smart indoor cycling accessories to offer the highest degree of in-game control and feedback, providing the ultimate interactive indoor cycling experience.

## Requirements

1. A bike, such as [this absolute beauty](https://www.bastioncycles.com/)
2. A smart steering plate (such as [Elite Sterzo](https://www.elite-it.com/en/products/home-trainers/ecosystem-accessories/sterzo-smart))
3. A smart Trainer (such as [Elite Suito T](https://www.elite-it.com/en/products/home-trainers/interactive-trainers/suito-t))
4. A gaming computer running Windows 10 or Ubuntu 22.04. See [more detailed requirements](#computer-requirements)

## Quickstart

### 1. Install Carla

Download [Carla Simulator 0.9.15](https://github.com/carla-simulator/carla/releases/tag/0.9.15/) from the Github Release.

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

# The Story

The very first iteration of Metacycle was created in 2021 by [tensorturtle](https://github.com/tensorturtle) for a fun and imaginative 'future of cycling vision' demo by Boreal Bikes GmbH at the [Bike2CAV](https://www.bike2cav.at/en/home-2/) consortium presentation at Salzburg Research.

In 2023, Metacycle was officially adopted and re-written by [Velovision Labs](https://github.com/velovision) as an AI training and validation environment. It was crucial to the development of [Velovision Rearview](https://velovision.app) and its advanced computer vision algorithms. See Tesla's [simulation presentation (YouTube)](https://www.youtube.com/live/j0z4FweCy4M?si=XWvyaFaxmshTBO1n&t=5715) to get a sense of how CyCARLA is used at Velovision Labs.

This project is a free, open source fork maintained by [tensorturtle](https://github.com/tensorturtle), the original author of Metacycle, with the goal of creating a superior, free, and modifiable alternative to indoor cycling games like Zwift.

# More Details

## Install CARLA Simulator on Windows

Download [CARLA 0.9.15 pre-compiled ZIP for Windows](https://carla-releases.s3.eu-west-3.amazonaws.com/Windows/CARLA_0.9.15.zip). Other versions can be found [here](https://github.com/carla-simulator/carla/releases)

Unzip it and find `CarlaUE4.exe`. Double click to launch it.

Right-click on the icon and pin it to the taskbar.

## Install CARLA Simulator on Ubuntu

Download [CARLA 0.9.15 pre-compiled TAR.GZ for Ubuntu](https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz). 

Un-tar it and run `./CarlaUE4.sh` script to launch it.

## Open Source

CYCARLA accepts Pull Requests. If you have an improvement idea please go ahead and submit and Issue. Let's discuss!

CyCARLA is made possible thanks to the following open source projects:
+ [Unreal Engine](https://github.com/EpicGames)
+ [CARLA](https://github.com/carla-simulator/carla)
+ [Pycycling](https://github.com/zacharyedwardbull/pycycling) - major contributions made to this project by [tensorturtle](https://github.com/tensorturtle)


## Computer Requirements

CYCARLA is based Unreal Engine 4, a serious 3D game engine with full customizability, so it'll be more demanding than typical indoor cycling games.

+ CPU: Intel Core i5 6th-gen or better.
+ GPU: NVIDIA RTX 2070 or better (2080, 3060, 3070, 3080, 4060, 4070, 4080 etc.)
+ Hard drive: 30GB of free space.
+ Bluetooth Low Energy (BLE) support.
+ Internet connection required for installation, not required to run the game.

