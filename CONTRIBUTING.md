# Contributing to Metacycle

Metacycle accepts Pull Requests. If you have an improvement idea, create a Github issue so that we can discuss how best to approach it.

# Development Setup

Clone this repository.

Install uv.

```
uv python install 3.10
uv run --python 3.10 metacycle
```

At each version increment, the main branch is built into a python package and uploaded to PyPI like so:
```
rm -r dist/
uv build 
uv publish --token $PYPI_TOKEN
```

# We ♥ Open Source

Metacycle is made possible thanks to the following open source projects:
+ [Unreal Engine](https://github.com/EpicGames)
+ [CARLA](https://github.com/carla-simulator/carla)
+ [Pycycling](https://github.com/zacharyedwardbull/pycycling) - major contributions made to this project by [tensorturtle](https://github.com/tensorturtle)