<p align="center">
    <img src="https://github.com/mochji/synapsea/assets/117334318/abec23f1-06ee-47cc-8685-70589b3ba7d1">
</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Lua](https://img.shields.io/badge/Lua-5.4%2B-blueviolet)](https://www.lua.org/)
[![Release](https://img.shields.io/github/v/release/mochji/synapsea)](https://github.com/mochji/synapsea/releases)

## Overview

Synapsea is a simple yet powerful machine learning library designed for building, saving, training and running powerful machine learning models.

Synapsea is built from the ground up to be simple, easy to understand and portable with no external libraries, requring only the Lua interpreter.

## Table of Contents

 - [Installation](#installation)
 - [Usage](#usage)
 - [Examples](#examples)
 - [Documentation](#documentation)
 - [Contributing](#contributing)
 - [License](#license)

## Installatation

Synapsea is very simple to install, either clone the Github repository using Git:

```
git clone https://github.com/mochji/synapsea
```

Or download the zip file from Github and unzip it.

Once you have cloned the Synapsea repository just move or copy it to a directory where you can `require` it.

## Usage

To use Synapsea, just `require` the library:

```lua
local synapsea = require("synapsea")
```

## Examples

Creating a simple CNN:

```lua
local synapsea = require("synapsea")

local model = synapsea.model.new(
    {64, 64, 3}, -- Input shape (image height, image width, image color channels)
    {            -- Model metadata
        name = "Example CNN",
        description = "An example CNN for the README.md file in the Synapsea Github repository.",
        author = "mochji (Skye)",
        version = 1
    }
)

model:addLayer(
    "maxPooling3D",
    {
        kernel = {2, 2}
    }
)

model:addLayer(
    "convolutional3D",
    {
        activation = "leakyRelu",
        alpha = 0.1,
        filters = 16,
        kernel = {3, 3},
        filterInitializer = "normalRandom",
        filterInitParameters = {
            mean = 0,
            sd = 0.1
        },
        biasInitializer = "constant",
        biasInitParameters = {
            value = 0.1
        },
        filterTrainable = true,
        biasTrainable = true,
        useBias = true
    }
)

model:addLayer("flatten")

model:addLayer(
    "dense",
    {
        activation = "sigmoid",
        outputSize = 10,
        weightsInitializer = "uniformRandom",
        weightsInitParameters = {
            lowerLimit = -0.1,
            upperLimit = 0.1
        },
        weightsTrainable = true
    }
)

model:summary()

model:initialize(
    "momentum", -- Optimizer
    {           -- Optimizer parameters
        momentum = 0.9,
        stepSize = 0.1
    },
    "l1",       -- Regularizer
    {           -- Regularizer parameters
        lambda = 0.1 -- tbh idrk what this should be lol
    }
)
```

For more examples, please see [Examples](https://sites.google.com/view/synapsea/api/examples).

## Documentation

For detailed documentation, including API documentation and usage, please see [Documentation](https://sites.google.com/view/synapsea/api).

## Contributing

Contributions are welcome! If you find any issues or have suggestions, please open an issue or create a pull request. Please also make sure to follow the [Contribution Guidelines](https://sites.google.com/view/synapsea/contributing).

## License

This library is licensed under the GNU GPL v3, see [License](https://www.gnu.org/licenses/gpl-3.0.en.html) for details.
