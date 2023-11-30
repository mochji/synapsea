<p align="center">
    <img src="https://github.com/mochji/synapsea/assets/117334318/abec23f1-06ee-47cc-8685-70589b3ba7d1">
</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Lua](https://img.shields.io/badge/Lua-5.4%2B-blueviolet)](https://www.lua.org/)
[![Release](https://img.shields.io/github/v/release/mochji/synapsea)](https://github.com/mochji/synapsea/releases)

## Overview

Synapsea is a simple yet powerful machine learning library designed for building, saving, training and running powerful machine learning models.

Synapsea is built from the ground up to be simple, easy to understand and portable with no external libraries, requring only a Lua interpreter.

## Table of Contents

 - [Installation](#installation)
 - [Usage](#usage)
 - [Examples](#examples)
 - [Documentation](#documentation)
 - [Contributing](#contributing)
 - [License](#license)

## Installatation

Synapsea is very simple to install, either clone the GitHub repository using Git:

```
git clone https://github.com/mochji/synapsea.git
```

Or download the zip file from GitHub and unzip it.

Once you have cloned the Synapsea repository just move or copy it to a directory where you can `require` it.

## Usage

To use Synapsea, just `require` the library:

```lua
local synapsea = require("synapsea")
```

## Examples

Try the Synapsea API:

```lua
> synapsea = require("synapsea")
> synapsea.version
v2.0.00-unstable
> synapsea.path
/usr/share/lua/5.4/synapsea/
> synapsea.activations.sigmoid(tonumber(io.read()))
2.9
0.94784643692158
> input = synapsea.initializers.normalRandom({3, 3}, {mean = 0, sd = 0.1})
> output = synapsea.layers.flatten{input = input}
> for a = 1, #output do print(a, output[a]) end
1       0.12884759243026
2       0.16374163630011
3       0.076140250714198
4       0.078038014880484
5       0.037395010665463
6       -0.13098177989544
7       0.032318751265418
8       -0.14966043873079
9       0.33041979998093
```

For more complex examples, please see [Examples](https://sites.google.com/view/synapsea/api/examples).

## Documentation

For detailed documentation, including API documentation and usage, please see [Documentation](https://sites.google.com/view/synapsea/api/documentation).

## Contributing

Contributions are welcome! If you find any issues or have suggestions, please open an issue or create a pull request. Please also make sure to follow the [Contribution Guidelines](https://sites.google.com/view/synapsea/contributing).

## License

This library is licensed under the GNU GPL v3, see [License](https://www.gnu.org/licenses/gpl-3.0.en.html) for details.
