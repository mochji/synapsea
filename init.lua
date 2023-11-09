--[[
	Synapsea v2.0.00-unstable

	A simple yet powerful machine learning library made in pure Lua.

	Read the README.md file for documentation and information, 

	https://github.com/x-xxoa/synapsea
	init.lua

	Synapsea, a simple yet powerful machine learning library made in pure Lua.
	Copyright (C) 2023 x-xxoa
																		   
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.
																		   
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
																		   
	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <https://www.gnu.org/licenses/>.
]]--

-- Require the libraries to avoid importing them in the core files over and over again
local mathModule = require("core.math")
local activationsModule = require("core.activations")
local layerBuildModule = require("core.layerBuild")
local initializersModule = require("core.initializers")
local layersModule = require("core.layers")

local synapsea = {
	version = "v2.0.00-unstable",
	activations = require("core.activations"),
	losses = require("core.losses"),
	math = require("core.math"),
	initializers = require("core.initializers"),
	optimizers = require("core.optimizers"),
	regularizers = require("core.regularizers"),
	layers = require("core.layers"),
	backProp = require("core.backProp"),
	model = require("core.model")
}

-- Convert all layers to metatables
for name, func in pairs(synapsea.layers) do
	synapsea.layers[name] = setmetatable(
		{build = layerBuildModule[name]},
		{__call = func}
	)
end

return synapsea
