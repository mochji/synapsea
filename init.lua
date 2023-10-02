--[[
	Synapsea v1.3.00-unstable

	A Lua Neural Network library made in pure Lua.

	Read the README.md file for documentation and information, 

	https://github.com/x-xxoa/synapsea
	init.lua

	Synapsea, a machine learning library made in pure Lua.
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

-- import the core files

local synapsea = {
	math = require("core.math"),
	activations = require("core.activations"),
	losses = require("core.losses"),
	initializers = require("core.initializers"),
	optimizers = require("core.optimizers"),
	regularizers = require("core.regularizers"),
	layers = require("core.layers"),
	backProp = require("core.backProp"),
	model = require("core.model"),
	metrics = require("core.metrics"),
	callBacks = require("core.callBacks"),
	array = require("core.array.init")
	debug = require("core.debug")
}

-- add lua math functions into synapsea.math

for mathName, _ in pairs(math) do
	synapsea.math[mathName] = math[mathName]
end

-- convert all layers to metatables

local layerBuild = require("core.layerBuild")

for layerName, _ in pairs(synapsea.layers) do
	synapsea.layers[layerName] = setmetatable(
		{build = layerBuild[layerName]},
		{__call = synapsea.layers[layerName]}
	)
end

return synapsea
