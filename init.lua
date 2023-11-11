--[[
	Synapsea v2.0.00-unstable

	A simple yet powerful machine learning library made in pure Lua.

	Read the README.md file for documentation and information, 

	https://github.com/mochji/synapsea
	init.lua

	Synapsea, a simple yet powerful machine learning library made in pure Lua.
	Copyright (C) 2023 mochji
																		   
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

-- Get the path of the required file
_SYNAPSEA_PATH = debug.getinfo(1).short_src:gsub("/", "."):gsub("\\", ".")

-- Remove file extension and file name
_SYNAPSEA_PATH:match("(.*%.)")
_SYNAPSEA_PATH:sub(1, #_SYNAPSEA_PATH - 1)
_SYNAPSEA_PATH:match("(.*%.)")

-- Avoid concatenating nil
_SYNAPSEA_PATH = _SYNAPSEA_PATH or ""

_SYNAPSEA_VERSION = "v2.0.00-unstable"

local synapsea = {
	activations = require(_SYNAPSEA_PATH .. "core.activations"),
	losses = require(_SYNAPSEA_PATH .. "core.losses"),
	math = require(_SYNAPSEA_PATH .. "core.math"),
	initializers = require(_SYNAPSEA_PATH .. "core.initializers"),
	optimizers = require(_SYNAPSEA_PATH .. "core.optimizers"),
	regularizers = require(_SYNAPSEA_PATH .. "core.regularizers"),
	layers = require(_SYNAPSEA_PATH .. "core.layers"),
	backProp = require(_SYNAPSEA_PATH .. "core.backProp"),
	model = require(_SYNAPSEA_PATH .. "core.model")
}

-- Add the layer build call to the layer functions by converting the layers to a metatable

local layerBuildModule = require(_SYNAPSEA_PATH .. "core.layerBuild")

for name, func in pairs(synapsea.layers) do
	synapsea.layers[name] = setmetatable(
		{build = layerBuildModule[name]},
		{__call = func}
	)
end

return synapsea
