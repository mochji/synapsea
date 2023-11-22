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

local tempBackup = {
	SYNAPSEA_PATH    = SYNAPSEA_PATH,
	SYNAPSEA_VERSION = SYNAPSEA_VERSION,
	canindex         = canindex
}

local synapseaPath    = debug.getinfo(1).short_src:match("(.*[/\\])") or ""
local synapseaVersion = "v2.0.00-unstable"

SYNAPSEA_PATH    = synapseaPath
SYNAPSEA_VERSION = synapseaVersion

function canindex(item)
	return type(item) == "table" or (type(item) == "userdata" and getmetatable(item).__index)
end

local synapsea = {
	version      = synapseaVersion,
	activations  = require(synapseaPath .. "core.activations"),
	losses       = require(synapseaPath .. "core.losses"),
	math         = require(synapseaPath .. "core.math"),
	initializers = require(synapseaPath .. "core.initializers"),
	optimizers   = require(synapseaPath .. "core.optimizers"),
	regularizers = require(synapseaPath .. "core.regularizers"),
	model        = require(synapseaPath .. "core.model.model"),
	layers       = {}
}

do
	local layersModule   = require(synapseaPath .. "core.layers.layers")
	local buildModule    = require(synapseaPath .. "core.layers.build")
	local errorModule    = require(synapseaPath .. "core.layers.error")
	local gradientModule = require(synapseaPath .. "core.layers.gradient")

	for layerName, layerFunc in pairs(layersModule) do
		synapsea.layers[layerName] = setmetatable(
			{
				build    = buildModule[layerName],
				error    = errorModule[layerName],
				gradient = gradientModule[layerName]
			},
			{
				__call = function(_, args)
					return layerFunc(args)
				end
			}
		)
	end
end

SYNAPSEA_PATH = tempBackup.SYNAPSEA_PATH
canindex      = tempBackup.canindex

return synapsea
