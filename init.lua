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

local synapseaVersion = "v2.0.00-unstable"
local synapseaPath = 
	debug.getinfo(1, "S").source
		:sub(2)
		:match("(.*" .. package.config:sub(1, 1) .. ")")

local oldPackagePath = package.path
package.path = synapseaPath .. "?.lua"

local synapsea = {
	path         = synapseaPath,
	version      = synapseaVersion,

	activations  = require("core.activations"),
	losses       = require("core.losses"),
	math         = require("core.math"),
	initializers = require("core.initializers"),
	optimizers   = require("core.optimizers"),
	regularizers = require("core.regularizers"),
	layers       = {},
	model        = require("core.model.model")
}

package.path = oldPackagePath

return synapsea
