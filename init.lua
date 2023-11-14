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

SYNAPSEA_PATH = debug.getinfo(1).short_src:match("(.*[/\\])") or ""
SYNAPSEA_VERSION = "v2.0.00-unstable"

local synapsea = {
	version      = SYNAPSEA_VERSION,

	activations  = require(SYNAPSEA_PATH .. "core.activations"),
	losses       = require(SYNAPSEA_PATH .. "core.losses"),
	math         = require(SYNAPSEA_PATH .. "core.math"),
	initializers = require(SYNAPSEA_PATH .. "core.initializers"),
	optimizers   = require(SYNAPSEA_PATH .. "core.optimizers"),
	regularizers = require(SYNAPSEA_PATH .. "core.regularizers"),
	layers       = require(SYNAPSEA_PATH .. "core.layers.layers"),
	backProp     = require(SYNAPSEA_PATH .. "core.backProp"),
	model        = require(SYNAPSEA_PATH .. "core.model")
}

do
	local buildModule    = require(SYNAPSEA_PATH .. "core.layers.build")
	local errorModule    = require(SYNAPSEA_PATH .. "core.layers.error")
	local gradientModule = require(SYNAPSEA_PATH .. "core.layers.gradient")

	for name, func in pairs(synapsea.layers) do
		synapsea.layers[name] = setmetatable(
			{
				build    = buildModule[name],
				error    = errorModule[name],
				gradient = gradientModule[name]
			},
			{__call = func}
		)
	end
end

return synapsea
