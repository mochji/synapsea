--[[
	Synapsea v2.0.00-development

	Read the README.md file for documentation and information, 

	https://github.com/mochji/synapsea
	init.lua

	Synapsea, simple yet powerful machine learning framework for Lua.
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

local synapseaPath =
	debug.getinfo(1, "S").source
		:sub(2)
		:match("(.*" .. package.config:sub(1, 1) .. ")")
		or "." .. package.config:sub(1, 1)

local oldPackagePath = package.path
package.path = synapseaPath .. "?.lua"

local synapsea = {
	path         = synapseaPath,
	version      = "v2.0.00-development",

	activations  = require("core.activations"),
	losses       = require("core.losses"),
	initializers = require("core.initializers"),
	optimizers   = require("core.optimizers"),
	regularizers = require("core.regularizers"),
	layers       = require("core.layers.layers"),
	model        = {
		Sequential = require("core.model.sequential.Sequential")
	}
}

for layerName, layerFunc in pairs(synapsea.layers) do
	local buildModule    = require("core.layers.build")
	local errorModule    = require("core.layers.error")
	local gradientModule = require("core.layers.gradient")

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

package.path = oldPackagePath

if synapsea.version:match("-(.*)") then
	io.write("\27[1m\27[33mWARNING:\27[0m You are using a development release of Synapsea!\n")
	io.flush()
end

return synapsea
