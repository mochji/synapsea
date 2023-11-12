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

-- Get the path Synapsea was required from
_SYNAPSEA_PATH = debug.getinfo(1).short_src:match("(.*[/\\])") or ""

-- Temporary function to make requiring modules cleaner
local function getModule(module)
	return require(_SYNAPSEA_PATH .. "core." .. module)
end

-- Wrap the core modules into a table
local synapsea = {
	version =      "v2.0.00-unstable",

	activations =  getModule("activations"),
	losses =       getModule("losses"),
	math =         getModule("math"),
	initializers = getModule("initializers"),
	optimizers =   getModule("optimizers"),
	regularizers = getModule("regularizers"),
	layers =       getModule("layers.layers"),
	backProp =     getModule("backProp"),
	model =        getModule("model")
}


-- Convert layers to metatables for cleaner code

do
	local buildModule = require(_SYNAPSEA_PATH .. "core.layers.build")
	local errorModule = require(_SYNAPSEA_PATH .. "core.layers.error")
	local gradientModule = require(_SYNAPSEA_PATH .. "core.layers.gradient")

	for name, func in pairs(synapsea.layers) do
		synapsea.layers[name] = setmetatable(
			{
				build = buildModule[name],
				error = errorModule[name],
				gradient = gradientModule[name]
			},
			{__call = func}
		)
	end
end

return synapsea
