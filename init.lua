--[[
	Synapsea v1.3.00-unstable

	A Lua Neural Network library made in pure Lua.

	Please read the README.md file for documentation and information, 

	https://github.com/x-xxoa/synapsea

	MIT License
]]--

--import the core files

local syn = {
	version = "v1.3.00-unstable",
	activation = require("core.activation"),
	model = require("core.model")
}

--other functions

return syn
