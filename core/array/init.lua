--[[
	https://github.com/x-xxoa/synapsea
	core/array/init.lua

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

local arrayModule = {
	math = require("core.array.core.math"),
	bitwise = require("core.array.core.bitwise"),
	string = require("core.array.core.string"),
	initializers = require("core.array.core.initializers"),
	debug = require("core.array.core.debug"),
	full,
	zeros,
	ones,
	uniformRandom,
	normalRandom,
	asArray,
	asTable,
	range,
	shape,
	reshape,
	flatten,
	asFlat,
	transpose,
	swapAxis,
	concat,
	stack,
	split,
	rotateIn90,
	tile,
	pad,
	customPad,
	tableToString,
	exportTable,
	importTable,
	arrayToString,
	exportArray,
	importArray
}

function arrayModule.full(shape, value)
	return arrayModule.asArray(arrayModule.initializers.full(shape, value))
end

function arrayModule.zeros(shape)
	return arrayModule.asArray(arrayModule.initializers.zeros(shape))
end

function arrayModule.ones(shape)
	return arrayModule.asArray(arrayModule.initializers.ones(shape))
end

function arrayModule.uniformRandom(shape, lowerLimit, upperLimit)
	return arrayModule.asArray(arrayModule.initializers.uniformRandom(shape, lowerLimit, upperLimit))
end

function arrayModule.normalRandom(shape, mean, sd)
	return arrayModule.asArray(arrayModule.initializers.normalRandom(shape, mean, sd))
end

-- array utils functions

for name, func in pairs(require("core.array.core.arrayUtils")) do
	arrayModule[name] = func
end

--[[ array transformation functions

for name, func in pairs(require("core.array.core.arrayTransformation")) do
	arrayModule[name] = func
end

-- data functions ]]

for name, func in pairs(require("core.array.core.data")) do
	arrayModule[name] = func
end

return arrayModule
