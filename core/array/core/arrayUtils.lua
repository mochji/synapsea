--[[
	https://github.com/x-xxoa/synapsea
	core/array/core/arrayUtils.lua

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

local bitwiseModule = require("core.array.core.bitwise")
local mathModule = require("core.array.core.math")
local arrayUtilsModule = {
	asArray,
	range,
	shape,
	reshape,
	flatten,
	asFlat
}

function arrayUtilsModule.asArray(tbl)
	return setmetatable(
		tbl,
		{
			__add = mathModule.add,
			__sub = mathModule.sub,
			__mul = mathModule.multiply,
			__div = mathModule.divide,
			__mod = mathModule.modulo,
			__unm = mathModule.negate,
			__pow = mathModule.power,
			__idiv = mathModule.floorDivide,
			__lt = mathModule.lessThan,
			__le = mathModule.lessThanOrEqualTo,
			__band = bitwiseModule.bAnd,
			__bor = bitwiseModule.bOr,
			__bxor = bitwiseModule.bXor,
			__bnot = bitwiseModule.bNot,
			__shl = bitwiseModule.leftShift,
			__shr = bitwiseModule.rightShift
		}
	)
end

function arrayUtilsModule.range(start, stop, step)
	start, step = start or 1, step or 1
	local output = {}

	local current, nextNumber = start, start

	while nextNumber <= stop do
		output[#output + 1] = nextNumber

		nextNumber = nextNumber + step
	end
end

function arrayUtilsModule.aRange(start, stop, step)
	return arrayUtilsModule.asArray(arrayUtilsModule.range(start, stop, step))
end

function arrayUtilsModule.shape(tbl)
end

function arrayUtilsModule.reshape(tbl, shape)
end

function arrayUtilsModule.flatten(tbl)
end

arrayUtilsModule.asFlat = next

return arrayUtilsModule
