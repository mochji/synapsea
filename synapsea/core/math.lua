--[[
	https://github.com/mochji/synapsea
	core/math.lua

	Synapsea, a simple yet powerful machine learning framework for Lua.
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

local mathModule = {
	random = {
		uniform,
		normal
	},
	round,
	sign,
	root
}

function mathModule.random.uniform(lowerLimit, upperLimit)
	return lowerLimit + math.random() * (upperLimit - lowerLimit)
end

function mathModule.random.normal(mean, sd)
	return mean + math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) * sd -- https://forum.cheatengine.org/viewtopic.php?p=5724230 omg cheatengine
end

function mathModule.round(x)
	return math.floor(x + 0.5)
end

function mathModule.sign(x)
	if x > 0 then
		return 1
	end

	if x < 0 then
		return -1
	end

	return 0
end

function mathModule.root(x, root)
	return x^(1 / root)
end

return mathModule
