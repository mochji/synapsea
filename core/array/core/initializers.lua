--[[
	https://github.com/x-xxoa/synapsea
	core/array/core/initializers.lua

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

local initializersModule = {
	full,
	zeros,
	ones,
	uniformRandom,
	normalRandom
}

function initializersModule.full(shape, value, index)
	local output = {}
	index = index or 1

	if index == #shape then
		for a = 1, shape[index] do
			output[a] = value
		end

		return output
	end

	for a = 1, shape[index] do
		output[a] = initializersModule.full(shape, value, index + 1)
	end

	return output
end

function initializersModule.zeros(shape)
	return initializersModule.full(shape, 0)
end

function initializersModule.ones(shape)
	return initializersModule.full(shape, 1)
end

function initializersModule.uniformRandom(shape, lowerLimit, upperLimit, index)
	local output = {}
	index = index or 1

	if index == #shape then
		for a = 1, shape[index] do
			output[a] = lowerLimit + math.random() * (upperLimit - lowerLimit)
		end

		return output
	end

	for a = 1, shape[index] do
		output[a] = initializersModule.uniformRandom(shape, lowerLimit, upperLimit, index + 1)
	end

	return output
end

function initializersModule.normalRandom(shape, mean, sd, index)
	local output = {}
	index = index or 1

	if index == #shape then
		for a = 1, shape[index] do
			output[a] = mean + math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) * sd
		end

		return output
	end

	for a = 1, shape[index] do
		output[a] = initializersModule.normalRandom(shape, mean, sd, index + 1)
	end

	return output
end

return initializersModule
