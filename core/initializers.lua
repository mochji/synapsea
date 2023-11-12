--[[
	https://github.com/mochji/synapsea
	core/initializers.lua

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

local mathModule = require(_SYNAPSEA_PATH .. "core.math")

local initializersModule = {
	zeros,
	uniformRandom,
	normalRandom,
	uniformXavier,
	normalXavier,
	uniformHe,
	normalHe,
	constant
}

function initializersModule.zeros(shape, args, index)
	index = index or 1

	local output = {}

	if index == #args.shape then
		for a = 1, args.shape[index] do
			output[a] = 0
		end
	else
		for a = 1, args.shape[index] do
			output[a] = initializersModule.zeros(
				{
					shape = args.shape
				},
				index + 1
			)
		end
	end

	return output
end

function initializersModule.uniformRandom(shape, args, index)
	index = index or 1
	local lowerLimit, upperLimit = args.lowerLimit, args.upperLimit

	local output = {}

	if index == #args.shape then
		for a = 1, args.shape[index] do
			output[a] = mathModule.random.uniform(lowerLimit, upperLimit)
		end
	else
		for a = 1, args.shape[index] do
			output[a] = initializersModule.uniformRandom(
				{
					shape = args.shape,
					lowerLimit = lowerLimit,
					upperLimit = upperLimit
				},
				index + 1
			)
		end
	end

	return output
end

function initializersModule.normalRandom(shape, args, index)
	index = index or 1
	local mean, sd = args.mean, args.sd

	local output = {}

	if index == #shape then
		for a = 1, shape[index] do
			output[a] = mathModule.random.normal(mean, sd)
		end
	else
		for a = 1, shape[index] do
			output[a] = initializersModule.normalRandom(
				{
					shape = shape,
					mean = mean,
					sd = sd
				},
				index + 1
			)
		end
	end

	return output
end

function initializersModule.uniformXavier(shape, args, index)
	index = index or 1
	local inputs, outputs = args.inputs, args.outputs

	local output = {}

	local limit = math.sqrt(6 / (inputs + outputs))

	if index == #shape then
		for a = 1, shape[index] do
			output[a] = mathModule.random.uniform(-limit, limit)
		end
	else
		for a = 1, shape[index] do
			output[a] = initializersModule.uniformXavier(
				{
					shape = shape,
					inputs = inputs,
					outputs = outputs
				},
				index + 1
			)
		end
	end

	return output
end

function initializersModule.normalXavier(shape, args, index)
	index = index or 1
	local inputs, outputs = args.inputs, args.outputs

	local output = {}

	local sd = math.sqrt(6 / (inputs + outputs))

	if index == #shape then
		for a = 1, shape[index] do
			output[a] = mathModule.random.normal(0, sd)
		end
	else
		for a = 1, shape[index] do
			output[a] = initializersModule.normalXavier(
				{
					shape = shape,
					inputs = inputs,
					outputs = outputs
				},
				index + 1
			)
		end
	end

	return output
end

function initializersModule.uniformHe(shape, args, index)
	index = index or 1
	local inputs = args.inputs

	local output = {}

	local limit = math.sqrt(2 / inputs)

	if index == #shape then
		for a = 1, shape[index] do
			output[a] = mathModule.random.uniform(-limit, limit)
		end
	else
		for a = 1, shape[index] do
			output[a] = initializersModule.uniformHe(
				{
					shape = shape,
					inputs = inputs
				},
				index + 1
			)
		end
	end

	return output
end

function initializersModule.normalHe(shape, args)
	index = index or 1
	local inputs = args.inputs

	local output = {}

	local sd = math.sqrt(2 / inputs)

	if index == #shape then
		for a = 1, shape[index] do
			output[a] = mathModule.random.normal(0, sd)
		end
	else
		for a = 1, shape[index] do
			output[a] = initializersModule.normalHe(
				{
					shape = shape,
					inputs = inputs
				},
				index + 1
			)
		end
	end

	return output
end

function initializersModule.constant(shape, args, index)
	index = index or 1
	local value = args.value

	local output = {}

	if index == #shape then
		for a = 1, shape[index] do
			output[a] = value
		end
	else
		for a = 1, shape[index] do
			output[a] = initializersModule.constant(
				{
					shape = shape,
					value = value
				},
				index + 1
			)
		end
	end

	return output
end

return initializersModule
