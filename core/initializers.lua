--[[
	https://github.com/x-xxoa/synapsea
	core/initializers.lua

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

local mathModule = require("core.math")
local initializersModule = {
	uniformRandom,
	normalRandom,
	uniformXavier,
	normalXavier,
	uniformHe,
	normalHe,
	constant
}

function initializersModule.uniformRandom(args)
	if type(args.input) ~= "table" then
		return mathModule.random.uniform(args.lowerLimit, args.upperLimit)
	end

	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = initializersModule.uniformRandom{
				input = args.input[a],
				lowerLimit = args.lowerLimit,
				upperLimit = args.upperLimit
			}
		else
			args.input[a] = mathModule.random.uniform(args.lowerLimit, args.upperLimit)
		end
	end

	return args.input
end

function initializersModule.normalRandom(args)
	if type(args.input) ~= "table" then
		return mathModule.random.normal(args.mean, args.sd)
	end

	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = initializersModule.normalRandom{
				input = args.input[a],
				mean = args.mean,
				sd = args.sd
			}
		else
			args.input[a] = mathModule.random.normal(args.mean, args.sd)
		end
	end

	return args.input
end

function initializersModule.uniformXavier(args)
	local limit = math.sqrt(6 / (args.inputs + args.outputs))

	if type(args.input) ~= "table" then
		return mathModule.random.uniform(-limit, limit)
	end

	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = initializersModule.uniformXavier{
				input = args.input[a],
				inputs = args.inputs,
				outputs = args.outputs
			}
		else
			args.input[a] = mathModule.random.uniform(-limit, limit)
		end
	end

	return args.input
end

function initializersModule.normalXavier(args)
	local sd = math.sqrt(6 / (args.inputs + args.outputs))

	if type(args.input) ~= "table" then
		return mathModule.random.normal(0, sd)
	end

	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = initializersModule.normalXavier{
				input = args.input[a],
				inputs = args.inputs,
				outputs = args.outputs
			}
		else
			args.input[a] = mathModule.random.normal(0, sd)
		end
	end

	return args.input
end

function initializersModule.uniformHe(args)
	local limit = math.sqrt(2 / args.inputs)

	if type(args.input) ~= "table" then
		return mathModule.random.uniform(-limit, limit)
	end

	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = initializersModule.uniformHe{
				input = args.input[a],
				inputs = args.inputs
			}
		else
			args.input[a] = mathModule.random.uniform(-limit, limit)
		end
	end

	return args.input
end

function initializersModule.normalHe(args)
	local sd = math.sqrt(2 / args.inputs)

	if type(args.input) ~= "table" then
		return mathModule.random.normal(0, sd)
	end

	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = initializersModule.normalHe{
				input = args.input[a],
				inputs = args.inputs
			}
		else
			args.input[a] = mathModule.random.normal(0, sd)
		end
	end

	return args.input
end

function initializersModule.constant(args)
	if type(args.input) ~= "table" then
		return args.value
	end

	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = initializersModule.constant{
				input = args.input[a],
				value = args.value
			}
		else
			args.input[a] = args.value
		end
	end

	return args.input
end

return initializersModule
