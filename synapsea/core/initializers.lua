--[[
	https://github.com/mochji/synapsea
	core/initializers.lua

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

local initializersModule = {
	uniformRandom,
	normalRandom,
	uniformXavier,
	normalXavier,
	uniformHe,
	normalHe,
	constant
}

function initializersModule.uniformRandom(shape, args)
	local function initializerFunc(shape, lowerLimit, upperLimit, index)
		local output = {}

		if index == #shape then
			for a = 1, shape[index] do
				output[a] = lowerLimit + math.random() * (upperLimit - lowerLimit)
			end
		else
			for a = 1, shape[index] do
				output[a] = initializerFunc(
					shape,
					lowerLimit,
					upperLimit,
					index + 1
				)
			end
		end

		return output
	end

	return initializerFunc(shape, args.lowerLimit, args.upperLimit, 1)
end

function initializersModule.normalRandom(shape, args)
	local function initializerFunc(shape, mean, sd, index)
		local output = {}

		if index == #shape then
			for a = 1, shape[index] do
				output[a] = mean + math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) * sd
			end
		else
			for a = 1, shape[index] do
				output[a] = initializerFunc(
					shape,
					mean,
					sd,
					index + 1
				)
			end
		end

		return output
	end

	return initializerFunc(shape, args.mean, args.sd, 1)
end

function initializersModule.uniformXavier(shape, args)
	local function initializerFunc(shape, limit, index)
		local output = {}

		if index == #shape then
			for a = 1, shape[index] do
				output[a] = -limit + math.random() * (limit + limit)
			end
		else
			for a = 1, shape[index] do
				output[a] = initializerFunc(
					shape,
					limit,
					index + 1
				)
			end
		end

		return output
	end

	return initializerFunc(shape, math.sqrt(6 / (args.inputs + args.outputs)), 1)
end

function initializersModule.normalXavier(shape, args)
	local function initializerFunc(shape, sd, index)
		local output = {}

		if index == #shape then
			for a = 1, shape[index] do
				output[a] = math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) * sd
			end
		else
			for a = 1, shape[index] do
				output[a] = initializerFunc(
					shape,
					sd,
					index + 1
				)
			end
		end

		return output
	end

	return initializerFunc(shape, math.sqrt(6 / (args.inputs + args.outputs)), 1)
end

function initializersModule.uniformHe(shape, args)
	local function initializerFunc(shape, limit, index)
		local output = {}

		if index == #shape then
			for a = 1, shape[index] do
				output[a] = -limit + math.random() * (limit + limit)
			end
		else
			for a = 1, shape[index] do
				output[a] = initializersModule.uniformHe(
					shape,
					limit,
					index + 1
				)
			end
		end

		return output
	end

	return initializerFunc(shape, math.sqrt(2 / args.inputs), 1)
end

function initializersModule.normalHe(shape, args)
	local function initializerFunc(shape, sd, index)
		local output = {}

		if index == #shape then
			for a = 1, shape[index] do
				output[a] = math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) * sd
			end
		else
			for a = 1, shape[index] do
				output[a] = initializerFunc(
					shape,
					sd,
					index + 1
				)
			end
		end

		return output
	end

	return initializerFunc(shape, math.sqrt(2 / args.inputs), 1)
end

function initializersModule.constant(shape, args, index)
	local function initializerFunc(shape, value, index)
		local output = {}

		if index == #shape then
			for a = 1, shape[index] do
				output[a] = value
			end
		else
			for a = 1, shape[index] do
				output[a] = initializerFunc(
					shape,
					value,
					index + 1
				)
			end
		end

		return output
	end

	return initializerFunc(shape, args.value, 1)
end

return initializersModule
