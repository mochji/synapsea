--[[
	https://github.com/x-xxoa/synapsea
	core/activations.lua

	Synapsea, a simple yet powerful machine learning library made in pure Lua.
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

local activationsModule = {
	sigmoid,
	tanh,
	relu,
	leakyRelu,
	elu,
	exponential,
	swish,
	binaryStep,
	softMax,
	softPlus,
	softSign,
	linear,
	variableLinear,
	hardSigmoid,
	hardTanh
}

function activationsModule.sigmoid(x, derivative)
	if derivative then
		return (1 / (1 + math.exp(-x))) * (1 - (1 / (1 + math.exp(-x))))
	end

	return 1 / (1 + math.exp(-x))
end

function activationsModule.tanh(x, derivative)
	if derivative then
		return 1 - (((math.exp(2 * x) - 1) / (math.exp(2 * x) + 1))^2)
	end

	return math.tanh(x)
end

function activationsModule.relu(x, derivative)
	if derivative then
		if x > 0 then
			return 1
		end

		return 0
	end

	return math.max(0, x)
end

function activationsModule.leakyRelu(x, derivative, alpha)
	if derivative then
		if x > 0 then
			return 1
		end

		return alpha
	end

	if x > 0 then
		return x
	end

	return x * alpha
end

function activationsModule.elu(x, derivative, alpha)
	if derivative then
		if x < 0 then
			return alpha * math.exp(x)
		end

		return 1
	end

	if x < 0 then
		return alpha * (math.exp(x) - 1)
	end

	return x
end

function activationsModule.exponential(x)
	return math.exp(x)
end

function activationsModule.swish(x, derivative, alpha)
	if derivative then
		return (math.exp(-alpha * x) * x + math.exp(-alpha * x) + 1) / ((math.exp(-alpha * x) + 1)^2)
	end

	return x / (1 + math.exp(-alpha * x))
end

function activationsModule.binaryStep(x, derivative)
	if derivative then
		return 0
	end

	if x > 0 then
		return 1
	end

	return 0
end

function activationsModule.softMax(x, derivative)
	local expSum, output = 0, {}

	for a = 1, #x do
		expSum = expSum + math.exp(x[a])
	end

	if derivative then
		for a = 1, #x do
			output[a] = (math.exp(x[a]) / expSum) * (1 - (math.exp(x[a]) / expSum))
		end

		return output
	end

	for a = 1, #x do
		output[a] = math.exp(x[a]) / expSum
	end

	return output
end

function activationsModule.softPlus(x, derivative)
	if derivative then
		return 1 / (1 + math.exp(-x))
	end

	return math.log(1 + math.exp(x))
end

function activationsModule.softSign(x, derivative)
	if derivative then
		if x == 0 then
			return 1 -- At x = 0 this is undefined
		end

		return x / (x * (1 + math.abs(x)^2))
	end

	return x / (1 + math.abs(x))
end

function activationsModule.linear(x, derivative)
	if derivative then
		return 1
	end

	return x
end

function activationsModule.variableLinear(x, derivative, alpha)
	if derivative then
		return alpha
	end

	return x * alpha
end

function activationsModule.hardSigmoid(x, derivative)
	if derivative then
		if x < -2.5 or x > 2.5 then
			return 0
		end

		return 0.2
	end

	return math.max(0, math.min(1, x * 0.2 + 0.5))
end

function activationsModule.hardTanh(x, derivative)
	if derivative then
		if x < -1 or x > 1 then
			return 0
		end

		return 1
	end

	return math.max(0, math.min(1, x * 2))
end

return activationsModule
