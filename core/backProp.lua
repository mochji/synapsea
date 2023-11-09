--[[
	https://github.com/x-xxoa/synapsea
	core/backProp.lua

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

--[[
	notes >w<!

	ewwow backpwopagation
	error = (output - expected) * activation'(output)  | output
	error = (weight_k * error_j) * activation'(output) | hidden

	weight update
	weight = weight - lr * error * input

	bias update
	figuwe out latew -w-
]]--

local backPropModule = {
	layersError = require("core.layersError"),
	layersGradient = require("core.layersGradient"),
	outputError,
	stochasticGradientDescent,
	batchGradientDescent
}

function backPropModule.outputError(output, expectedOutput, activation, alpha)
	local outputError = {}

	local activation = activationsModule[activation]

	for a = 1, #output do
		if type(output[a]) == "table" then
			outputError[a] = backPropModule.outputError(output[a], expectedOutput[a], activation)
		else
			outputError[a] = (output[a] - expectedOutput[a]) * activation(output[a], true, alpha)
		end
	end

	return outputError
end

function backPropModule.stochasticGradientDescent(gradient, weights, biases)
end

function backPropModule.batchGradientDescent(gradient, weights, biases, gradientDescentArgs)
end

return backPropModule
