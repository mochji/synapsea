--[[
	https://github.com/mochji/synapsea
	core/optimizers.lua

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

local canindex = require("core.utils.canindex")

local optimizersModule = {
	stochastic,
	batch
}

local function applyMomentum(parameters, momentum, learningRate)
	local function optimizerFunc(gradient, momentum, learningRate, change)
		for a, _ in pairs(gradient) do
			if canindex(gradient[a]) then
				gradient[a], lastGradient = optimizerFunc(gradient[a], momentum, learningRate)
			else
				change = learningRate * gradient[a] + momentum * change
				gradient[a] = gradient[a] - change
			end
		end

		return gradient, change
	end

	local momentum, learningRate = args.momentum, args.learningRate

	local change = 0

	for _, parameter in pairs(parameters) do
		if type(parameter) == "number" then
			change = args.learningRate * gradient[a] + momentum * change
			parameter = parameter - change
		else
			parameter, change = optimizerFunc(parameter, momentum, learningRate)
		end
	end

	return parameters
end

function optimizersModule.stochastic(parameters, args)
end

function optimizersModule.batch(parameters, args)
end

return optimizersModule
