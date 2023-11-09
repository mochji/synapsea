--[[
	https://github.com/x-xxoa/synapsea
	core/optimizers.lua

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

local optimizersModule = {
	momentum
}

function optimizersModule.momentum(args)
	local change = 0
	local function optimizerFunc(gradient, momentum, stepSize, change)
		for a, _ in pairs(gradient) do
			if type(gradient[a]) == "table" then
				gradient[a], lastGradient = optimizerFunc(gradient[a], momentum, stepSize)
			else
				change = stepSize * gradient[a] + momentum * change
				gradient[a] = gradient[a] - change
			end
		end

		return gradient, change
	end

	for _, parameter in pairs(args.trainableParameters) do
		if type(parameter) == "number" then
			change = args.stepSize * gradient[a] + momentum * change
			parameter = parameter - change
		else
			parameter, change = optimizerFunc(parameter, args.momentum, args.stepSize)
		end
	end

	return args.trainableParameters
end

return optimizersModule
