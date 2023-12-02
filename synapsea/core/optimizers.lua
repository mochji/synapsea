--[[
	https://github.com/mochji/synapsea
	core/optimizers.lua

	Synapsea, simple yet powerful machine learning platform for Lua.
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
	momentum
}

function optimizersModule.momentum(args)
	local function optimizerFunc(gradient, momentum, stepSize, change)
		for a, _ in pairs(gradient) do
			if canindex(gradient[a]) then
				gradient[a], lastGradient = optimizerFunc(gradient[a], momentum, stepSize)
			else
				change = stepSize * gradient[a] + momentum * change
				gradient[a] = gradient[a] - change
			end
		end

		return gradient, change
	end

	local momentum, stepSize = args.momentum, args.stepSize

	local change = 0

	for _, parameter in pairs(args.parameters) do
		if type(parameter) == "number" then
			change = args.stepSize * gradient[a] + momentum * change
			parameter = parameter - change
		else
			parameter, change = optimizerFunc(parameter, momentum, stepSize)
		end
	end

	return args.parameters
end

return optimizersModule
