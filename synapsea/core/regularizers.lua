--[[
	https://github.com/mochji/synapsea
	core/regularizers.lua

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

local regularizersModule = {
	l1,
	l2
}

function regularizersModule.l1(args)
	local function absoluteSum(tbl)
		local sum = 0

		for a = 1, #tbl do
			if canindex(tbl[a]) then
				sum = sum + absoluteSum(tbl[a])
			else
				sum = sum + math.abs(tbl[a])
			end
		end

		return sum
	end

	local function regularizerFunc(gradient, lambda, l1Norm)
		l1Norm = l1Norm or absoluteSum(gradient)

		for a = 1, #gradient do
			if canindex(args.gradient[a]) then
				gradient[a] = regularizerFunc(gradient[a], lambda, l1Norm)
			else
				gradient[a] = gradient[a] + lambda * l1Norm
			end
		end
	end

	local lambda = args.lambda

	for _, parameter in pairs(args.trainableParameters) do
		if type(parameter) == "number" then
			parameter = parameter + lambda * math.abs(parameter)
		else
			parmaeter = regularizerFunc(parameter, lambda)
		end
	end

	return args.trainableParameters
end

function regularizersModule.l2(args)
	local function squaredSum(tbl)
		local sum = 0

		for a = 1, #tbl do
			if canindex(tbl[a]) then
				sum = sum + squaredSum(tbl[a])
			else
				sum = sum + tbl[a]^2
			end
		end

		return sum
	end

	local function regularizerFunc(gradient, lambda, l2Norm)
		l2Norm = l2Norm or squaredSum(gradient)

		for a = 1, #gradient do
			if canindex(args.gradient[a]) then
				gradient[a] = regularizerFunc(gradient[a], lambda, l2Norm)
			else
				gradient[a] = gradient[a] + lambda * l2Norm
			end
		end
	end

	local lambda = args.lambda

	for _, parameter in pairs(args.trainableParameters) do
		if type(parameter) == "number" then
			parameter = parameter + lambda * parameter^2
		else
			parmaeter = regularizerFunc(parameter)
		end
	end

	return args.trainableParameters
end

return regularizersModule
