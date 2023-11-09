--[[
	https://github.com/x-xxoa/synapsea
	core/regularizers.lua

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

local regularizersModule = {
	l1,
	l2
}

local function absoluteSum(tbl)
	local sum = 0

	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			sum = sum + absoluteSum(tbl[a])
		else
			sum = sum + math.abs(tbl[a])
		end
	end

	return sum
end

local function squaredSum(tbl)
	local sum = 0

	for a = 1, #tbl do
		if type(tbl[a]) == "table" then
			sum = sum + squaredSum(tbl[a])
		else
			sum = sum + tbl[a]^2
		end
	end

	return sum
end

function regularizersModule.l1(args)
	local function regularizerFunc(gradient, lambda, l1Norm)
		l1Norm = l1Norm or absoluteSum(gradient)

		for a = 1, #gradient do
			if type(args.gradient[a]) == "table" then
				gradient[a] = regularizerFunc(gradient[a], lambda, l1Norm)
			else
				gradient[a] = gradient[a] + lambda * l1Norm
			end
		end
	end

	for _, parameter in pairs(args.trainableParameters) do
		if type(parameter) == "number" then
			parameter = parameter + args.lambda * math.abs(parameter)
		else
			parmaeter = regularizerFunc(parameter, args.lambda)
		end
	end

	return args.trainableParameters
end

function regularizersModule.l2(args)
	local function regularizerFunc(gradient, lambda, l2Norm)
		l2Norm = l2Norm or squaredSum(gradient)

		for a = 1, #gradient do
			if type(args.gradient[a]) == "table" then
				gradient[a] = regularizerFunc(gradient[a], lambda, l2Norm)
			else
				gradient[a] = gradient[a] + lambda * l2Norm
			end
		end
	end

	for _, parameter in pairs(args.trainableParameters) do
		if type(parameter) == "number" then
			parameter = parameter + args.lambda * parameter^2
		else
			parmaeter = regularizerFunc(parameter, args.lambda)
		end
	end

	return args.trainableParameters
end

return regularizersModule
