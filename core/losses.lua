--[[
	https://github.com/x-xxoa/synapsea
	core/losses.lua

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

local lossesModule = {
	meanSquaredError,
	meanAbsoluteError,
	sumOfSquaredError,
	rootOfMeanSquaredError,
	crossEntropy,
	binaryCrossEntropy,
	hinge,
	huber,
	klDivergence
}

function lossesModule.meanSquaredError(output, expectedOutput)
	if type(output) == "number" then
		return (expectedOutput - output)^2
	end

	local function lossFunc(output, expectedOutput)
		local loss, count = 0, 0

		for a = 1, #output do
			if type(output[a]) == "table" then
				local returnedLoss, returnedCount = lossFunc(output[a], expectedOutput[a])
				loss, count = loss + returnedLoss, count + returnedCount
			else
				loss = loss + (expectedOutput[a] - output[a])^2
				count = count + 1
			end
		end

		return loss, count
	end

	local loss, count = lossFunc(output, expectedOutput)

	return loss / count
end

function lossesModule.meanAbsoluteError(output, expectedOutput)
	if type(output) == "number" then
		return math.abs(expectedOutput - output)
	end

	local function lossFunc(output, expectedOutput)
		local loss, count = 0, 0

		for a = 1, #output do
			if type(output[a]) == "table" then
				local returnedLoss, returnedCount = lossFunc(output[a], expectedOutput[a])
				loss, count = loss + returnedLoss, count + returnedCount
			else
				loss = loss + math.abs(expectedOutput[a] - output[a])
				count = count + 1
			end
		end

		return loss, count
	end

	local loss, count = lossFunc(output, expectedOutput)

	return loss / count
end

function lossesModule.sumOfSquaredError(output, expectedOutput)
	if type(output) == "number" then
		return (expectedOutput - output)^2
	end

	local function lossFunc(output, expectedOutput)
		local loss = 0

		for a = 1, #output do
			if type(output[a]) == "table" then
				loss = loss + lossFunc(output[a], expectedOutput[a])
			else
				loss = loss + (expectedOutput[a] - output[a])^2
			end
		end

		return loss
	end

	local loss = lossFunc(output, expectedOutput)

	return loss
end

function lossesModule.rootOfMeanSquaredError(output, expectedOutput)
	if type(output) == "number" then
		return expectedOutput - output -- sqrt(anything^2) = anything
	end

	local function lossFunc(output, expectedOutput)
		local loss, count = 0, 0

		for a = 1, #output do
			if type(output[a]) == "table" then
				local returnedLoss, returnedCount = lossFunc(output[a], expectedOutput[a])
				loss, count = loss + returnedLoss, count + returnedCount
			else
				loss = loss + (expectedOutput[a] - output[a])^2
				count = count + 1
			end
		end

		return loss, count
	end

	local loss, count = lossFunc(output, expectedOutput)

	return math.sqrt(loss / count)
end

function lossesModule.crossEntropy(output, expectedOutput)
	if type(output) == "number" then
		return -(expectedOutput - math.log(output))
	end

	local function lossFunc(output, expectedOutput)
		local loss = 0

		for a = 1, #output do
			if type(output[a]) == "table" then
				loss = loss + lossFunc(output[a], expectedOutput[a])
			else
				loss = loss + expectedOutput[a] * math.log(output[a])
			end
		end

		return loss
	end

	local loss = lossFunc(output, expectedOutput)

	return -loss
end

function lossesModule.binaryCrossEntropy(output, expectedOutput)
	if type(output) == "number" then
		return -(expectedOutput - math.log(output) + (1 - expectedOutput) * math.log(1 - output))
	end

	local function lossFunc(output, expectedOutput)
		local loss = 0

		for a = 1, #output do
			if type(output[a]) == "table" then
				loss = loss + lossFunc(output[a], expectedOutput[a])
			else
				loss = loss + expectedOutput[a] * math.log(output[a]) + (1 - expectedOutput[a]) * math.log(1- output[a])
			end
		end

		return loss
	end

	local loss = lossFunc(output, expectedOutput)

	return -loss
end

function lossesModule.hinge(output, expectedOutput)
	if type(output) == "number" then
		return math.max(0, 1 - expectedOutput * output)
	end

	local function lossFunc(output, expectedOutput)
		local loss = 0

		for a = 1, #output do
			if type(output[a]) == "table" then
				loss = loss + lossFunc(output[a], expectedOutput[a])
			else
				loss = loss + math.max(0, 1 - expectedOutput[a] * output[a])
			end
		end

		return loss
	end

	local loss = lossFunc(output, expectedOutput)

	return loss
end

function lossesModule.huber(output, expectedOutput, lossArgs)
	local delta = lossArgs.delta

	if type(output) == "number" then
		local x = output - expectedOutput

		if math.abs(x) <= delta then
			return 0.5 * x^2
		else
			return delta * math.abs(x) - 0.5 * delta^2
		end
	end

	local function lossFunc(output, expectedOutput, delta)
		local loss = 0

		for a = 1, #output do
			if type(output[a]) == "table" then
				loss = loss + lossFunc(output[a], expectedOutput[a])
			else
				local x = output[a] - expectedOutput[a]

				if math.abs(x) <= delta then
					loss = loss + 0.5 * x^2
				else
					loss = loss + delta * math.abs(x) - 0.5 * delta^2
				end
			end
		end

		return loss
	end

	local loss = lossFunc(output, expectedOutput, delta)

	return loss
end

function lossesModule.klDivergence(output, expectedOutput, lossArgs)
	local elipson = lossArgs.elipson

	if type(output) == "number" then
		return output * math.log(output + elipson / expectedOutput + elipson)
	end

	local function lossFunc(output, expectedOutput)
		local loss = 0

		for a = 1, #output do
			if type(output[a]) == "table" then
				loss = loss + lossFunc(output[a], expectedOutput[a])
			else
				loss = loss + output[a] * math.log(output[a] + elipson / expectedOutput[a] + elipson)
			end
		end

		return loss
	end

	local loss = lossFunc(output, expectedOutput, delta)

	return loss
end

return lossesModule
