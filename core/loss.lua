--[[
	https://github.com/x-xxoa/synapsea
	core/loss.lua

	MIT License

	| || || |_
]]--

local loss = {
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

function loss.meanSquaredError(output, expectedOutput)
	local sum = 0

	if type(output) == "number" then
		return (expectedOutput - output)^2
	end

	for a = 1, #output do
		sum = sum + (expectedOutput[a] - output[a])^2
	end

	return sum / #output
end

function loss.meanAbsoluteError(output, expectedOutput)
	local sum = 0

	if type(output) == "number" then
		return math.abs(expectedOutput - output)
	end

	for a = 1, #output do
		sum = sum + math.abs(expectedOutput[a] - output[a])
	end

	return sum / #output
end

function loss.sumOfSquaredError(output, expectedOutput)
	local sum = 0

	if type(output) == "number" then
		return (expectedOutput - output)^2
	end

	for a = 1, #output do
		sum = sum + (expectedOutput[a] - output[a])^2
	end

	return sum
end

function loss.rootOfMeanSquaredError(output, expectedOutput)
	local sum = 0

	if type(output) == "number" then
		return expectedOutput - output -- sqrt(anything^2) == anything
	end

	for a = 1, #output do
		sum = sum + (expectedOutput[a] - output[a])^2
	end

	return math.sqrt(sum / #output)
end

function loss.crossEntropy(output, expectedOutput)
	local sum = 0

	if type(output) == "number" then
		return -(expectedOutput * math.log(output))
	end

	for a = 1, #output do
		sum = sum + expectedOutput[a] * math.log(output[a])
	end

	return -sum
end

function loss.binaryCrossEntropy(output, expectedOutput)
	local sum = 0

	if type(output) == "number" then
		return -(expectedOutput * math.log(output) + (1 - expectedOutput) * math.log(1 - output))
	end

	for a = 1, #output do
		sum = sum + expectedOutput[a] * math.log(output[a]) + (1 - expectedOutput[a]) * math.log(1 - output[a])
	end

	return -sum
end

function loss.hinge(output, expectedOutput)
	local sum = 0

	if type(output) == "number" then
		return math.max(0, 1 - expectedOutput[a] * output[a])
	end

	for a = 1, #output do
		sum = sum + math.max(0, 1 - expectedOutput[a] * output[a])
	end

	return sum
end

function loss.huber(output, expectedOutput, lossArgs)
	local sum = 0

	if type(output) == "number" then
		local x = output - expectedOutput

		if math.abs(x) <= delta then
			return 0.5 * x^2
		else
			return delta * math.abs(x) - 0.5 * delta^2
		end
	end

	for a = 1, #output do
		local x = output[a] - expectedOutput[a]

		if math.abs(x) <= delta then
			sum = sum + 0.5 * x^2
		else
			sum = sum + delta * math.abs(x) - 0.5 * delta^2
		end
	end

	return sum
end

function loss.klDivergence(output, expectedOutput)
	local sum = 0

	if type(output) == "number" then
		return output * math.log(output / expectedOutput)
	end

	for a = 1, #output do
		sum = sum + output[a] * math.log(output[a] / expectedOutput[a])
	end

	return sum
end

return loss
