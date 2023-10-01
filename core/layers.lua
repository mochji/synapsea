--[[
	https://github.com/x-xxoa/synapsea
	core/layers.lua

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

local arrayRequire = dofile("synArrayRequireInfo.lua")
package.path = package.path .. ";" .. arrayRequire.addRequirePath

local arrayModule = require(arrayRequire.requireString)
local activationsModule = require("core.activations")
local mathModule = require("core.math")
local layersModule = {
	dense,
	averagePooling1D,
	averagePooling2D,
	averagePooling3D,
	maxPooling1D,
	maxPooling2D,
	maxPooling3D,
	sumPooling1D,
	sumPooling2D,
	sumPooling3D,
	averageGlobalPooling1D,
	averageGlobalPooling2D,
	averageGlobalPooling3D,
	maxGlobalPooling1D,
	maxGlobalPooling2D,
	maxGlobalPooling3D,
	sumGlobalPooling1D,
	sumGlobalPooling2D,
	sumGlobalPooling3D,
	upSample1D,
	upSample2D,
	upSample3D,
	zeroPad1D,
	zeroPad2D,
	zeroPad3D,
	crop1D,
	crop2D,
	crop3D,
	convolutional1D,
	convolutional2D,
	convolutional3D,
	convolutionalTranspose1D,
	convolutionalTranspose2D,
	convolutionalTranspose3D,
	convolutionalDepthwise1D,
	convolutionalDepthwise2D,
	convolutionalSeparable1D,
	convolutionalSeparable2D,
	convolutionalSeparable3D,
	convolutionalDepthwiseSeparable1D,
	convolutionalDepthwiseSeparable2D,
	locallyConnected1D,
	locallyConnected2D,
	locallyConnected3D,
	flatten,
	reshape,
	minMaxNormalize1D,
	minMaxNormalize2D,
	minMaxNormalize3D,
	vectorAdd1D,
	vectorAdd2D,
	vectorAdd3D,
	vectorSubtract1D,
	vectorSubtract2D,
	vectorSubtract3D,
	dot1D,
	dot2D,
	dot3D,
	vectorDivide1D,
	vectorDivide2D,
	vectorDivide3D,
	add1D,
	add2D,
	add3D,
	subtract1D,
	subtract2D,
	subtract3D,
	multiply1D,
	multiply2D,
	multiply3D,
	divide1D,
	divide2D,
	divide3D,
	dropOut,
	uniformNoise,
	normalNoise,
	softmax,
	activate
}

function layersModule.dense(args)
	local output = {}

	local activation = activationsModule[args.activation]
	local input, weights, bias, alpha = args.input, args.weights, args.bias, args.alpha
	bias = bias or 0

	for a = 1, args.outputSize do
		local sum = 0

		for b = 1, #args.input do
			sum = sum + input[b] * weights[b][a]
		end

		output[a] = activation(sum + bias, false, alpha)
	end

	return output
end

function layersModule.averagePooling1D(args)
	local output, startIndex = {}, 0 -- even though its named startIndex its actually startIndex - 1 but startIndexMinusOne isnt a good variable name and idrk what else to call it :/
	local outputSize = math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1

	local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

	for a = 1, outputSize do
		local sum = 0

		for b = 1, #kernel[1] do
			if b % dilation[1] == 0 then
				sum = sum + input[startIndex + b]
			end
		end

		startIndex = startIndex + stride[1]

		output[a] = sum / kernel[1]
	end

	return output
end

function layersModule.averagePooling2D(args)
	local output, startIndexA, kernelProduct = {}, 0, syntable.product(args.kernel) -- were getting the product of the kernel so we dont perform the same calculation every output node
	local outputHeight, outputWidth = math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1, math.floor((#args.input[1] - args.kernel[2]) / args.stride[2]) + 1

	local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

	for a = 1, outputHeight do
		output[a] = {}
		local startIndexB = 0

		for b = 1, outputWidth do
			local sum = 0

			for c = 1, kernel[1] do
				for d = 1, kernel[2] do
					if c % dilation[1] == 0 and d & dilation[2] == 1 then
						sum = sum + input[startIndexA + c][startIndexB + d]
					end
				end
			end

			startIndexB = startIndexB + stride[2]

			output[a][b] = sum / kernelProduct
		end

		startIndexA = startIndexA + stride[1]
	end

	return output
end

function layersModule.averagePooling3D(args)
	local output, startIndexA, kernelProduct = {}, 0, syntable.product(args.kernel)
	local outputDepth, outputHeight, outputWidth = math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1, math.floor((#args.input[1] - args.kernel[2]) / args.stride[2]) + 1, math.floor((#args.input[1][1] - args.kernel[3]) / args.stride[3]) + 1

	local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

	for a = 1, outputDepth do
		output[a] = {}
		local startIndexB = 0

		for b = 1, outputHeight do
			output[a][b] = {}
			local startIndexC = 0

			for c = 1, outputWidth do
				local sum = 0

				for d = 1, kernel[1] do
					for e = 1, kernel[2] do
						for f = 1, kernel[3] do
							if d % dilation[1] == 0 and e % dilation[2] == 0 and f % dilation[3] == 0 then
								sum = sum + input[startIndexA + d][startIndexB + e][startIndexC + f]
							end
						end
					end
				end

				startIndexC = startIndexC + stride[3]

				output[a][b][c] = sum / kernelProduct
			end

			startIndexB = startIndexB + stride[2]
		end

		startIndexA = startIndexA + stride[1]
	end

	return output
end

function layersModule.maxPooling1D(args)
	local output, startIndex = {}, 0
	local outputSize = math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1

	local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

	for a = 1, outputSize do
		local max = input[startIndex + 1]

		for b = 1, #kernel[1] do -- yes i know we're comparing the first one against itself in the first iteration of this loop but its not really that much and 1 per output node is probably fine
			if b % dilation[1] == 0 then
				max = math.max(max, input[startIndex + b])
			end
		end

		start = start + stride[1]

		output[a] = max
	end

	return output
end

function layersModule.maxPooling2D(args)
	local output, startIndexA = {}, 0
	local outputHeight, outputWidth = math.floor((#args.filter - args.kernel[1]) / args.stride[1]) + 1, math.floor((#args.filter[1] - args.kernel[2]) / args.stride[2]) + 1

	local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

	for a = 1, outputHeight do
		output[a] = {}
		local startIndexB = 0

		for b = 1, outputWidth do
			local max = input[startIndexA + 1][startIndexB + 1]

			for c = 1, kernel[1] do
				for d = 1, kernel[2] do
					if c % dilation[1] == 0 and d % dilation[2] == 0 then
						max = math.max(max, input[startIndexA + c][startIndexB + d])
					end
				end
			end

			startIndexB = startIndexB + stride[2]

			output[a][b] = max
		end

		startIndexA = startIndexA + stride[1]
	end

	return output
end

function layersModule.maxPooling3D(args)
	local output, startIndexA = {}, 0
	local outputDepth, outputHeight, outputWidth = math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1, math.floor((#args.input[1] - args.kernel[2]) / args.stride[2]) + 1, math.floor((#args.input[1][1] - args.kernel[3]) / args.stride[3]) + 1

	local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

	for a = 1, outputDepth do
		output[a] = {}
		local startIndexB = 0

		for b = 1, outputHeight do
			output[a][b] = {}
			local startIndexC = 0

			for c = 1, outputWidth do
				local max = input[startIndexA + 1][startIndexB + 1][startIndexC + 1]

				for d = 1, kernel[1] do
					for e = 1, kernel[2] do
						for f = 1, kernel[3] do
							if d % dilation[1] == 0 and e % dilation[2] == 0 and f % dilation[3] == 0 then
								max = math.max(max, input[startIndexA + d][startIndexB + e][startIndexC + f])
							end
						end
					end
				end

				startIndexC = startIndexC + stride[3]

				output[a][b][c] = max
			end

			startIndexB = startIndexB + stride[2]
		end

		startIndexA = startIndexA + stride[1]
	end

	return output
end

function layersModule.sumPooling1D(args)
	local output, startIndex = {}, 0
	local outputSize = math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1

	local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

	for a = 1, outputSize do
		local sum = 0

		for b = 1, #kernel[1] do
			if b % dilation[1] == 0 then
				sum = sum + input[startIndex + b]
			end
		end

		start = start + stride[1]

		output[a] = sum
	end

	return output
end

function layersModule.sumPooling2D(args)
	local output, startIndexA = {}, 0
	local outputHeight, outputWidth = math.floor((#args.filter - args.kernel[1]) / args.stride[1]) + 1, math.floor((#args.filter[1] - args.kernel[2]) / args.stride[2]) + 1

	local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

	for a = 1, outputHeight do
		output[a] = {}
		local startIndexB = 0

		for b = 1, outputWidth do
			local sum = 0

			for c = 1, kernel[1] do
				for d = 1, kernel[2] do
					if c % dilation[1] == 0 and d & dilation[2] == 1 then
						sum = sum + input[startIndexA + c][startIndexB + d]
					end
				end
			end

			startIndexB = startIndexB + stride[2]

			output[a][b] = sum
		end

		startIndexA = startIndexA + stride[1]
	end

	return output
end

function layersModule.sumPooling3D(args)
	local output, startIndexA = {}, 0
	local outputDepth, outputHeight, outputWidth = math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1, math.floor((#args.input[1] - args.kernel[2]) / args.stride[2]) + 1, math.floor((#args.input[1][1] - args.kernel[3]) / args.stride[3]) + 1

	local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

	for a = 1, outputDepth do
		output[a] = {}
		local startIndexB = 0

		for b = 1, outputHeight do
			output[a][b] = {}
			local startIndexC = 0

			for c = 1, outputWidth do
				local sum = 0

				for d = 1, kernel[1] do
					for e = 1, kernel[2] do
						for f = 1, kernel[3] do
							if d % dilation[1] == 0 and e % dilation[2] == 0 and f % dilation[3] == 0 then
								sum = sum + input[startIndexA + d][startIndexB + e][startIndexC + f]
							end
						end
					end
				end

				startIndexC = startIndexC + stride[3]

				output[a][b][c] = sum
			end

			startIndexB = startIndexB + stride[2]
		end

		startIndexA = startIndexA + stride[1]
	end

	return output
end

function layersModule.averageGlobalPooling1D(args)
	return {arrayModule.math.sum(args.input) / #args.input}
end

function layersModule.averageGlobalPooling2D(args)
	local output = {}
	local input = args.input

	for a = 1, #input do
		output[a] = arrayModule.math.sum(input[a]) / #input
	end

	return output
end

layersModule.averageGlobalPooling3D = layersModule.averageGlobalPooling2D -- syntable.sum() is recursive so doing this works

function layersModule.maxGlobalPooling1D(args)
	return {math.max(table.unpack(args.input))}
end

function layersModule.maxGlobalPooling2D(args)
	local output = {}
	local input = args.input

	for a = 1, #input do
		output[a] = arrayModule.math.max(input[a])
	end

	return output
end

layersModule.maxGlobalPooling3D = layersModule.maxGlobalPooling2D

function layersModule.sumGlobalPooling1D(args)
	return {arrayModule.math.sum(args.input)}
end

function layersModule.sumGlobalPooling2D(args)
	local output = {}
	local input = args.input

	for a = 1, #input do
		output[a] = arrayModule.math.product.sum(input[a])
	end

	return output
end

layersModule.sumGlobalPooling3D = layersModule.sumGlobalPooling2D

function layersModule.upSample1D(args)
	local output = {}
	local input, kernel = args.input, args.kernel

	for a = 1, #input * kernel[1] do
		output[a] = input[math.ceil(a / kernel[1])]
	end

	return output
end

function layersModule.upSample2D(args)
	local output = {}
	local input, kernel = args.input, args.kernel

	for a = 1, #input * kernel[1] do
		output[a] = {}

		for b = 1, #input[1] * kernel[2] do
			output[a][b] = input[math.ceil(a / kernel[1])][math.ceil(b / kernel[2])]
		end
	end

	return output
end

function layersModule.upSample3D(args)
	local output = {}
	local input, kernel = args.input, args.kernel

	for a = 1, #input * kernel[1] do
		output[a] = {}

		for b = 1, #input * kernel[2] do
			output[a][b] = {}

			for c = 1, #input * kernel[3] do
				output[a][b][c] = input[math.ceil(a / kernel[1])][math.ceil(b / kernel[2])][math.ceil(c / kernel[3])]
			end
		end
	end

	return output
end

function layersModule.zeroPad1D(args)
	local input, paddingAmount = args.input, args.paddingAmount

	for a = 1, paddingAmount[1] do
		table.insert(input, 1, 0)

		input[#input + 1] = 0
	end

	return input
end

function layersModule.zeroPad2D(args)
	local input, paddingAmount = args.input, args.paddingAmount

	-- pad the tops and bottoms

	for a = 1, paddingAmount[1] do
		table.insert(input, 1, {})

		for b = 1, #input[a + 1] do
			input[1][b] = 0
		end

		input[#input + 1] = input[1]
	end

	-- pad the sides

	for a = 1, #input do
		for b = 1, paddingAmount[2] do
			table.insert(input[a], 1, 0)

			input[a][#input[a] + 1] = 0
		end
	end

	return input
end

function layersModule.zeroPad3D(args)
	local input, paddingAmount = args.input, args.paddingAmount

	for a = 1, #input do
		input[a] = layer.zeroPad2D{
			input = input[a],
			paddingAmount = {paddingAmount[2], paddingAmount[3]}
		}
	end

	for a = 1, paddingAmount[1] do
		table.insert(input, 1, {})

		for b = 1, #input[a + 1] do
			input[1][b] = {}
			for c = 1, #input[a + 1][b] do
				input[1][b][c] = 0
			end
		end

		input[#input + 1] = input[1]
	end

	return input
end

function layersModule.crop1D(args)
	local output = {}
	local input, start, outputShape = args.input, args.start, args.outputShape

	for a = 1, start[1], outputShape[1] + start[1] do
		output[#output + 1] = input[a]
	end

	return output
end

function layersModule.crop2D(args)
	local output = {}
	local input, start, outputShape = args.input, args.start, args.outputShape

	for a = 1, start[1], outputShape[1] + start[1] do
		local row = {}

		for b = 1, start[2], outputShape[2] + start[2] do
			row[#row + 1] = input[a][b]
		end

		output[#output + 1] = row
	end

	return output
end

function layersModule.crop3D(args)
	local output = {}
	local input, start, outputShape = args.input, args.start, args.outputShape

	for a = 1, start[1], outputShape[1] + start[1] do
		local crop = {}

		for b = 1, start[2], outputShape[2] + start[2] do
			local row = {}

			for c = 1, start[3], outputShape[3] + start[3] do
				row[#row + 1] = input[a][b][c]
			end

			crop[#crop + 1] = row
		end

		output[#output + 1] = crop
	end

	return output
end

function layersModule.convolutional1D(args)
	local output, startIndex = {}, 0
	local outputSize = math.floor((#args.input - #args.filter[1]) / args.stride[1]) + 1

	local activation = activationsModule[args.activation]
	local input, filter, stride, dilation, biases, alpha = args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha

	for a = 1, outputSize do
		local sum = 0

		for b = 1, #filter do
			for c = 1, #filter[b] do
				if c % dilation[1] == 0 then
					sum = sum + input[startIndex + c] * filter[b][c]
				end
			end
		end

		startIndex = startIndex + stride[1]

		if args.biases then
			output[a] = activation(sum + biases[a], false, alpha)
		else
			output[a] = activation(sum, false, alpha)
		end
	end

	return output
end

function layersModule.convolutional2D(args)
	local output, startIndexA = {}, 0
	local outputHeight, outputWidth = math.floor((#args.input - #args.filter[1]) / args.stride[1]) + 1, math.floor((#args.input[1] - #args.filter[1][1]) / args.stride[2]) + 1

	local activation = activationsModule[args.activation]
	local input, filter, stride, dilation, biases, alpha = args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha

	for a = 1, outputHeight do
		output[a] = {}
		local startIndexB = 0

		for b = 1,outputWidth do
			local sum = 0

			for c = 1, #filter do
				for d = 1, #filter[c] do
					for e = 1, #filter[c][d] do
						if d % dilation[1] == 0 and e % dilation[2] == 0 then
							sum = sum + input[startIndexA + d][startIndexB + e] * filter[c][d][e]
						end
					end
				end
			end

			startIndexB = startIndexB + stride[2]

			if args.biases then
				output[a][b] = activation(sum + biases[a][b], false, alpha)
			else
				output[a][b] = activation(sum, false, alpha)
			end
		end

		startIndexA = startIndexA + stride[1]
	end

	return output
end

function layersModule.convolutional3D(args)
	local output, startIndexA = {}, 0
	local outputDepth, outputHeight, outputWidth = math.floor((#args.input - #args.filter[1]) / args.stride[1]) + 1, math.floor((#args.input[1] - #args.filter[1][1]) / args.stride[2]) + 1, math.floor((#args.input[1][1] - #args.filter[1][1][1]) / args.stride[3]) + 1

	local activation = activationsModule[args.activation]
	local input, filter, stride, dilation, biases, alpha = args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha

	for a = 1, outputDepth do
		output[a] = {}
		local startIndexB = 0

		for b = 1, outputHeight do
			output[a][b] = {}
			local startIndexC = 0

			for c = 1, outputWidth do
				local sum = 0

				for d = 1, #filter do
					for e = 1, #filter[d] do
						for f = 1, #filter[d][e] do
							for g = 1, #filter[d][e][f] do
								if e % dilation[1] == 0 and f % dilation[2] == 0 and g % dilation[3] == 0 then
									sum = sum + input[startIndexA + e][startIndexB + f][startIndexC + g] * filter[d][e][f][g]
								end
							end
						end
					end
				end

				startIndexC = startIndexC + stride[3]

				if args.biases then
					output[a][b][c] = activation(sum + biases[a][b][c], false, alpha)
				else
					output[a][b][c] = activation(sum, false, alpha)
				end
			end

			startIndexB = startIndexB + stride[2]
		end

		startIndexA = startIndexA + stride[1]
	end

	return output
end

function layersModule.convolutionalTranspose1D(args)
	return layersModule.convolutional1D{
		input = layesModuler.zeroPad1D(args.input, args.paddingAmount),
		filter = args.filter,
		biases = args.biases,
		stride = args.stride,
		dilation = args.dilation,
		activation = args.activation,
		alpha = args.alpha
	}
end

function layersModule.convolutionalTranspose2D(args)
	return layersModule.convolutional2D{
		input = layersModule.zeroPad2D(args.input, args.paddingAmount),
		filter = args.filter,
		biases = args.biases,
		stride = args.stride,
		dilation = args.dilation,
		activation = args.activation,
		alpha = args.alpha
	}
end

function layersModule.convolutionalTranspose3D(args)
	return layersModule.convolutional3D{
		input = layersModule.zeroPad3D(args.input, args.paddingAmount),
		filter = args.filter,
		biases = args.biases,
		stride = args.stride,
		dilation = args.dilation,
		activation = args.activation,
		alpha = args.alpha
	}
end

function layersModule.convolutionalDepthwise1D(args)
	local output = {}
	local input, filter, biases, stride, dilation, activation, alpha = args.input, args.filter, args.biases, args.stride, args.dilation, args.activation, args.alpha

	for a = 1, #args.input do
		local tempArgs = {
			input = input[a],
			filter = filter[a],
			stride = stride,
			dilation = dilation,
			activation = activation,
			alpha = alpha
		}

		if args.biases then
			tempArgs.biases = biases[a]
		end

		output[a] = layersModule.convolutional1D(tempArgs)
	end

	return output
end

function layersModule.convolutionalDepthwise2D(args)
	local output = {}
	local input, filter, biases, stride, dilation, activation, alpha = args.input, args.filter, args.biases, args.stride, args.dilation, args.activation, args.alpha

	for a = 1, #args.input do
		local tempArgs = {
			input = input[a],
			filter = filter[a],
			stride = stride,
			dilation = dilation,
			activation = activation,
			alpha = alpha
		}

		if args.biases then
			tempArgs.biases = biases[a]
		end

		output[a] = layersModule.convolutional2D(tempArgs)
	end

	return output
end

--[[
function layer.convolutionalSeparable1D(args)
	local output = {}

	for a = 1, #args.filter do
		local tempArgs = {
			input = args.input,
			filter = args.filter[a],
			stride = args.stride,
			dilation = args.dilation,
			activation = args.activation,
			alpha = args.alpha
		}

		if args.biases then
			tempArgs.biases = args.biases[a]
		end

		output[a] = layer.convolutional1D(tempArgs)
	end

	return output
end

function layer.convolutionalSeparable2D(args)
	local output = {}

	for a = 1, #args.filter do
		local tempArgs = {
			input = args.input,
			filter = args.filter[a],
			dilation = args.dilation,
			activation = args.activation,
			alpha = args.alpha
		}

		if args.biases then
			tempArgs.biases = args.biases[a]
		end

		output[a] = layer.convolutional2D(tempArgs)
	end

	return output
end

function layer.convolutionalSeparable3D(args)
	local output = {}

	for a = 1, #args.filter do
		local tempArgs = {
			input = args.input,
			filter = args.filter[a],
			dilation = args.dilation,
			activation = args.activation,
			alpha = args.alpha
		}

		if args.biases then
			tempArgs.biases = args.biases[a]
		end

		output[a] = layer.convolutional3D(tempArgs)
	end

	return output
end

function layer.convolutionalDepthwiseSeparable1D(args)
	local output = {}

	for a = 1, #args.filter do
		local tempArgs = {
			input = args.input,
			filter = args.filter[a],
			dilation = args.dilation,
			activation = args.activation,
			alpha = args.alpha
		}

		if args.biases then
			tempArgs.biases = args.biases[a]
		end

		output[a] = layer.convolutionalDepthwise1D(tempArgs)
	end

	return output
end

function layer.convolutionalDepthwiseSeparable2D(args)
	local output = {}

	for a = 1, #args.filter do
		local tempArgs = {
			input = args.input,
			filter = args.filter[a],
			dilation = args.dilation,
			activation = args.activation,
			alpha = args.alpha
		}

		if args.biases then
			tempArgs.biases = args.biases[a]
		end

		output[a] = layer.convolutionalDepthwise2D(tempArgs)
	end

	return output
end
]]--

function layersModule.flatten(args)
	return arrayModule.flatten(args.input)
end

function layersModule.reshape(args)
	return arrayModule.reshape(args.input, args.shape)
end

function layersModule.minMaxNormalize1D(args)
	local max, min = arrayModule.math.max(args.input), arrayModule.math.min(args.input)
	local maxMinusMin = max - min
	local input = args.input

	for a = 1, #args.input do
		input[a] = input[a] - min / maxMinusMin
	end

	return input
end

function layersModule.minMaxNormalize2D(args)
	local max, min = arrayModule.math.max(args.input), arrayModule.math.min(args.input)
	local maxMinusMin = max - min
	local input = args.input

	for a = 1, #input do
		for b = 1, #input[a] do
			input[a][b] = input[a][b] - min / maxMinusMin
		end
	end

	return args.input
end

function layersModule.minMaxNormalize3D(args)
	local max, min = arrayModule.math.max(args.input), arrayModule.math.min(args.input)
	local maxMinusMin = max - min
	local input = args.input

	for a = 1, #input do
		for b = 1, #input[a] do
			for c = 1, #input[a][b] do
				input[a][b][c] = input[a][b][c] - min / maxMinusMin
			end
		end
	end

	return input
end

function layersModule.vectorAdd1D(args)
	local input, biases = args.input, args.biases

	for a = 1, #input do
		input[a] = input[a] + biases[a]
	end

	return input
end

function layersModule.vectorAdd2D(args)
	local input, biases = args.input, args.biases

	for a = 1, #input do
		for b = 1, #input[a] do
			input[a][b] = input[a][b] + biases[a][b]
		end
	end

	return input
end

function layersModule.vectorAdd3D(args)
	local input, biases = args.input, args.biases

	for a = 1, #input do
		for b = 1, #input[a] do
			for c = 1, #input[a][b] do
				input[a][b][c] = input[a][b][c] + biases[a][b][c]
			end
		end
	end

	return args.input
end

function layersModule.vectorSubtract1D(args)
	local input, biases = args.input, args.biases

	for a = 1, #input do
		input[a] = input[a] - biases[a]
	end

	return input
end

function layersModule.vectorSubtract2D(args)
	local input, biases = args.input, args.biases

	for a = 1, #input do
		for b = 1, #input[a] do
			input[a][b] = input[a][b] - biases[a][b]
		end
	end

	return input
end

function layersModule.vectorSubtract3D(args)
	local input, biases = args.input, args.biases

	for a = 1, #input do
		for b = 1, #input[a] do
			for c = 1, #input[a][b] do
				input[a][b][c] = input[a][b][c] - biases[a][b][c]
			end
		end
	end

	return input
end

function layersModule.dot1D(args)
	local input, weights = args.input, args.weights

	for a = 1, #input do
		input[a] = input[a] * weights[a]
	end

	return input
end

function layersModule.dot2D(args)
	local input, weights = args.input, args.weights

	for a = 1, #input do
		for b = 1, #input[a] do
			input[a][b] = input[a][b] * weights[a][b]
		end
	end

	return input
end

function layersModule.dot3D(args)
	local input, weights = args.input, args.weights

	for a = 1, #input do
		for b = 1, #input[a] do
			for c = 1, #input[a][b] do
				input[a][b][c] = input[a][b][c] * weights[a][b][c]
			end
		end
	end

	return args.input
end

function layersModule.vectorDivide1D(args)
	local input, weights = args.input, args.weights

	for a = 1, #input do
		input[a] = input[a] / weights[a]
	end

	return input
end

function layersModule.vectorDivide2D(args)
	local input, weights = args.input, args.weights

	for a = 1, #input do
		for b = 1, #input[a] do
			input[a][b] = input[a][b] / weights[a][b]
		end
	end

	return input
end

function layersModule.vectorDivide3D(args)
	local input, weights = args.input, args.weights

	for a = 1, #input do
		for b = 1, #input[a] do
			for c = 1, #input[a][b] do
				input[a][b][c] = input[a][b][c] / weights[a][b][c]
			end
		end
	end

	return args.input
end

function layersModule.add1D(args)
	local input, bias = args.input, args.bias

	for a = 1, #input do
		input[a] = input[a] + bias
	end

	return input
end

function layersModule.add2D(args)
	local input, bias = args.input, args.bias

	for a = 1, #input do
		for b = 1, #input[a] do
			input[a][b] = input[a][b] + bias
		end
	end

	return input
end

function layersModule.add3D(args)
	local input, bias = args.input, args.bias

	for a = 1, #input do
		for b = 1, #input[a] do
			for c = 1, #input[a][b] do
				input[a][b][c] = input[a][b][c] + bias
			end
		end
	end

	return input
end

function layersModule.subtract1D(args)
	local input, bias = args.input, args.bias

	for a = 1, #input do
		input[a] = input[a] - bias
	end

	return input
end

function layersModule.subtract2D(args)
	local input, bias = args.input, args.bias

	for a = 1, #input do
		for b = 1, #input[a] do
			input[a][b] = input[a][b] - bias
		end
	end

	return input
end

function layersModule.subtract3D(args)
	local input, bias = args.input, args.bias

	for a = 1, #input do
		for b = 1, #input[a] do
			for c = 1, #input[a][b] do
				input[a][b][c] = input[a][b][c] - bias
			end
		end
	end

	return input
end

function layersModule.multiply1D(args)
	local input, weight = args.input, args.bias

	for a = 1, #input do
		input[a] = input[a] * weight
	end

	return input
end

function layersModule.multiply2D(args)
	local input, weight = args.input, args.bias

	for a = 1, #input do
		for b = 1, #input[a] do
			input[a][b] = input[a][b] * weight
		end
	end

	return input
end

function layersModule.multiply3D(args)
	local input, weight = args.input, args.bias

	for a = 1, #input do
		for b = 1, #input[a] do
			for c = 1, #input[a][b] do
				input[a][b][c] = input[a][b][c] * weight
			end
		end
	end

	return input
end

function layersModule.divide1D(args)
	local input, weight = args.input, args.bias

	for a = 1, #input do
		input[a] = input[a] / weight
	end

	return input
end

function layersModule.divide2D(args)
	local input, weight = args.input, args.bias

	for a = 1, #input do
		for b = 1, #input[a] do
			input[a][b] = input[a][b] / weight
		end
	end

	return input
end

function layersModule.divide3D(args)
	local input, weight = args.input, args.bias

	for a = 1, #input do
		for b = 1, #input[a] do
			for c = 1, #input[a][b] do
				input[a][b][c] = input[a][b][c] / weight
			end
		end
	end

	return input
end

function layersModule.dropOut(args)
	return args.input -- dropout is only applied during training, hence why were just returning the input
end

function layersModule.uniformNoise(args)
	if args.backPropOnly then
		return
	end

	local input, lowerLimit, upperLimit = args.input, args.lowerLimit, args.upperLimit

	for a = 1, #input do
		if type(input[a]) == "table" then
			input[a] = layersModule.uniformNoise{
				input = input[a],
				lowerLimit = lowerLimit,
				upperLimit = upperLimit,
				backPropOnly = false
			}
		else
			input[a] = input[a] + mathModule.random.uniform(lowerLimit, upperLimit)
		end
	end

	return input
end

function layersModule.normalNoise(args)
	if args.backPropOnly then
		return
	end

	local input, sd, mean = args.input, args.sd, args.mean

	for a = 1, #args.input do
		if type(args.input[a]) == "table" then
			args.input[a] = layersModule.uniformNoise{
				input = input,
				sd = sd,
				mean = mean,
				backPropOnly = false
			}
		else
			input[a] = input[a] + mathModule.random.normal(sd, mean)
		end
	end

	return args.input
end

function layersModule.softmax(args)
	return activation.softmax(args.input)
end

function layersModule.activate(args)
	local activation = activationsModule[args.activation]

	local input, activation, derivative, alpha = args.input, args.activation, args.derivative, args.alpha

	for a = 1, #input do
		if type(input[a]) == "table" then
			input[a] = layersModule.activate{
				input = input[a],
				activation = activation,
				derivative = derivative,
				alpha = alpha
			}
		else
			input[a] = activation(input[a], derivative, alpha)
		end
	end

	return input
end

return layersModule
