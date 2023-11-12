--[[
	https://github.com/mochji/synapsea
	core/layers/layers.lua

	Synapsea, a simple yet powerful machine learning library made in pure Lua.
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

local activationsModule = require(_SYNAPSEA_PATH .. "core.activations")
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
	flatten,
	reshape,
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
	softmax,
	activate,
	dropOut
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

		startIndex = startIndex + stride[1]

		output[a] = sum / kernel[1]
	end

	return output
end

function layersModule.averagePooling2D(args)
	local output, startIndexA, kernelProduct = {}, 0, syntable.product(args.kernel)
	local outputHeight, outputWidth = math.floor((#args.input - args.kernel[1]) / args.stride[1]) + 1, math.floor((#args.input[1] - args.kernel[2]) / args.stride[2]) + 1

	local input, kernel, stride, dilation = args.input, args.kernel, args.stride, args.dilation

	for a = 1, outputHeight do
		output[a] = {}
		local startIndexB = 0

		for b = 1, outputWidth do
			local sum = 0

			for c = 1, kernel[1] do
				for d = 1, kernel[2] do
					if c % dilation[1] == 0 and d % dilation[2] == 1 then
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

		for b = 1, #kernel[1] do
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
					if c % dilation[1] == 0 and d % dilation[2] == 1 then
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
	local sum = 0

	local input = args.input

	for a = 1, #input do
		sum = sum + input[a]
	end

	return {sum / #input}
end

function layersModule.averageGlobalPooling2D(args)
	local output = {}

	local input = args.input

	for a = 1, #input do
		output[a] = 0

		for b = 1, #input[a] do
			output[a] = output[a] + input[a][b]
		end

		output[a] = output[a] / #input[a]
	end

	return output
end

function layersModule.averageGlobalPooling3D(args)
	local output = {}

	local input = args.input

	for a = 1, #input do
		output[a] = {}

		for b = 1, #input[a] do
			output[a][b] = 0

			for c = 1, #input[a][b] do
				output[a][b] = output[a][b] + input[a][b][c]
			end

			output[a][b] = output[a][b] / #input[a][b]
		end
	end

	return output
end

function layersModule.maxGlobalPooling1D(args)
	return {math.max(table.unpack(args.input))}
end

function layersModule.maxGlobalPooling2D(args)
	local output = {}

	local input = args.input

	for a = 1, #input do
		output[a] = math.max(table.unpack(input[a]))
	end

	return output
end

function layersModule.maxGlobalPooling3D(args)
	local output = {}

	local input = args.input

	for a = 1, #input do
		output[a] = {}

		for b = 1, #input[a] do
			output[a][b] = math.max(table.unpack(input[a][b]))
		end
	end

	return output
end

function layersModule.sumGlobalPooling1D(args)
	local output = 0

	local input = args.input

	for a = 1, #input do
		output = output + input[a]
	end

	return {output}
end

function layersModule.sumGlobalPooling2D(args)
	local output = {}

	local input = args.input

	for a = 1, #input do
		output[a] = 0

		for b = 1, #input[a] do
			output[a] = output[a] + input[a][b]
		end
	end

	return input
end

function layersModule.sumGlobalPooling3D(args)
	local output = {}

	local input = args.input

	for a = 1, #input do
		output[a] = {}

		for b = 1, #input[a] do
			output[a][b] = 0

			for c = 1, #input[a][b] do
				output[a][b] = output[a][b] + input[a][b][c]
			end
		end
	end

	return output
end

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

	-- Pad the tops and bottoms

	for a = 1, paddingAmount[1] do
		table.insert(input, 1, {})

		for b = 1, #input[a + 1] do
			input[1][b] = 0
		end

		input[#input + 1] = input[1]
	end

	-- Pad the sides

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
	local output = {}
	local outputSize = math.floor((#args.input - #args.filter[1]) / args.stride[1]) + 1

	local activation = activationsModule[args.activation]
	local input, filter, stride, dilation, biases, alpha = args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha

	-- Check whether to do a normal 1D convolution or do a 2D convolution (depth from filters)

	if type(input[1]) == "table" then
		for a = 1, #input do
			output[a] = layersModule.convolutional1D{
				input = input[a],
				filter = filter,
				stride = stride,
				dilation = dilation,
				biases = biases,
				activation = activation,
				alpha = alpha
			}
		end
	else
		for a = 1, #filter do
			local startIndex = 0
			output[a] = {}

			for b = 1, outputSize do
				local sum = 0

				for c = 1, #filter[a] do
					if c % dilation[1] == 0 then
						sum = sum + input[startIndex + c] * filter[a][c]
					end
				end

				startIndex = startIndex + stride[1]

				if biases then
					output[a][b] = activation(sum + biases[b], false, alpha)
				else
					output[a][b] = activation(sum, false, alpha)
				end
			end
		end
	end

	if #filter == 1 then
		return output[1]
	end

	return output
end

function layersModule.convolutional2D(args)
	local output = {}
	local outputShape = {math.floor((#args.input - #args.filter[1]) / args.stride[1]) + 1, math.floor((#args.input[1] - #args.filter[1][1]) / args.stride[2]) + 1}

	local activation = activationsModule[args.activation]
	local input, filter, stride, dilation, biases, alpha = args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha

	-- Check whether to do a normal 2D convolution or do a 3D convolution (depth from filters)

	if type(input[1][1]) == "table" then
		for a = 1, #input do
			output[a] = layersModule.convolutional2D{
				input = input[a],
				filter = filter,
				stride = stride,
				dilation = dilation,
				biases = biases,
				activation = activation,
				alpha = alpha
			}
		end
	else
		for a = 1, #filter do
			local startIndexA = 0
			output[a] = {}

			for b = 1, outputShape[1] do
				local startIndexB = 0
				output[a][b] = {}

				for c = 1, outputShape[2] do
					local sum = 0

					for d = 1, #filter[a] do
						for e = 1, #filter[a][d] do
							if d % dilation[1] == 0 and e % dilation[2] == 0 then
								sum = sum + input[startIndexA + d][startIndexB + e] * filter[a][d][e]
							end
						end
					end

					startIndexB = startIndexB + stride[2]

					if biases then
						output[a][b][c] = activation(sum + biases[b][c], false, alpha)
					else
						output[a][b][c] = activation(sum, false, alpha)
					end
				end

				startIndexA = startIndexA + 1
			end
		end
	end

	if #filter == 1 then
		return output[1]
	end

	return output
end

function layersModule.convolutional3D(args)
	local output = {}
	local outputShape = {math.floor((#args.input - #args.filter[1]) / args.stride[1]) + 1, math.floor((#args.input[1] - #args.filter[1][1]) / args.stride[2]) + 1, math.floor((#args.input - #args.filter[1][1][1]) / args.stride[3]) + 1}

	local activation = activationsModule[args.activation]
	local input, filter, stride, dilation, biases, alpha = args.input, args.filter, args.stride, args.dilation, args.biases, args.alpha

	-- Check whether to do a normal 3D convolution or do a 4D convolution (depth from filters)

	if type(input[1][1][1]) == "table" then
		for a = 1, #input do
			output[a] = layersModule.convolutional3D{
				input = input[a],
				filter = filter,
				stride = stride,
				dilation = dilation,
				biases = biases,
				activation = activation,
				alpha = alpha
			}
		end
	else
		for a = 1, #filter do
			local startIndexA = 0
			output[a] = {}

			for b = 1, outputShape[1] do
				local startIndexB = 0
				output[a][b] = {}

				for c = 1, outputShape[2] do
					local startIndexC = 0
					output[a][b][c] = {}

					for d = 1, outputShape[3] do
						local sum = 0

						for e = 1, #filter[a] do
							for f = 1, #filter[a][e] do
								for g = 1, #filter[a][e][f] do
									if e % dilation[1] == 0 and f % dilation[2] == 0 and g % dilation[3] == 0 then
										sum = sum + input[startIndexA + e][startIndexB + f][startIndexC + g] * filter[a][e][f][g]
									end
								end
							end
						end

						startIndexC = startIndexC + stride[3]

						if biases then
							output[a][b][c][d] = activation(sum + biases[b][c][d], false, alpha)
						else
							output[a][b][c][d] = activation(sum, false, alpha)
						end
					end

					startIndexB = startIndexB + 1
				end

				startIndexA = startIndexA + 1
			end
		end
	end

	if #filter == 1 then
		return output[1]
	end

	return output
end

function layersModule.flatten(args)
	local output = {}

	local a = 1

	for _, v in next, args.input do
		output[a] = v
	end

	return output
end

function layersModule.reshape(args)
end

function layersModule.add1D(args)
	local input, biases = args.input, args.biases

	for a = 1, #input do
		input[a] = input[a] + biases[a]
	end

	return input
end

function layersModule.add2D(args)
	local input, biases = args.input, args.biases

	for a = 1, #input do
		for b = 1, #input[a] do
			input[a][b] = input[a][b] + biases[a][b]
		end
	end

	return input
end

function layersModule.add3D(args)
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

function layersModule.subtract1D(args)
	local input, biases = args.input, args.biases

	for a = 1, #input do
		input[a] = input[a] - biases[a]
	end

	return input
end

function layersModule.subtract2D(args)
	local input, biases = args.input, args.biases

	for a = 1, #input do
		for b = 1, #input[a] do
			input[a][b] = input[a][b] - biases[a][b]
		end
	end

	return input
end

function layersModule.subtract3D(args)
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

function layersModule.multiply1D(args)
	local input, weights = args.input, args.weights

	for a = 1, #input do
		input[a] = input[a] * weights[a]
	end

	return input
end

function layersModule.multiply2D(args)
	local input, weights = args.input, args.weights

	for a = 1, #input do
		for b = 1, #input[a] do
			input[a][b] = input[a][b] * weights[a][b]
		end
	end

	return input
end

function layersModule.multiply3D(args)
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

function layersModule.divide1D(args)
	local input, weights = args.input, args.weights

	for a = 1, #input do
		input[a] = input[a] / weights[a]
	end

	return input
end

function layersModule.divide2D(args)
	local input, weights = args.input, args.weights

	for a = 1, #input do
		for b = 1, #input[a] do
			input[a][b] = input[a][b] / weights[a][b]
		end
	end

	return input
end

function layersModule.divide3D(args)
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

function layersModule.dropOut(args)
	return args.input
end

return layersModule
