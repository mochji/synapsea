--[[
	https://github.com/mochji/synapsea
	core/layers/build.lua

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

local buildModule = {
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

function layerBuildModule.dense(args)
	local layer, parameterBuild = {
		config = {
			activation = args.activation or "linear",
			outputSize = args.outputSize
		},
		parameters = {
			alpha = args.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape,
		outputShape = {args.outputSize}
	}, {}

	-- Initializers

	layer.initializer.weights = {
		layer.initializer = args.weightsInitializer or "constant",
		layer.parameters = args.weightsInitParameters or {value = 0}
	}

	if args.useBias then
		layer.initializer.bias = {
			layer.initializer = args.biasInitializer or "constant",
			layer.parameters = args.biasInitParameters or {value = 0}
		}
	end

	-- Trainable

	if args.weightsTrainable then
		layer.trainable.weights = true
	end

	if args.biasTrainable and args.useBias then
		layer.trainable.bias = true
	end

	-- Parameters

	parameterBuild.weights = {
		shape = {args.inputShape[1], args.outputSize}
	}

	if args.useBias then
		layer.parameters.bias = 0
	end

	return layer, parameterBuild
end

function layerBuildModule.averagePooling1D(args)
	local layer = {
		config = {
			kernel = args.kernel,
			stride = args.stride or {1},
			dilation = args.dilation or {1}
		},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1}
	}

	return layer
end

function layerBuildModule.averagePooling2D(args)
	local layer = {
		config = {
			kernel = args.kernel,
			stride = args.stride or {1, 1},
			dilation = args.dilation or {1, 1}
		},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] - args.kernel[2]) / args.stride[2]) + 1}
	}

	return layer
end

function layerBuildModule.averagePooling3D(args)
	local layer = {
		config = {
			kernel = args.kernel,
			stride = args.stride or {1, 1, 1},
			dilation = args.dilation or {1, 1, 1}
		},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] - args.kernel[2]) / args.stride[2]) + 1, math.floor((args.inputShape[3] - args.kernel[3]) / args.stride[3]) + 1}
	}

	return layer
end

layerBuildModule.maxPooling1D = layerBuildModule.averagePooling1D

layerBuildModule.maxPooling2D = layerBuildModule.averagePooling2D

layerBuildModule.maxPooling3D = layerBuildModule.averagePooling3D

layerBuildModule.sumPooling1D = layerBuildModule.averagePooling1D

layerBuildModule.sumPooling2D = layerBuildModule.averagePooling2D

layerBuildModule.sumPooling3D = layerBuildModule.averagePooling3D

function layerBuildModule.averageGlobalPooling1D(args)
	return {
		inputShape = args.inputShape,
		outputShape = {1}
	}
end

function layerBuildModule.averageGlobalPooling2D(args)
	return {
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1]}
	}
end

layerBuildModule.averageGlobalPooling3D = layerBuildModule.averageGlobalPooling2D

layerBuildModule.maxGlobalPooling1D = layerBuildModule.averageGlobalPooling1D

layerBuildModule.maxGlobalPooling2D = layerBuildModule.averageGlobalPooling2D

layerBuildModule.maxGlobalPooling3D = layerBuildModule.averageGlobalPooling3D

layerBuildModule.sumGlobalPooling1D = layerBuildModule.averageGlobalPooling1D

layerBuildModule.sumGlobalPooling2D = layerBuildModule.averageGlobalPooling2D

layerBuildModule.sumGlobalPooling3D = layerBuildModule.averageGlobalPooling3D

function layerBuildModule.upSample1D(args)
	return {
		config = {
			kernel = args.kernel
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] * args.kernel[1]}
	}
end

function layerBuildModule.upSample2D(args)
	return {
		config = {
			kernel = args.kernel
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] * args.kernel[1], args.inputShape[2] * args.kernel[2]}
	}
end

function layerBuildModule.upSample3D(args)
	return {
		config = {
			kernel = args.kernel
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] * args.kernel[1], args.inputShape[2] * args.kernel[2], args.inputShape[3] * args.kernel[3]}
	}
end

function layerBuildModule.zeroPad1D(args)
	return {
		config = {
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] + args.paddingAmount[1] * 2}
	}
end

function layerBuildModule.zeroPad2D(args)
	return {
		config = {
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] + args.paddingAmount[1] * 2, args.inputShape[2] + args.paddingAmount[2] * 2}
	}
end

function layerBuildModule.zeroPad3D(args)
	return {
		config = {
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] + args.paddingAmount[1] * 2, args.inputShape[2] + args.paddingAmount[2] * 2, args.inputShape[3] + args.paddingAmount[3] * 2}
	}
end

function layerBuildModule.crop1D(args)
	return {
		config = {
			start = args.start,
			outputShape = args.outputShape
		},
		inputShape = args.inputShape,
		outputShape = args.outputShape
	}
end

layerBuildModule.crop2D = layerBuildModule.crop1D

layerBuildModule.crop3D = layerBuildModule.crop1D

function layerBuildModule.convolutional1D(args)
	local layer, parameterBuild = {
		config = {
			activation = args.activation or "linear",
			stride = args.stride or {1},
			dilation = args.dilation or {1},
			filters = args.filters or 1
		},
		parameters = {
			alpha = args.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1}
	}, {}

	-- Initializers

	layer.initializer.filter = {
		initializer = args.filterInitializer or "constant",
		parameters = args.filterInitParameters or {value = 0}
	}

	if args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInitializer or "constant",
			parameters = args.biasesInitParameters or {value = 0}
		}
	end

	-- Trainable

	if args.filterTrainable then
		layer.trainable.filter = true
	end

	if args.biasesTrainable and args.useBias then
		layer.trainable.biases = true
	end

	-- Parameters

	parameterBuild.filter = {
		shape = {args.filters, args.kernel[1]}
	}

	if args.useBias then
		parameterBuild.biases = {
			shape = layer.outputShape
		}
	end

	return layer, parameterBuild
end

function layerBuildModule.convolutional2D(args)
	local layer, parameterBuild = {
		config = {
			activation = args.activation or "linear",
			stride = args.stride or {1, 1},
			dilation = args.dilation or {1, 1},
			filters = args.filters or 1
		},
		parameters = {
			alpha = args.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] - args.kernel[2]) / args.stride[2]) + 1}
	}, {}

	-- Initializers

	layer.initializer.filter = {
		initializer = args.filterInitializer or "constant",
		parameters = args.filterInitParameters or {value = 0}
	}

	if args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInitializer or "constant",
			parameters = args.biasesInitParameters or {value = 0}
		}
	end

	-- Trainable

	if args.filterTrainable then
		layer.trainable.filter = true
	end

	if args.biasesTrainable and args.useBias then
		layer.trainable.biases = true
	end

	-- Parameters

	parameterBuild.filter = {
		shape = {args.filters, args.kernel[1], args.kernel[2]}
	}

	if args.useBias then
		parameterBuild.biases = {
			shape = layer.outputShape
		}
	end

	return layer, parameterBuild
end

function layerBuildModule.convolutional3D(args)
	local layer, parameterBuild = {
		config = {
			activation = args.activation or "linear",
			stride = args.stride or {1, 1, 1},
			dilation = args.dilation or {1, 1, 1},
			filters = args.filters or 1
		},
		parameters = {
			alpha = args.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] - args.kernel[2]) / args.stride[2]) + 1, math.floor((args.inputShape[3] - args.kernel[3]) / args.stride[3]) + 1}
	}, {}

	-- Initializers

	layer.initializer.filter = {
		initializer = args.filterInitializer or "constant",
		parameters = args.filterInitParameters or {value = 0}
	}

	if args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInitializer or "constant",
			parameters = args.biasesInitParameters or {value = 0}
		}
	end

	-- Trainable

	if args.filterTrainable then
		layer.trainable.filter = true
	end

	if args.biasesTrainable and args.useBias then
		layer.trainable.biases = true
	end

	-- Parameters

	parameterBuild.filter = {
		shape = {args.filters, args.kernel[1], args.kernel[2], args.kernel[3]}
	}

	if args.useBias then
		parameterBuild.biases = {
			shape = layer.outputShape
		}
	end

	return layer, parameterBuild
end

function layerBuildModule.flatten(args)
	local outputShape = 1

	for a = 1, #args.inputShape do
		outputShape = outputShape * args.inputShape[a]
	end

	return {
		inputShape = args.inputShape,
		outputShape = {outputShape}
	}
end

function layerBuildModule.add1D(args)
	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}, {}

	-- Initializers

	layer.initializer.biases = {
		initializer = args.biasesInitializer or "constant",
		parameters = args.biasesInitParameters or {value = 0}
	}

	-- Trainable

	if args.biasTrainable then
		layer.trainable.bias = true
	end

	-- Parameters

	parameterBuild.biases = {
		shape = args.inputShape
	}

	return layer, parameterBuild
end

layerBuildModule.add2D = layerBuildModule.add1D

layerBuildModule.add3D = layerBuildModule.add1D

layerBuildModule.subtract1D = layerBuildModule.add1D

layerBuildModule.subtract2D = layerBuildModule.add1D

layerBuildModule.subtract3D = layerBuildModule.add1D

function layerBuildModule.multiply1D(args)
	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}, {}

	-- Initializers

	layer.initializer.weights = {
		initializer = args.weightsInitializer or "constant",
		parameters = args.weightsInitParameters or {value = 0}
	}

	-- Trainable

	if args.biasTrainable then
		layer.trainable.bias = true
	end

	-- Parameters

	parameterBuild.weights = {
		shape = args.inputShape
	}

	return layer, parameterBuild
end

layerBuildModule.multiply2D = layerBuildModule.multiply1D

layerBuildModule.multiply3D = layerBuildModule.multiply1D

layerBuildModule.vectorDivide1D = layerBuildModule.multiply1D

layerBuildModule.divide2D = layerBuildModule.multiply1D

layerBuildModule.divide3D = layerBuildModule.multiply1D

function layerBuildModule.softmax(args)
	return {
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}
end

function layerBuildModule.activate(args)
	return {
		config = {
			activation = args.activation,
			derivative = args.derivative
		},
		parameters = {
			alpha = args.alpha
		},
		inputShape = args.inputShape,
		outputShape = args.outputShape
	}
end


function layerBuildModule.dropOut(args)
	return {
		config = {
			rate = args.rate
		},
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}
end

function layerBuildModule.uniformNoise(args)
	return {
		config = {
			lowerLimit = args.lowerLimit,
			upperLimit = args.upperLimit
		},
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}
end

function layerBuildModule.normalNoise(args)
	return {
		config = {
			mean = args.mean,
			sd = args.sd,
		},
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}
end

return layerBuildModule
